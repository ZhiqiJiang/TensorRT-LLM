import json
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields
from enum import Enum, EnumMeta
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Literal, Optional, Union

import torch
import yaml
from pydantic import BaseModel, Field, PrivateAttr, validator
from strenum import StrEnum
from transformers import PreTrainedTokenizerBase

from tensorrt_llm.lora_manager import (LoraConfig,
                                       get_default_trtllm_modules_to_hf_modules)

from .._utils import mpi_rank
from ..auto_parallel import AutoParallelConfig, infer_cluster_config

# yapf: disable
# isort: off
from ..bindings.executor import (
                                 BatchingType as _BatchingType,
                                 CacheTransceiverConfig as _CacheTransceiverConfig,
                                 CapacitySchedulerPolicy as _CapacitySchedulerPolicy,
                                 ContextChunkingPolicy as _ContextChunkingPolicy,
                                 DecodingConfig,
                                 DecodingMode,
                                 DynamicBatchConfig as _DynamicBatchConfig,
                                 EagleConfig as _EagleConfig,
                                 ExecutorConfig as _ExecutorConfig,
                                 ExtendedRuntimePerfKnobConfig as _ExtendedRuntimePerfKnobConfig,
                                 KvCacheConfig as _KvCacheConfig,
                                 LookaheadDecodingConfig as _LookaheadDecodingConfig,
                                 PeftCacheConfig as _PeftCacheConfig,
                                 SchedulerConfig as _SchedulerConfig) # isort: skip
# isort: on
# yapf: enable
from ..builder import BuildConfig, EngineConfig
from ..logger import logger
from ..mapping import Mapping
from ..models.automodel import AutoConfig
from ..models.modeling_utils import (PretrainedConfig, QuantAlgo, QuantConfig,
                                     SpeculativeDecodingMode)
from ..sampling_params import BatchedLogitsProcessor
from .build_cache import BuildCacheConfig
from .tokenizer import TokenizerBase, tokenizer_factory
from .utils import (generate_api_docs_as_docstring, get_type_repr,
                    print_traceback_on_error)

# TODO[chunweiy]: move the following symbols back to utils scope, and remove the following import


@dataclass
class _ParallelConfig:
    ''' The model distribution configs for LLM.  '''
    tp_size: int = 1
    pp_size: int = 1
    cp_size: int = 1
    gpus_per_node: int = 8
    moe_cluster_size: int = 1
    moe_tp_size: int = 1
    moe_ep_size: int = 1
    cp_config: dict = field(default_factory=dict)
    enable_attention_dp: bool = False
    auto_parallel: bool = False

    _world_size: int = field(default=1, init=False)
    _devices: Optional[List[int]] = field(default=None, init=False)

    @property
    def devices(self) -> List[int]:
        if self._devices is None:
            return list(range(self.world_size))
        return self._devices

    @devices.setter
    def devices(self, devices: List[int]):
        if len(devices) != self.world_size:
            raise ValueError(
                f"devices {devices} should have the same length as world_size {self.world_size}"
            )
        self._devices = devices

    @property
    def world_size(self) -> bool:

        if self.auto_parallel:
            if self.tp_size > 1 or self.pp_size > 1 or self.cp_size > 1:
                raise RuntimeError(
                    "manually TP and PP are not supported in auto parallel mode."
                )
            return self._world_size

        if self._world_size > 1:
            raise RuntimeError(
                "world_size > 1 is only supported in auto parallel mode.")
        return self.tp_size * self.pp_size * self.cp_size

    @property
    def world_size_per_node(self) -> int:
        world_size = self.world_size
        total_nodes = math.ceil(world_size / self.gpus_per_node)
        return world_size // total_nodes  #TODO is this right?

    @world_size.setter
    def world_size(self, world_size: int):
        if self.auto_parallel:
            self._world_size = world_size
        elif (not self.auto_parallel
              ) and world_size != self.tp_size * self.pp_size * self.cp_size:
            raise ValueError(
                f"world_size {world_size} should be equal to tp_size * pp_size {self.tp_size * self.pp_size * self.cp_size} "
            )

    @property
    def is_multi_gpu(self) -> bool:
        return self.world_size > 1

    def to_mapping(self) -> Mapping:
        return Mapping(world_size=self.world_size,
                       rank=mpi_rank(),
                       gpus_per_node=self.gpus_per_node,
                       tp_size=self.tp_size,
                       pp_size=self.pp_size,
                       cp_size=self.cp_size,
                       cp_config=self.cp_config,
                       enable_attention_dp=self.enable_attention_dp,
                       moe_cluster_size=self.moe_cluster_size,
                       moe_tp_size=self.moe_tp_size,
                       moe_ep_size=self.moe_ep_size,
                       auto_parallel=self.auto_parallel)


class CalibConfig(BaseModel):
    """
    Calibration configuration.
    """
    device: Literal['cuda',
                    'cpu'] = Field(default='cuda',
                                   description="The device to run calibration.")
    calib_dataset: str = Field(
        default='cnn_dailymail',
        description="The name or local path of calibration dataset.")
    calib_batches: int = Field(
        default=512,
        description="The number of batches that the calibration runs.")
    calib_batch_size: int = Field(
        default=1, description="The batch size that the calibration runs.")
    calib_max_seq_length: int = Field(
        default=512,
        description="The maximum sequence length that the calibration runs.")
    random_seed: int = Field(
        default=1234, description="The random seed used for calibration.")
    tokenizer_max_seq_length: int = Field(
        default=2048,
        description=
        "The maximum sequence length to initialize tokenizer for calibration.")

    @classmethod
    def from_dict(cls, config: dict) -> 'CalibConfig':
        """Create a CalibConfig instance from a dict.

        Args:
            config (dict): The dict used to create CalibConfig.

        Returns:
            tensorrt_llm.llmapi.CalibConfig: The CalibConfig created from dict.
        """
        return cls(**config)

    def to_dict(self) -> dict:
        """Dump a CalibConfig instance to a dict.

        Returns:
            dict: The dict dumped from CalibConfig.
        """
        return self.model_dump()


class _ModelFormatKind(Enum):
    HF = 0
    TLLM_CKPT = 1
    TLLM_ENGINE = 2


class DecodingBaseConfig(BaseModel):
    max_draft_len: Optional[int] = None
    speculative_model: Optional[Union[str, Path]] = None

    @classmethod
    def from_dict(cls, data: dict):
        # dispatch to the correct decoding config
        decoding_type = data.get("decoding_type")
        config_classes = {
            "MTP": MTPDecodingConfig,
            "Medusa": MedusaDecodingConfig,
            "Eagle": EagleDecodingConfig,
            "Lookahead": LookaheadDecodingConfig,
            "NGram": NGramDecodingConfig,
        }

        config_class = config_classes.get(decoding_type)
        if config_class is None:
            raise ValueError(f"Invalid decoding type: {decoding_type}")

        return config_class(**data)

    def _check_fields(self):
        pass


class MedusaDecodingConfig(DecodingBaseConfig):
    medusa_choices: Optional[List[List[int]]] = None
    num_medusa_heads: Optional[int] = None

    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)

    decoding_type: ClassVar[str] = "Medusa"


class EagleDecodingConfig(DecodingBaseConfig):
    eagle_choices: Optional[List[List[int]]] = None
    greedy_sampling: Optional[bool] = True
    posterior_threshold: Optional[float] = None
    use_dynamic_tree: Optional[bool] = False
    dynamic_tree_max_topK: Optional[int] = None
    num_eagle_layers: Optional[int] = None
    max_non_leaves_per_layer: Optional[int] = None
    pytorch_eagle_weights_path: Optional[str] = None

    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)

    decoding_type: ClassVar[str] = "Eagle"


class NGramDecodingConfig(DecodingBaseConfig):
    """
    Configuration for NGram drafter speculative decoding.

    Arguments:
        prompt_lookup_num_tokens: int
                The length maximum of draft tokens (can be understood as length maximum of output draft tokens).

        max_matching_ngram_size: int
            The length maximum of searching tokens (can be understood as length maximum of input tokens to search).

        is_keep_all: bool = True
            Whether to keep all candidate pattern-matches pairs, only one match is kept for each pattern if False.

        is_use_oldest: bool = True
            Whether to provide the oldest match when pattern is hit, the newest one is provided if False.

        is_public_pool: bool = True
            Whether to use a common pool for all requests, or the pool is private for each request if False.
    """

    prompt_lookup_num_tokens: int = 2
    max_matching_ngram_size: int = 4
    is_keep_all: bool = True
    is_use_oldest: bool = True
    is_public_pool: bool = True

    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)

    decoding_type: ClassVar[str] = "NGram"


class MTPDecodingConfig(DecodingBaseConfig):
    num_nextn_predict_layers: Optional[int] = 1
    use_relaxed_acceptance_for_thinking: Optional[bool] = False
    relaxed_topk: Optional[int] = 1
    relaxed_delta: Optional[float] = 0.

    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)

    decoding_type: ClassVar[str] = "MTP"


class PybindMirror(ABC):
    ''' A class containing the utilities for mirroring Python classes to
    pybinding classes.
    '''

    @abstractmethod
    def _to_pybind(self):
        pass

    @staticmethod
    def maybe_to_pybind(ins):
        if isinstance(
                ins,
                PybindMirror) or type(ins).__class__ == PybindMirrorEnumMeta:
            return ins._to_pybind()
        return ins

    @staticmethod
    def mirror_pybind_fields(pybind_class):
        """
        Class decorator that ensures Python class fields mirror those of a C++ class.

        Args:
            pybind_class: The C++ class whose fields should be mirrored

        Returns:
            A decorator function that validates field mirroring
        """

        def decorator(cls):
            assert issubclass(cls, BaseModel)
            # Get all non-private fields from the C++ class
            cpp_fields = PybindMirror.get_pybind_variable_fields(pybind_class)
            python_fields = set(cls.model_fields.keys())

            # Check if all C++ fields exist in the Python class
            for field in cpp_fields:
                if field not in python_fields:
                    raise ValueError(
                        f"Field {field} is not mirrored in Python class {cls.__name__} from C++ class {pybind_class.__name__}. Please update the class."
                    )

            # Return the original class
            return cls

        return decorator

    @staticmethod
    def get_pybind_enum_fields(pybind_class):
        ''' Get all the enum fields from the pybind class. '''
        return [
            f for f in pybind_class.__members__.keys()
            if not f.startswith('_') and not callable(getattr(pybind_class, f))
        ]

    @staticmethod
    def mirror_pybind_enum(pybind_class):
        ''' Mirror the enum fields from the pybind class to the Python class. '''

        def decorator(cls):
            assert issubclass(cls, Enum)
            cpp_fields = PybindMirror.get_pybind_enum_fields(pybind_class)
            python_fields = set(cls.__members__.keys())

            for field in cpp_fields:
                if field not in python_fields:
                    raise ValueError(
                        f"Field {field} is not mirrored in Python class {cls.__name__} from C++ class {pybind_class.__name__}. Please update the class."
                    )
            return cls

        return decorator

    @staticmethod
    def get_pybind_variable_fields(config_cls):
        ''' Get all the variable fields from the pybind class. '''
        return [
            f for f in dir(config_cls)
            if not f.startswith('_') and not callable(getattr(config_cls, f))
        ]

    @staticmethod
    def pybind_equals(obj0, obj1):
        ''' Check if two pybind objects are equal. '''
        assert type(obj0) is type(obj1)
        for field in PybindMirror.get_pybind_variable_fields(type(obj0)):
            if getattr(obj0, field) != getattr(obj1, field):
                return False
        return True


class PybindMirrorMeta(type(PybindMirror)):
    pass


class PybindMirrorEnumMeta(EnumMeta, PybindMirrorMeta):
    """
    Combined metaclass for Enum and PybindMirror.  This is crucial.
    """


@PybindMirror.mirror_pybind_enum(_BatchingType)
class BatchingType(StrEnum, metaclass=PybindMirrorEnumMeta):
    STATIC = "STATIC"
    INFLIGHT = "INFLIGHT"

    def _to_pybind(self):
        return getattr(_BatchingType, self.value)


@PybindMirror.mirror_pybind_enum(_CapacitySchedulerPolicy)
class CapacitySchedulerPolicy(StrEnum, metaclass=PybindMirrorEnumMeta):
    MAX_UTILIZATION = "MAX_UTILIZATION"
    GUARANTEED_NO_EVICT = "GUARANTEED_NO_EVICT"
    STATIC_BATCH = "STATIC_BATCH"

    def _to_pybind(self):
        return getattr(_CapacitySchedulerPolicy, self.value)


@PybindMirror.mirror_pybind_enum(_ContextChunkingPolicy)
class ContextChunkingPolicy(StrEnum, metaclass=PybindMirrorEnumMeta):
    ''' Context chunking policy. '''
    FIRST_COME_FIRST_SERVED = "FIRST_COME_FIRST_SERVED"
    EQUAL_PROGRESS = "EQUAL_PROGRESS"

    def _to_pybind(self):
        return getattr(_ContextChunkingPolicy, self.value)


@PybindMirror.mirror_pybind_fields(_DynamicBatchConfig)
class DynamicBatchConfig(BaseModel, PybindMirror):
    """Dynamic batch configuration.

    Controls how batch size and token limits are dynamically adjusted at runtime.
    """
    enable_batch_size_tuning: bool = Field(
        description="Controls if the batch size should be tuned dynamically")

    enable_max_num_tokens_tuning: bool = Field(
        description="Controls if the max num tokens should be tuned dynamically"
    )

    dynamic_batch_moving_average_window: int = Field(
        description=
        "The window size for moving average of input and output length which is used to calculate dynamic batch size and max num tokens"
    )

    def _to_pybind(self):
        return _DynamicBatchConfig(
            enable_batch_size_tuning=self.enable_batch_size_tuning,
            enable_max_num_tokens_tuning=self.enable_max_num_tokens_tuning,
            dynamic_batch_moving_average_window=self.
            dynamic_batch_moving_average_window)


@PybindMirror.mirror_pybind_fields(_SchedulerConfig)
class SchedulerConfig(BaseModel, PybindMirror):
    capacity_scheduler_policy: CapacitySchedulerPolicy = Field(
        default=CapacitySchedulerPolicy.GUARANTEED_NO_EVICT,
        description="The capacity scheduler policy to use")

    context_chunking_policy: Optional[ContextChunkingPolicy] = Field(
        default=None, description="The context chunking policy to use")

    dynamic_batch_config: Optional[DynamicBatchConfig] = Field(
        default=None, description="The dynamic batch config to use")

    def _to_pybind(self):
        return _SchedulerConfig(
            capacity_scheduler_policy=self.capacity_scheduler_policy._to_pybind(
            ),
            context_chunking_policy=self.context_chunking_policy._to_pybind()
            if self.context_chunking_policy else None,
            dynamic_batch_config=self.dynamic_batch_config._to_pybind()
            if self.dynamic_batch_config else None)


@PybindMirror.mirror_pybind_fields(_PeftCacheConfig)
class PeftCacheConfig(BaseModel, PybindMirror):
    """
    Configuration for the PEFT cache.
    """
    num_host_module_layer: int = Field(
        default=0,
        description=
        "number of max sized 1-layer 1-module adapterSize=1 sets of weights that can be stored in host cache"
    )
    num_device_module_layer: int = Field(
        default=0,
        description=
        "number of max sized 1-layer 1-module sets of weights that can be stored in host cache"
    )
    optimal_adapter_size: int = Field(
        default=
        8,  # There are tests to keep the default value consistent with the pybind default value
        description="optimal adapter size used to set page width")
    max_adapter_size: int = Field(
        default=64,
        description="max supported adapter size. Used to compute minimum")
    num_put_workers: int = Field(
        default=1,
        description=
        "number of worker threads used to put weights into host cache")
    num_ensure_workers: int = Field(
        default=1,
        description=
        "number of worker threads used to copy weights from host to device")
    num_copy_streams: int = Field(
        default=1,
        description="number of streams used to copy weights from host to device"
    )
    max_pages_per_block_host: int = Field(
        default=24,
        description="Number of cache pages per allocation block (host)")
    max_pages_per_block_device: int = Field(
        default=8,
        description="Number of cache pages per allocation block (device)")
    device_cache_percent: Optional[float] = Field(
        default=None,
        description="percent of memory after engine load to use for cache")
    host_cache_size: Optional[int] = Field(
        default=None, description="size in bytes to use for host cache")
    lora_prefetch_dir: Optional[str] = Field(
        default=None,
        description=
        "folder to store the LoRA weights we hope to load during engine initialization"
    )

    def _to_pybind(self):
        return _PeftCacheConfig(
            num_host_module_layer=self.num_host_module_layer,
            num_device_module_layer=self.num_device_module_layer,
            optimal_adapter_size=self.optimal_adapter_size,
            max_adapter_size=self.max_adapter_size,
            num_put_workers=self.num_put_workers,
            num_ensure_workers=self.num_ensure_workers,
            num_copy_streams=self.num_copy_streams,
            max_pages_per_block_host=self.max_pages_per_block_host,
            max_pages_per_block_device=self.max_pages_per_block_device,
            device_cache_percent=self.device_cache_percent,
            host_cache_size=self.host_cache_size,
            lora_prefetch_dir=self.lora_prefetch_dir)


@PybindMirror.mirror_pybind_fields(_LookaheadDecodingConfig)
class LookaheadDecodingConfig(DecodingBaseConfig, PybindMirror):
    """
    Configuration for lookahead speculative decoding.
    """

    max_window_size: int = Field(
        default=_LookaheadDecodingConfig.get_default_lookahead_decoding_window(
        ),
        description="Number of NGrams in lookahead branch per step.")
    max_ngram_size: int = Field(
        default=_LookaheadDecodingConfig.get_default_lookahead_decoding_ngram(),
        description="Number of tokens per NGram.")
    max_verification_set_size: int = Field(
        default=_LookaheadDecodingConfig.
        get_default_lookahead_decoding_verification_set(),
        description="Number of NGrams in verification branch per step.")

    @validator('max_window_size', 'max_ngram_size', 'max_verification_set_size')
    def validate_positive_values(cls, v):
        if v <= 0:
            raise ValueError(f"Value must be positive, got {v}")
        return v

    def __init__(self, **data):
        super().__init__(**data)
        self._check_fields()

    def calculate_speculative_resource(self):
        return _LookaheadDecodingConfig.calculate_speculative_resource_tuple(
            self.max_window_size, self.max_ngram_size,
            self.max_verification_set_size)

    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)

    def _to_pybind(self):
        return _LookaheadDecodingConfig(self.max_window_size,
                                        self.max_ngram_size,
                                        self.max_verification_set_size)

    decoding_type: ClassVar[str] = "Lookahead"


@PybindMirror.mirror_pybind_fields(_KvCacheConfig)
class KvCacheConfig(BaseModel, PybindMirror):
    """
    Configuration for the KV cache.
    """
    enable_block_reuse: bool = Field(
        default=True,
        description=
        "Controls if KV cache blocks can be reused for different requests.")
    max_tokens: Optional[int] = Field(
        default=None,
        description=
        "The maximum number of tokens that should be stored in the KV cache. If both `max_tokens` and `free_gpu_memory_fraction` are specified, memory corresponding to the minimum will be used."
    )
    max_attention_window: Optional[List[int]] = Field(
        default=None,
        description=
        "Size of the attention window for each sequence. Only the last tokens will be stored in the KV cache. If the number of elements in `max_attention_window` is less than the number of layers, `max_attention_window` will be repeated multiple times to the number of layers."
    )
    sink_token_length: Optional[int] = Field(
        default=None,
        description=
        "Number of sink tokens (tokens to always keep in attention window).")
    free_gpu_memory_fraction: Optional[float] = Field(
        default=None,
        description=
        "The fraction of GPU memory fraction that should be allocated for the KV cache. Default is 90%. If both `max_tokens` and `free_gpu_memory_fraction` are specified, memory corresponding to the minimum will be used."
    )
    host_cache_size: Optional[int] = Field(
        default=None,
        description=
        "Size of the host cache in bytes. If both `max_tokens` and `host_cache_size` are specified, memory corresponding to the minimum will be used."
    )
    onboard_blocks: bool = Field(
        default=True, description="Controls if blocks are onboarded.")
    cross_kv_cache_fraction: Optional[float] = Field(
        default=None,
        description=
        "The fraction of the KV Cache memory should be reserved for cross attention. If set to p, self attention will use 1-p of KV Cache memory and cross attention will use p of KV Cache memory. Default is 50%. Should only be set when using encoder-decoder model."
    )
    secondary_offload_min_priority: Optional[int] = Field(
        default=None,
        description=
        "Only blocks with priority > mSecondaryOfflineMinPriority can be offloaded to secondary memory."
    )
    event_buffer_max_size: int = Field(
        default=0,
        description=
        "Maximum size of the event buffer. If set to 0, the event buffer will not be used."
    )
    enable_partial_reuse: bool = Field(
        default=True,
        description=
        "Whether blocks that are only partially matched can be reused.")
    copy_on_partial_reuse: bool = Field(
        default=True,
        description=
        "Whether partially matched blocks that are in use can be reused after copying them."
    )

    def _to_pybind(self):
        return _KvCacheConfig(
            enable_block_reuse=self.enable_block_reuse,
            max_tokens=self.max_tokens,
            max_attention_window=self.max_attention_window,
            sink_token_length=self.sink_token_length,
            free_gpu_memory_fraction=self.free_gpu_memory_fraction,
            host_cache_size=self.host_cache_size,
            onboard_blocks=self.onboard_blocks,
            cross_kv_cache_fraction=self.cross_kv_cache_fraction,
            secondary_offload_min_priority=self.secondary_offload_min_priority,
            event_buffer_max_size=self.event_buffer_max_size,
            enable_partial_reuse=self.enable_partial_reuse,
            copy_on_partial_reuse=self.copy_on_partial_reuse)


@PybindMirror.mirror_pybind_fields(_ExtendedRuntimePerfKnobConfig)
class ExtendedRuntimePerfKnobConfig(BaseModel, PybindMirror):
    """
    Configuration for extended runtime performance knobs.
    """

    multi_block_mode: bool = Field(
        default=True, description="Whether to use multi-block mode.")

    enable_context_fmha_fp32_acc: bool = Field(
        default=False,
        description="Whether to enable context FMHA FP32 accumulation.")

    cuda_graph_mode: bool = Field(default=False,
                                  description="Whether to use CUDA graph mode.")

    cuda_graph_cache_size: int = Field(
        default=0,
        description=
        "Number of cuda graphs to be cached in the runtime. The larger the cache, the better the perf, but more GPU memory is consumed."
    )

    def _to_pybind(self):
        res = _ExtendedRuntimePerfKnobConfig(
            multi_block_mode=self.multi_block_mode,
            enable_context_fmha_fp32_acc=self.enable_context_fmha_fp32_acc)
        res.cuda_graph_mode = self.cuda_graph_mode
        res.cuda_graph_cache_size = self.cuda_graph_cache_size
        return res


@PybindMirror.mirror_pybind_fields(_CacheTransceiverConfig)
class CacheTransceiverConfig(BaseModel, PybindMirror):
    """
    Configuration for the cache transceiver.
    """
    max_num_tokens: Optional[int] = Field(
        default=None,
        description="The max number of tokens the transfer buffer can fit.")

    def _to_pybind(self):
        return _CacheTransceiverConfig(max_num_tokens=self.max_num_tokens)


@dataclass
class _ModelWrapper:
    model: Union[str, Path]

    def __post_init__(self):
        if not self.model:
            raise ValueError("model should be provided.")
        assert isinstance(self.model,
                          (str, Path)), f"Invalid model: {self.model}"

        model_dir = Path(self.model)

        if model_dir.exists() and model_dir.is_dir():
            self.model = model_dir

    @property
    def is_hub_model(self) -> bool:
        return not self.is_local_model

    @property
    def is_local_model(self) -> bool:
        return isinstance(self.model, Path)

    @property
    def model_dir(self) -> Path:
        assert self.is_local_model, f"model_dir is only available for local model, {self.model}."
        return self.model

    @model_dir.setter
    def model_dir(self, model_dir: Union[str, Path]):
        model_dir = Path(model_dir)
        assert model_dir.exists() and model_dir.is_dir(
        ), f"model_dir is not a valid path, {model_dir}"
        self.model = model_dir

    @property
    def model_name(self) -> Union[str, Path]:
        return self.model if isinstance(self.model, str) else None


class BaseLlmArgs(BaseModel):
    """
    Base class for both TorchLlmArgs and TrtLlmArgs. It contains all the arguments that are common to both.
    """
    model_config = {
        "arbitrary_types_allowed": True,
        "extra": "allow",
    }

    # Explicit arguments
    model: Union[str, Path] = Field(
        description=
        "The path to the model checkpoint or the model name from the Hugging Face Hub."
    )

    tokenizer: Optional[Union[
        str, Path, TokenizerBase, PreTrainedTokenizerBase]] = Field(
            description=
            "The path to the tokenizer checkpoint or the tokenizer name from the Hugging Face Hub.",
            default=None)

    tokenizer_mode: Literal['auto', 'slow'] = Field(
        default='auto',
        description="The mode to initialize the tokenizer.",
        json_schema_extra={"type": "Literal['auto', 'slow']"})

    skip_tokenizer_init: bool = Field(
        default=False,
        description="Whether to skip the tokenizer initialization.")

    trust_remote_code: bool = Field(
        default=False, description="Whether to trust the remote code.")

    tensor_parallel_size: int = Field(default=1,
                                      description="The tensor parallel size.")

    dtype: str = Field(default="auto",
                       description="The data type to use for the model.")

    revision: Optional[str] = Field(
        default=None, description="The revision to use for the model.")

    tokenizer_revision: Optional[str] = Field(
        default=None, description="The revision to use for the tokenizer.")

    # Below are all remaining arguments

    pipeline_parallel_size: int = Field(
        default=1, description="The pipeline parallel size.")

    context_parallel_size: int = Field(default=1,
                                       description="The context parallel size.")

    gpus_per_node: Optional[int] = Field(
        default=None, description="The number of GPUs per node.")

    moe_cluster_parallel_size: Optional[int] = Field(
        default=None,
        description="The cluster parallel size for MoE models's expert weights."
    )

    moe_tensor_parallel_size: Optional[int] = Field(
        default=None,
        description="The tensor parallel size for MoE models's expert weights.")

    moe_expert_parallel_size: Optional[int] = Field(
        default=None,
        description="The expert parallel size for MoE models's expert weights.")

    enable_attention_dp: bool = Field(
        default=False, description="Enable attention data parallel.")

    cp_config: Optional[dict] = Field(default_factory=dict,
                                      description="Context parallel config.")

    load_format: Literal['auto', 'dummy'] = Field(
        default='auto',
        description="The format to load the model.",
        json_schema_extra={"type": "Literal['auto', 'dummy']"})

    # LoRA arguments
    enable_lora: bool = Field(default=False, description="Enable LoRA.")

    max_lora_rank: Optional[int] = Field(
        default=None,
        description="The maximum LoRA rank.",
        deprecated="Use lora_config.max_lora_rank instead.")

    max_loras: int = Field(default=4,
                           description="The maximum number of LoRA.",
                           deprecated="Use lora_config.max_loras instead.")

    max_cpu_loras: int = Field(
        default=4,
        description="The maximum number of LoRA on CPU.",
        deprecated="Use lora_config.max_cpu_loras instead.")

    lora_config: Optional[LoraConfig] = Field(
        default=None, description="LoRA configuration for the model.")

    # Prompt adapter arguments
    enable_prompt_adapter: bool = Field(default=False,
                                        description="Enable prompt adapter.")

    max_prompt_adapter_token: int = Field(
        default=0, description="The maximum number of prompt adapter tokens.")

    # Quantization and calibration configurations
    quant_config: Optional[QuantConfig] = Field(
        default=None, description="Quantization config.")

    # Several options from ExecutorConfig, expanded here for less hierarchy
    kv_cache_config: Optional[KvCacheConfig] = Field(
        default=None, description="KV cache config.")

    enable_chunked_prefill: bool = Field(default=False,
                                         description="Enable chunked prefill.")

    guided_decoding_backend: Optional[str] = Field(
        default=None, description="Guided decoding backend.")

    batched_logits_processor: Optional[object] = Field(
        default=None,
        description="Batched logits processor.",
        json_schema_extra={
            "type": f"Optional[{get_type_repr(BatchedLogitsProcessor)}]"
        })

    iter_stats_max_iterations: Optional[int] = Field(
        default=None,
        description="The maximum number of iterations for iter stats.")

    request_stats_max_iterations: Optional[int] = Field(
        default=None,
        description="The maximum number of iterations for request stats.")

    # A handful of options from PretrainedConfig
    peft_cache_config: Optional[PeftCacheConfig] = Field(
        default=None, description="PEFT cache config.")

    scheduler_config: Optional[SchedulerConfig] = Field(
        default=None, description="Scheduler config.")

    cache_transceiver_config: Optional[CacheTransceiverConfig] = Field(
        default=None, description="Cache transceiver config.")

    # Speculative decoding parameters
    speculative_config: Optional[Union[
        LookaheadDecodingConfig, MedusaDecodingConfig, EagleDecodingConfig,
        MTPDecodingConfig, NGramDecodingConfig]] = Field(
            default=None, description="Speculative decoding config.")

    batching_type: Optional[BatchingType] = Field(default=None,
                                                  description="Batching type.")

    normalize_log_probs: bool = Field(
        default=False, description="Normalize log probabilities.")

    max_batch_size: Optional[int] = Field(default=None,
                                          description="The maximum batch size.")

    # generation constraints
    max_input_len: int = Field(default=1024,
                               description="The maximum input length.")

    max_seq_len: Optional[int] = Field(
        default=None, description="The maximum sequence length.")

    max_beam_width: int = Field(default=1,
                                description="The maximum beam width.")

    max_num_tokens: Optional[int] = Field(
        default=None, description="The maximum number of tokens.")

    backend: Optional[str] = Field(default=None,
                                   description="The backend to use.",
                                   exclude=True)

    gather_generation_logits: bool = Field(
        default=False, description="Gather generation logits.")

    # private fields those are unstable and just for internal use
    num_postprocess_workers: int = Field(
        default=0,
        description="The number of postprocess worker processes.",
        alias="_num_postprocess_workers")

    postprocess_tokenizer_dir: Optional[str] = Field(
        default=None,
        description="The postprocess tokenizer directory.",
        alias="_postprocess_tokenizer_dir")

    reasoning_parser: Optional[str] = Field(
        default=None,
        description="The parser to separate reasoning content from output.",
        alias="_reasoning_parser")

    # TODO[Superjomn]: To deprecate this config.
    decoding_config: Optional[object] = Field(
        default=None,
        description="The decoding config.",
        json_schema_extra={"type": "Optional[DecodingConfig]"},
        deprecated="Use speculative_config instead.",
    )

    mpi_session: Optional[object] = Field(
        default=None,
        description="The optional MPI session to use for this LLM instance.",
        json_schema_extra={"type": "Optional[MpiSession]"},
        exclude=True,  # exclude from serialization
        alias="_mpi_session")

    @print_traceback_on_error
    def model_post_init(self, __context: Any):

        if self.skip_tokenizer_init:
            self.tokenizer = None
        else:
            self.tokenizer = tokenizer_factory(
                self.tokenizer,
                trust_remote_code=self.trust_remote_code,
                use_fast=self.tokenizer_mode != 'slow')

        if torch.cuda.get_device_properties(0).major < 8:
            if self.dtype == 'auto':
                self.dtype = 'float16'
            if self.dtype == 'bfloat16':
                raise RuntimeError("Pre SM 80 GPUs do not support bfloat16")

        if self.gpus_per_node is None:
            logger.warning(
                f"Using default gpus_per_node: {torch.cuda.device_count()}")
            self.gpus_per_node = torch.cuda.device_count()
        assert self.gpus_per_node is not None

        if self.moe_cluster_parallel_size is None:
            self.moe_cluster_parallel_size = -1

        if self.moe_tensor_parallel_size is None:
            self.moe_tensor_parallel_size = -1

        if self.moe_expert_parallel_size is None:
            self.moe_expert_parallel_size = -1

        self.parallel_config = _ParallelConfig(
            tp_size=self.tensor_parallel_size,
            pp_size=self.pipeline_parallel_size,
            cp_size=self.context_parallel_size,
            gpus_per_node=self.gpus_per_node,
            moe_cluster_size=self.moe_cluster_parallel_size,
            moe_tp_size=self.moe_tensor_parallel_size,
            moe_ep_size=self.moe_expert_parallel_size,
            enable_attention_dp=self.enable_attention_dp,
            cp_config=self.cp_config)

        self.kv_cache_config = self.kv_cache_config or KvCacheConfig()

        self.scheduler_config = self.scheduler_config or SchedulerConfig()

    @classmethod
    def from_kwargs(cls, **kwargs: Any) -> "BaseLlmArgs":
        """Create `LlmArgs` instance from kwargs.

        Args:
            kwargs (Any): Arguments passed to `LlmArgs` constructor.

        Returns:
            tensorrt_llm.llmapi.llm_utils.BaseLlmArgs: The `BaseLlmArgs` instance.
        """
        kwargs = BaseLlmArgs._maybe_update_config_for_consistency(dict(kwargs))
        ret = cls(**kwargs)
        ret._setup()
        return ret

    def to_dict(self) -> dict:
        """Dump `LlmArgs` instance to a dict.

        Returns:
            dict: The dict that contains all fields of the `LlmArgs` instance.
        """
        return dict(
            (field.name, getattr(self, field.name)) for field in fields(self))

    @staticmethod
    def _maybe_update_config_for_consistency(
            kwargs_dict: Dict[str, Any]) -> Dict[str, Any]:
        # max_beam_width is not included since vague behavior due to lacking the support for dynamic beam width during
        # generation
        black_list = set(["max_beam_width"])
        executor_config_attrs = set(
            attr for attr in dir(_ExecutorConfig) if not attr.startswith('_')
            and callable(getattr(_ExecutorConfig, attr)))
        executor_config_attrs -= black_list
        llm_args_attr = set(BaseLlmArgs.model_fields.keys())
        # NOTE: When cpp ExecutorConfig add new options, please add the new options into `LlmArgs` with docs as well
        # ASK chunweiy for help if you are not sure about the new options.
        assert executor_config_attrs.issubset(
            llm_args_attr
        ), f"New options found in underlying ExecutorConfig: {llm_args_attr - executor_config_attrs}"

        # ensure build_config and LlmArgsBase consistency
        if kwargs_dict.get("backend") != "pytorch" and kwargs_dict.get(
                "build_config"):
            # TODO: move this to _perform_config_arbitration() once it's default-on.
            for field_name in [
                    "max_input_len", "max_seq_len", "max_beam_width"
            ]:
                build_val = getattr(kwargs_dict["build_config"], field_name,
                                    None)
                llmargs_val = kwargs_dict.get(
                    field_name) or BaseLlmArgs.model_fields[field_name]

                if build_val != llmargs_val:
                    logger.warning(
                        f"Overriding LlmArgsBase.{field_name} ({llmargs_val}) with build_config.{field_name} ({build_val})."
                    )
                    kwargs_dict[field_name] = build_val

        return kwargs_dict

    def _setup(self):
        ''' This method will setup the configs right before building the model. '''

        is_trt_llm_args = isinstance(self, TrtLlmArgs)

        assert isinstance(self.model,
                          (str, Path)), f"Invalid model: {self.model}"

        if is_trt_llm_args:
            self._setup_embedding_parallel_mode()

        if is_trt_llm_args and self.enable_build_cache:
            self.enable_build_cache = BuildCacheConfig() if isinstance(
                self.enable_build_cache, bool) else self.enable_build_cache
            if not isinstance(self.enable_build_cache, BuildCacheConfig):
                raise ValueError(
                    f"Invalid build_cache_config: {self.enable_build_cache}")
        model_obj = _ModelWrapper(self.model)

        self.speculative_model = getattr(self.speculative_config,
                                         "speculative_model", None)
        speculative_model_obj = _ModelWrapper(
            self.speculative_model
        ) if self.speculative_model is not None else None
        if model_obj.is_local_model and self.backend not in [
                'pytorch', 'autodeploy'
        ]:
            # Load parallel_config from the engine.
            self.model_format = get_model_format(self.model)

            if self.model_format is _ModelFormatKind.TLLM_ENGINE:
                if self.build_config is not None:
                    logger.warning(
                        "The build_config is ignored for model format of TLLM_ENGINE."
                    )
                self._load_config_from_engine(model_obj.model_dir)
                runtime_defaults = self._pretrained_config.runtime_defaults
                if runtime_defaults:
                    self.kv_cache_config.fill_empty_fields_from_runtime_defaults(
                        runtime_defaults)

            # Load parallel_config from the checkpoint.
            elif self.model_format is _ModelFormatKind.TLLM_CKPT:
                self._load_config_from_ckpt(model_obj.model_dir)
        else:
            self.model_format = _ModelFormatKind.HF

        if self.speculative_model and speculative_model_obj.is_local_model:
            self.speculative_model_format = _ModelFormatKind.HF

        self.quant_config = self.quant_config or QuantConfig()

        if is_trt_llm_args:
            self.calib_config = self.calib_config or CalibConfig()

        # Note: max_batch_size and max_num_tokens in LlmArgs are for runtime,
        # which will be passed to the C++ Executor API, overwriting the values
        # from an built engine. In order to set build configuration, it is
        # recommended to use build_config instead.
        if self.build_config is not None:
            if self.max_batch_size and self.build_config.max_batch_size != self.max_batch_size:
                logger.warning(
                    f"Conflict detected in LlmArgs build_config.max_batch_size "
                    f"({self.build_config.max_batch_size}) != max_batch_size ({self.max_batch_size})."
                    f"The 'max_batch_size' specified in LlmArgs is ignored at "
                    f"engine build and will override at runtime.")
            if self.max_num_tokens and self.build_config.max_num_tokens != self.max_num_tokens:
                logger.warning(
                    f"Conflict detected in LlmArgs build_config.max_num_tokens "
                    f"({self.build_config.max_num_tokens}) != max_batch_size ({self.max_num_tokens})."
                    f"The 'max_num_tokens' specified in LlmArgs is ignored at "
                    f"engine build and will override at runtime.")
        else:
            self.build_config = BuildConfig()
            if self.max_batch_size:
                self.build_config.max_batch_size = self.max_batch_size
            if self.max_num_tokens:
                self.build_config.max_num_tokens = self.max_num_tokens

        # TODO: remove the checker when manage weights support all data types
        if is_trt_llm_args and self.fast_build and (
                self.quant_config.quant_algo is QuantAlgo.FP8
                or self.quant_config.quant_algo is None):
            self._update_plugin_config("manage_weights", True)

        if self.parallel_config._world_size == 1:
            self.build_config.plugin_config.nccl_plugin = None

        self._ensure_lora_config_consistency()

        if self.enable_lora and self.lora_config is None and self.backend != 'pytorch':
            self.build_config.plugin_config.lora_plugin = 'auto'
            if self.max_lora_rank is not None:
                self.build_config.lora_config.max_lora_rank = self.max_lora_rank

        self._setup_speculative_config()

        if self.enable_prompt_adapter:
            self.build_config.max_prompt_embedding_table_size = self.max_prompt_adapter_token * self.build_config.max_batch_size

    def _setup_speculative_config(self):
        if self.speculative_config:
            if isinstance(self.speculative_config, LookaheadDecodingConfig):
                lookahead_config = self.speculative_config
                # Update the build config
                _, _, max_draft_tokens, _ = lookahead_config.calculate_speculative_resource(
                )
                self.build_config.speculative_decoding_mode = SpeculativeDecodingMode.LOOKAHEAD_DECODING
                if max_draft_tokens > self.build_config.max_draft_len:
                    self.build_config.max_draft_len = max_draft_tokens

                self.decoding_config = DecodingConfig(
                    decoding_mode=DecodingMode.Lookahead(),
                    lookahead_decoding_config=PybindMirror.maybe_to_pybind(
                        lookahead_config))
            elif isinstance(self.speculative_config, MedusaDecodingConfig):
                self.build_config.speculative_decoding_mode = SpeculativeDecodingMode.MEDUSA

                assert self.speculative_config.max_draft_len > 0
                self.build_config.max_draft_len = self.speculative_config.max_draft_len
                self.decoding_config = DecodingConfig(
                    decoding_mode=DecodingMode.Medusa(),
                    medusa_choices=self.speculative_config.medusa_choices)
            elif isinstance(self.speculative_config, EagleDecodingConfig):
                self.build_config.speculative_decoding_mode = SpeculativeDecodingMode.EAGLE
                assert self.speculative_config.max_draft_len > 0

                self.build_config.max_draft_len = self.speculative_config.max_draft_len

                if self.backend != 'pytorch':
                    eagle_config = _EagleConfig(
                        self.speculative_config.eagle_choices,
                        self.speculative_config.greedy_sampling,
                        self.speculative_config.posterior_threshold,
                        self.speculative_config.use_dynamic_tree,
                        self.speculative_config.dynamic_tree_max_topK)
                    self.decoding_config = DecodingConfig(
                        decoding_mode=DecodingMode.Eagle(),
                        eagle_config=eagle_config)
                else:
                    from tensorrt_llm._torch.speculative import Eagle3Config
                    self.speculative_config = Eagle3Config(
                        max_draft_tokens=self.speculative_config.max_draft_len,
                        eagle_weights_path=self.speculative_config.
                        pytorch_eagle_weights_path)
            elif isinstance(self.speculative_config, NGramDecodingConfig):
                self.build_config.speculative_decoding_mode = SpeculativeDecodingMode.NGRAM
                assert self.backend == 'pytorch'
                assert self.speculative_config.prompt_lookup_num_tokens > 0 and self.speculative_config.max_matching_ngram_size > 0
                self.build_config.max_draft_len = self.speculative_config.max_draft_len
                from tensorrt_llm._torch.speculative import NGramConfig
                self.speculative_config = NGramConfig(
                    prompt_lookup_num_tokens=self.speculative_config.
                    prompt_lookup_num_tokens,
                    max_matching_ngram_size=self.speculative_config.
                    max_matching_ngram_size,
                    is_keep_all=self.speculative_config.is_keep_all,
                    is_use_oldest=self.speculative_config.is_use_oldest,
                    is_public_pool=self.speculative_config.is_public_pool,
                )
            elif isinstance(self.speculative_config, MTPDecodingConfig):
                from tensorrt_llm._torch.speculative import MTPConfig
                self.speculative_config = MTPConfig(
                    num_nextn_predict_layers=self.speculative_config.
                    num_nextn_predict_layers,
                    max_batch_size=self.build_config.max_batch_size,
                    use_relaxed_acceptance_for_thinking=self.speculative_config.
                    use_relaxed_acceptance_for_thinking,
                    relaxed_topk=self.speculative_config.relaxed_topk,
                    relaxed_delta=self.speculative_config.relaxed_delta)
            else:
                raise ValueError(
                    f"Speculative config type not recognized: {self.speculative_config}"
                )
        else:
            self.decoding_config = None

    def _ensure_lora_config_consistency(self):
        if self.lora_config:
            if self.max_lora_rank is not None:
                logger.warning(
                    "max_lora_rank is ignored when lora_config is provided.")
            if self.max_loras != self.lora_config.max_loras:
                logger.warning(
                    "max_loras is ignored when lora_config is provided.")
            if self.max_cpu_loras != self.lora_config.max_cpu_loras:
                logger.warning(
                    "max_cpu_loras is ignored when lora_config is provided.")

            if len(self.lora_config.lora_dir) == 0:
                # TODO [TRTLLM-5173]
                logger.warning(
                    "lora_dir is empty, so custom embedding or lm head will not be applied."
                )

        if self.enable_lora and self.lora_config is not None and self.backend == 'pytorch':
            logger.warning(
                "enable_lora is ignored when lora_config is provided for pytorch backend."
            )

        if self.lora_config is not None:
            if len(self.lora_config.lora_dir) == 0 and len(
                    self.lora_config.lora_target_modules) == 0:
                logger.warning(
                    "Both lora_dir and lora_target_modules are empty, so all LoRA modules will be expected. "
                    "This will lead to serious memory consumption. Please provide either lora_dir or lora_target_modules if this behavior is not what you expect."
                )
                default_trtllm_modules_to_hf_modules = get_default_trtllm_modules_to_hf_modules(
                )
                self.lora_config.lora_target_modules = list(
                    default_trtllm_modules_to_hf_modules.keys())

    @property
    def _build_config_mutable(self) -> bool:
        return self.model_format is not _ModelFormatKind.TLLM_ENGINE

    def _update_plugin_config(self, key: str, value: Any):
        setattr(self.build_config.plugin_config, key, value)

    def _load_config_from_engine(self, engine_dir: Path):
        engine_config = EngineConfig.from_json_file(engine_dir / "config.json")
        self._pretrained_config = engine_config.pretrained_config
        self.build_config = engine_config.build_config

        # load and check parallel_config
        mapping = self._pretrained_config.mapping
        if self.parallel_config.tp_size not in (1, mapping.tp_size):
            raise ValueError(
                f"tp_size {self.parallel_config.tp_size} is not consistent with the engine's tp_size {mapping.tp_size}"
            )
        if self.parallel_config.pp_size not in (1, mapping.pp_size):
            raise ValueError(
                f"pp_size {self.parallel_config.pp_size} is not consistent with the engine's pp_size {mapping.pp_size}"
            )
        if self.parallel_config.cp_size not in (1, mapping.cp_size):
            raise ValueError(
                f"cp_size {self.parallel_config.cp_size} is not consistent with the engine's cp_size {mapping.cp_size}"
            )
        self.parallel_config = _ParallelConfig(
            tp_size=mapping.tp_size,
            pp_size=mapping.pp_size,
            cp_size=mapping.cp_size,
            gpus_per_node=mapping.gpus_per_node,
            moe_cluster_size=mapping.moe_cluster_size,
            moe_tp_size=mapping.moe_tp_size,
            moe_ep_size=mapping.moe_ep_size)

    def _load_config_from_ckpt(self, ckpt_dir: Path):
        pretrained_config = PretrainedConfig.from_json_file(ckpt_dir /
                                                            "config.json")
        tp_size = pretrained_config.mapping.tp_size
        pp_size = pretrained_config.mapping.pp_size
        cp_size = pretrained_config.mapping.cp_size
        moe_cluster_size = pretrained_config.mapping.moe_cluster_size
        moe_tp_size = pretrained_config.mapping.moe_tp_size
        moe_ep_size = pretrained_config.mapping.moe_ep_size
        world_size = pretrained_config.mapping.world_size
        gpus_per_node = pretrained_config.mapping.gpus_per_node
        # load parallel_config
        if self.parallel_config.tp_size != 1 and self.parallel_config.tp_size != tp_size:
            raise ValueError(
                f"tp_size {self.parallel_config.tp_size} is not consistent with the checkpoint's tp_size {tp_size}"
            )
        if self.parallel_config.pp_size != 1 and self.parallel_config.pp_size != pp_size:
            raise ValueError(
                f"pp_size {self.parallel_config.pp_size} is not consistent with the checkpoint's pp_size {pp_size}"
            )
        if self.parallel_config.cp_size != 1 and self.parallel_config.cp_size != cp_size:
            raise ValueError(
                f"cp_size {self.parallel_config.cp_size} is not consistent with the checkpoint's cp_size {cp_size}"
            )
        if (self.parallel_config.auto_parallel
                and self.parallel_config.world_size != 1 and world_size != 1):
            raise ValueError(
                f"auto parallel with world_size {self.parallel_config.world_size} does not support checkpoint with "
                "world_size {world_size} > 1")
        if not self.parallel_config.auto_parallel:
            self.parallel_config = _ParallelConfig(
                tp_size=tp_size,
                pp_size=pp_size,
                cp_size=cp_size,
                gpus_per_node=gpus_per_node,
                moe_cluster_size=moe_cluster_size,
                moe_tp_size=moe_tp_size,
                moe_ep_size=moe_ep_size)

    def _setup_embedding_parallel_mode(self):
        if self.embedding_parallel_mode == 'NONE':
            self._convert_checkpoint_options['use_parallel_embedding'] = False
        elif self.embedding_parallel_mode == 'SHARDING_ALONG_VOCAB':
            self._convert_checkpoint_options['use_parallel_embedding'] = True
            self._convert_checkpoint_options['embedding_sharding_dim'] = 0
        elif self.embedding_parallel_mode == 'SHARDING_ALONG_HIDDEN':
            self._convert_checkpoint_options['use_parallel_embedding'] = True
            self._convert_checkpoint_options['embedding_sharding_dim'] = 1
        else:
            raise ValueError(
                f"Invalid embedding_parallel_mode: {self.llm_args.embedding_parallel_mode}"
            )


class TrtLlmArgs(BaseLlmArgs):

    auto_parallel: bool = Field(
        default=False,
        description="Enable auto parallel mode.",
        deprecated=
        "Use tensor_parallel_size/pipeline_parallel_size/xxx_parallel_size instead.",
    )

    auto_parallel_world_size: Optional[int] = Field(
        default=None,
        description="The world size for auto parallel mode.",
        deprecated=
        "Use tensor_parallel_size/pipeline_parallel_size/xxx_parallel_size instead.",
    )

    enable_tqdm: bool = Field(default=False,
                              description="Enable tqdm for progress bar.")

    # BuildConfig is introduced to give users a familiar interface to configure the model building.
    build_config: Optional[object] = Field(
        default=None,
        description="Build config.",
        json_schema_extra={"type": f"Optional[{get_type_repr(BuildConfig)}]"})

    workspace: Optional[str] = Field(default=None,
                                     description="The workspace for the model.")

    # Once set, the model will reuse the build_cache
    enable_build_cache: object = Field(
        default=False,
        description="Enable build cache.",
        json_schema_extra={
            "type": f"Union[{get_type_repr(BuildCacheConfig)}, bool]"
        })

    extended_runtime_perf_knob_config: Optional[
        ExtendedRuntimePerfKnobConfig] = Field(
            default=None, description="Extended runtime perf knob config.")

    calib_config: Optional[CalibConfig] = Field(
        default=None, description="Calibration config.")

    embedding_parallel_mode: str = Field(
        default='SHARDING_ALONG_VOCAB',
        description="The embedding parallel mode.")

    fast_build: bool = Field(default=False, description="Enable fast build.")

    # Private attributes
    _auto_parallel_config: Optional[AutoParallelConfig] = PrivateAttr(
        default=None)
    # This is used to hold the options for convert_checkpoint
    _convert_checkpoint_options: Dict[str,
                                      Any] = PrivateAttr(default_factory=dict)

    @property
    def auto_parallel_config(self) -> AutoParallelConfig:
        return self._auto_parallel_config

    @print_traceback_on_error
    def model_post_init(self, __context):
        super().model_post_init(__context)

        self._auto_parallel_config = AutoParallelConfig(
            sharded_io_allowlist=[
                "past_key_value_\\d+",
                "present_key_value_\\d*",
            ],
            same_buffer_io={
                "past_key_value_(\\d+)": "present_key_value_\\1",
            },
            **infer_cluster_config(),
        )

        self.parallel_config.auto_parallel = self.auto_parallel

        if self.parallel_config.auto_parallel:
            self.parallel_config.world_size = self.auto_parallel_world_size


LlmArgs = TrtLlmArgs

LLMARGS_EXPLICIT_DOCSTRING = generate_api_docs_as_docstring(LlmArgs,
                                                            indent=' ' * 4)


class TorchLlmArgs(BaseLlmArgs):

    # Just a dummy BuildConfig to allow code reuse with the TrtLlmArgs
    build_config: Optional[object] = Field(
        default=None,
        description="Build config.",
        exclude_from_json=True,
        json_schema_extra={"type": f"Optional[{get_type_repr(BuildConfig)}]"})

    @print_traceback_on_error
    def model_post_init(self, __context):
        super().model_post_init(__context)

        self.model_format = _ModelFormatKind.HF


def update_llm_args_with_extra_dict(
        llm_args: Dict,
        llm_args_dict: Dict,
        extra_llm_api_options: Optional[str] = None) -> Dict:

    from .._torch.pyexecutor.config import PyTorchConfig
    field_mapping = {
        "quant_config": QuantConfig,
        "calib_config": CalibConfig,
        "build_config": BuildConfig,
        "kv_cache_config": KvCacheConfig,
        "decoding_config": DecodingConfig,
        "enable_build_cache": BuildCacheConfig,
        "peft_cache_config": PeftCacheConfig,
        "scheduler_config": SchedulerConfig,
        "speculative_config": DecodingBaseConfig,
        "batching_type": BatchingType,
        "extended_runtime_perf_knob_config": ExtendedRuntimePerfKnobConfig,
        "pytorch_backend_config": PyTorchConfig,
        "cache_transceiver_config": CacheTransceiverConfig,
    }
    for field, field_type in field_mapping.items():
        if field in llm_args_dict:
            if field == "speculative_config":
                llm_args_dict[field] = field_type.from_dict(
                    llm_args_dict[field])
            else:
                llm_args_dict[field] = field_type(**llm_args_dict[field])
            extra_llm_str = f"because it's specified in {extra_llm_api_options}" if extra_llm_api_options else ""
            logger.warning(f"Overriding {field} {extra_llm_str}")

    llm_args = llm_args | llm_args_dict
    return llm_args


def update_llm_args_with_extra_options(llm_args: Dict,
                                       extra_llm_api_options: str) -> Dict:
    if extra_llm_api_options is not None:
        with open(extra_llm_api_options, 'r') as f:
            llm_args_dict = yaml.safe_load(f)
            llm_args = update_llm_args_with_extra_dict(llm_args, llm_args_dict,
                                                       extra_llm_api_options)
    return llm_args


def get_model_format(model_dir: str) -> _ModelFormatKind:
    ''' Get the format of the model.  '''
    if not (Path(model_dir) / 'config.json').exists():
        raise ValueError(
            f"Failed to infer model format because no config.json exists in {model_dir}"
        )

    with open(Path(model_dir) / 'config.json') as f:
        config = json.load(f)

    try:
        if 'pretrained_config' in config and 'build_config' in config:
            model_format = _ModelFormatKind.TLLM_ENGINE
            EngineConfig.from_json_file(Path(model_dir) / 'config.json')
        elif 'architecture' in config and 'dtype' in config:
            model_format = _ModelFormatKind.TLLM_CKPT
            PretrainedConfig.from_checkpoint(model_dir)
        else:
            model_format = _ModelFormatKind.HF
            AutoConfig.from_hugging_face(model_dir)
    except Exception as e:
        raise ValueError(
            f"Inferred model format {model_format}, but failed to load config.json: {e}"
        )
    else:
        return model_format
