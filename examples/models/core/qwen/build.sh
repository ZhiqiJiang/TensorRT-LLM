
python convert_checkpoint.py --model_dir /root/TensorRT-LLM/Qwen2-7B-Instruct/ \
                              --output_dir /root/TensorRT-LLM/Qwen2-7B-Instruct_checkpoint \
                              --dtype bfloat16

trtllm-build --checkpoint_dir /root/TensorRT-LLM/Qwen2-7B-Instruct_checkpoint \
            --output_dir /root/TensorRT-LLM/Qwen2-7B-Instruct_trt \
            --gpt_attention_plugin bfloat16 \
            --gemm_plugin bfloat16
