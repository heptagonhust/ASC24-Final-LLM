# tllm_checkpoint
1. Build the LLaMA 7B model using a single GPU and FP16.
``` 
python convert_checkpoint.py --model_dir /mnt/Aquila/AquilaChat2-34B \
                              --output_dir ~/tllm_checkpoint/tllm_checkpoint_1gpu_fp16 \
                              --dtype float16
```

1. Build the LLaMA 7B model using a single GPU and apply INT8 weight-only quantization.
```
python convert_checkpoint.py --model_dir ./tmp/llama/7B/ \
                              --output_dir ./tllm_checkpoint_1gpu_fp16_wq \
                              --dtype float16 \
                              --use_weight_only \
                              --weight_only_precision int8
```