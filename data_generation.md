## Data Generation

### 1. Prepare CGBench 
You can download CGBench from [Huggingface](https://huggingface.co/datasets/CG-Bench/CG-Bench/tree/main).

### 2. Caption Generation

Use `data/videocaption_generation.py` to generate hierarchical captions.

- You need to deploy `Qwen3-VL-32B-Instruct` (or any compatible caption model) as server using vllm.
- To improve throughput, you can deploy multiple backend ports and use **Nginx** to map them to one endpoint with load balancing.
- Key hyperparameters can be adjusted in `data/constant.py` (e.g., sampled frames, segmentation width, prompts).
- We also provide pre-generated captions on  [Huggingface](https://huggingface.co/datasets/ChurchillQAQ/LongVideo-R1-Data)(you can skip this step if you already have them).

```
Example of vllm serve:
#Use nginx to bind ports 8081 to 8084 to one port,e.g.25600.
BASE_PORT=8081
MODEL_PATH="path/to/Qwen3-VL-32B-Instruct"

# Deploy one model per several GPUs according to the model size.
for i in 0 2 4 6; do
  PORT=$((BASE_PORT + i/2))   
  GPU_PAIR="$i,$((i+1))"
  echo "Starting GPU ${GPU_PAIR} vLLM serve (Port $PORT)..."
  CUDA_VISIBLE_DEVICES=$GPU_PAIR nohup vllm serve $MODEL_PATH \
    --tensor-parallel-size 2 \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.85 \
    --max-num-seqs 8 \
    --host 127.0.0.1 \
    --port $PORT \
    --mm-processor-cache-gb 0 \
    --served-model-name Qwen3-VL-32B > vllm_gpus${i}-${i+1}.log 2>&1 &
done
```



```bash
Example command:
python data/videocaption_generation.py \
  --base_path /path/to/cgbench/videos \
  --save_base_path /path/to/cgbench/captions \
  --subtitle_path /path/to/cgbench/subtitles \
  --use_subtitle 1 \
  --api_key 1111 \
  --base_url http://127.0.0.1:25600/v1 \
  --model Qwen3-VL-32B \
  --max_workers 32
```

### 3. SFT Data Generation

We use `GPT-5` to generate SFT data with `data/sft_data_generation.py`.

- Configure API/model settings in `data/constant.py` (e.g., `GPT_MODEL`, etc.).
- You need to deploy `Qwen3-VL-32B-Instruct` (video_qa tool) as server using vllm.
- We also provide generated sft data on [Huggingface](https://huggingface.co/datasets/ChurchillQAQ/LongVideo-R1-Data).

```bash
Example command:

python data/sft_data_generation.py \
  --file_path /path/to/cgbench.json \
  --video_base_path /path/to/cgbench/videos \
  --caption_base_path /path/to/cgbench/captions \
  --save_base_path /path/to/save \
  --gpt_api_key YOUR_GPT_API_KEY \
  --gpt_api_base YOUR_GPT_API_BASE \
  --qwen_api_key YOUR_Qwen_API_KEY \
  --qwen_api_base YOUR_Qwen_API_BASE \
  --max_workers 32 \
  --max_rounds 30
```
