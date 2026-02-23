## Data Generation

### 1. Prepare CGBench 
You can download CGBench from [Huggingface](https://huggingface.co/datasets/CG-Bench/CG-Bench/tree/main).

### 2. Caption Generation

Use `data/videocaption_generation.py` to generate hierarchical captions.

- You need to deploy `Qwen3-VL-32B-Instruct` (or any compatible caption model) as server using vllm.
- To improve throughput, you can deploy multiple backend ports and use **Nginx** to map them to one endpoint with load balancing.
- Key hyperparameters can be adjusted in `data/constant.py` (e.g., sampled frames, segmentation width, prompts).
- We also provide pre-generated captions on Huggingface (you can skip this step if you already have them).

```bash
Example command:
python data/videocaption_generation.py \
  --base_path /path/to/cgbench/videos \
  --save_base_path /path/to/cgbench/captions \
  --subtitle_path /path/to/cgbench/subtitles \
  --use_subtitle 1 \
  --api_key YOUR_API_KEY \
  --base_url http://127.0.0.1:9081/v1 \
  --model Qwen3-VL-32B \
  --max_workers 32
```

### 3. SFT Data Generation

We use `GPT-5` to generate SFT data with `data/sft_data_generation.py`.

- Configure API/model settings in `data/constant.py` (e.g., `GPT_MODEL`, etc.).
- You need to deploy `Qwen3-VL-32B-Instruct` (video_qa tool) as server using vllm.
- We also provide generated sft data on Huggingface.

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
