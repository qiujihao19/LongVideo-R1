# LongVideo-R1

This repository contains the full LongVideo-R1 pipeline, including data construction, SFT, RL, evaluation, and an online demo.

## 1. Data Preparation and Generation

### 1.1 Download Dataset
- First, download **CG-Bench** raw videos and subtitles (if available).
- Recommended directory layout (example):
  - `data/cgbench/videos/*.mp4`
  - `data/cgbench/subtitles/*.srt`

### 1.2 Generate CG-Bench Captions
Use `data/videocaption_generation.py` to generate hierarchical captions.

- You need to deploy `Qwen3-VL` (or any compatible caption model) with an OpenAI-compatible API.
- To improve throughput, you can deploy multiple backend ports and use **Nginx** to map them to one endpoint with load balancing.
- Key hyperparameters can be adjusted in `data/constant.py` (e.g., sampled frames, segmentation width, prompts).
- We also provide pre-generated captions (you can skip this step if you already have them).

Example command:

```bash
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

### 1.3 Generate SFT Data
We use `GPT-5` to generate SFT data with `data/sft_data_generation.py`.

- Configure API/model settings in `data/constant.py` (e.g., `GPT_MODEL`, base_url, etc.).
- Then run `sft_data_generation.py` to produce training json/jsonl files.

## 2. SFT

We use **LLaMA-Factory** for SFT.

### 2.1 Install LLaMA-Factory
Follow `LLaMA-Factory/README.md` for installation.

### 2.2 Prepare Data and Update `dataset_info.json`
- Put your generated SFT data (or our provided data) into:
  - `LLaMA-Factory/data/` (or `llamafactory/data/`)
- Update `dataset_info.json` (e.g., `LLaMA-Factory/data/dataset_info.json`) with your dataset path and field mapping.

### 2.3 Run SFT

```bash
cd LLaMA-Factory
llamafactory-cli train ./examples/train_full/qwen3.yaml
```

## 3. RL

We use **verl + verl-tool** for reinforcement learning.

### 3.1 Install verl and verl-tool
Please follow:
- `verl-tool/README.md`
- `verl-tool/verl/README.md`

### 3.2 Model Deployment (Example: 8xA800)
- Example strategy: deploy the tool vision model (e.g., Qwen3-VL) on `GPU 6,7`.
- Place policy/reference/other components on the remaining GPUs based on your training setup.

### 3.3 Tool Configuration
You need to edit:
- `verl-tool/verl_tool/servers/tools/utils/caption_videoqa_config.example.json`
- `verl-tool/verl_tool/servers/tool_init_config.example.json`

Replace the following with your actual values:
- `caption_dir` / `video_dir`
- `api_key` / `base_url` / `videoqa_model`
- `config_path`

### 3.4 Tool Connectivity Test
Use the example script to verify tools are working:
- `verl-tool/examples/server_test.sh`

If the test passes, you can start RL training.

## 4. Evaluation

Before evaluation, prepare captions for:
- LVBench
- VideoMME-long
- MLVU

Use `eval/evaluation.py` for testing. You need to deploy:
- A VL tool model (for `video_qa`)
- A reasoning model

You can also deploy multiple reasoning backends and unify them with Nginx for higher throughput.

Example command:

```bash
python eval/evaluation.py \
  --eval_data_file /path/to/eval.json \
  --eval_dataset lvbench \
  --caption_base_path /path/to/caption_base \
  --video_base_path /path/to/video_base \
  --save_base_path /path/to/save_dir \
  --max_rounds 30 \
  --max_workers 36
```

## 5. Demo

Use `cli.py` for online testing (tool use + multi-round reasoning).

Example command:

```bash
python cli.py \
  --video_path /path/to/video.mp4 \
  --question "What is the man doing in this video?" \
  --reasoning_base_url http://127.0.0.1:25600/v1 \
  --caption_base_url http://127.0.0.1:9081/v1 \
  --videoqa_base_url http://127.0.0.1:9081/v1
```
