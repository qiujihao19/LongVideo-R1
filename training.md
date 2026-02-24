## SFT Training

### 1. Install [LlamaFactory](https://github.com/hiyouga/LlamaFactory)

```
conda create -n llamafactory python=3.10 -y
conda activate llamafactory 
cd LongVideo-R1
cd LLaMA-Factory
pip install -e ".[torch,metrics,deepspeed,liger-kernel,bitsandbytes]" --no-build-isolation
```

### 2. Training

- We provide the SFT data `longvideor1-sft-qwen2.5.json` generated with **Qwen2.5-VL-72B** as the caption model and **Qwen2.5-VL-32B** as the video_qa model, as well as the SFT data `longvideor1-sft-qwen3.json` generated with **Qwen3-VL-32B** as both the caption model and video_qa model.

  You can download them from [HuggingFace](https://huggingface.co/datasets/ChurchillQAQ/LongVideo-R1-Data), and we recommend using the data generated with Qwen3.
- Put `longvideor1-sft-qwen3-llamafactory.json` in `LLaMA-Factory/data` and modify the file_name in [dataset_info.json](./LLaMA-Factory/data/dataset_info.json) file.

```bash
Training command:
llamafactory-cli train examples/train_full/qwen3.yaml
```

After training, replace the `chat_template.jinja` file in the trained checkpoint folder with the standard [chat_template.jinja](./LLaMA-Factory/examples/train_full/chat_template.jinja) to prevent the content within <think></think>  be masked. 



## RL Training

### 1. Install [verl-tool](https://github.com/TIGER-AI-Lab/verl-tool)&[verl](https://github.com/verl-project/verl) 

```
conda create -n verl-tool python=3.10 -y
conda activate verl-tool 
cd verl-tool
cd verl
USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh
cd ..
pip install -e ".[vllm]"
```

### 2. Data Preparing

- Download CGBench from [Huggingface](https://huggingface.co/datasets/CG-Bench/CG-Bench/tree/main).
- Download CGBench captions and rl data from [HuggingFace](https://huggingface.co/datasets/ChurchillQAQ/LongVideo-R1-Data).

### 3. Tool Preparing 

#### 3.1 Model Deployment (Example: 8xA800)

- Example strategy: deploy the tool vision model (e.g., Qwen3-VL) on `GPU 6,7` using vllm serve.
- Place policy/reference/other components on the remaining GPUs based on your training setup.

```
Example Command:

PORT=9081
MODEL_PATH="path/to/Qwen3-VL-32B-Instruct"

GPU_PAIR="6,7"
echo "Starting GPU ${GPU_PAIR} vLLM serve (Port $PORT)..."

CUDA_VISIBLE_DEVICES=$GPU_PAIR vllm serve $MODEL_PATH \
--tensor-parallel-size 2 \
--max-model-len 16384 \
--gpu-memory-utilization 0.8 \
--host 127.0.0.1 \
--port $PORT \
--mm-processor-cache-gb 0 \
--served-model-name Qwen3-VL-32B 
```

#### 3.2 Tool Configuration

You need to edit:

- `verl-tool/verl_tool/servers/tools/utils/caption_videoqa_config.example.json`
- `verl-tool/verl_tool/servers/tool_init_config.example.json`

Replace the following with your actual values:

- `caption_dir` / `video_dir`
- `api_key` / `base_url` / `videoqa_model`
- `config_path`

#### 3.3 Tool Connectivity Test

Use the example script to verify tools are working:

- `verl-tool/examples/server_test.sh`

If the test passes, you can start RL training. 

### 4. RL Training

We provide RL training data initialized with the caption from **Qwen2.5-VL-72B** and RL training data initialized with the caption from **Qwen3-VL-32B**. You can download them on [HuggingFace]().

Edit `verl-tool/examples/train/get_caption/train_7b_videoqa.sh`

```
#run rl training
cd verl-tool
bash ./examples/train/get_caption/train_7b_videoqa.sh
```

