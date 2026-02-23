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

- Download LongVideo-R1 llamafactory training data file(longvideo-r1-sft.json) from Hugginface.
- Put longvideo-r1-sft.json in LLaMA-Factory/data. If you use your own data, don't forget to modify the [dataset_info.json](./LLaMA-Factory/data/dataset_info.json) file.

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

### 2. Tool Preparing 

#### 2.1 Model Deployment (Example: 8xA800)

- Example strategy: deploy the tool vision model (e.g., Qwen3-VL) on `GPU 6,7` using vllm serve.
- Place policy/reference/other components on the remaining GPUs based on your training setup.

#### 2.2 Tool Configuration

You need to edit:

- `verl-tool/verl_tool/servers/tools/utils/caption_videoqa_config.example.json`
- `verl-tool/verl_tool/servers/tool_init_config.example.json`

Replace the following with your actual values:

- `caption_dir` / `video_dir`
- `api_key` / `base_url` / `videoqa_model`
- `config_path`

#### 2.3 Tool Connectivity Test

Use the example script to verify tools are working:

- `verl-tool/examples/server_test.sh`

If the test passes, you can start RL training.

### 3. RL Training

Edit `verl-tool/examples/train/get_caption/train_7b_videoqa.sh`

```
#run rl training
cd verl-tool
bash ./examples/train/get_caption/train_7b_videoqa.sh
```

