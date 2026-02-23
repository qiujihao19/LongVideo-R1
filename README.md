# LongVideo-R1

本仓库主要包含 LongVideo-R1 的数据构建、SFT、RL、评测与在线 Demo。

## 1. 数据准备与生成

### 1.1 下载数据集
- 先下载 **CG-Bench** 原始视频与字幕（如有字幕）。
- 推荐目录结构（示例）：
  - `data/cgbench/videos/*.mp4`
  - `data/cgbench/subtitles/*.srt`

### 1.2 生成 CG-Bench Caption
使用 `data/videocaption_generation.py` 生成分层 caption。

- 你需要先部署 `Qwen3-VL`（或其他可替代的 caption 模型）并提供 OpenAI 兼容接口。
- 为了提升吞吐，可部署多个后端端口，再用 **Nginx** 映射到同一入口端口做负载均衡。
- 关键超参数可在 `data/constant.py` 中修改（如采样帧数、分段宽度、prompt 等）。
- 仓库也提供了可直接使用的预生成 caption（如你已有该文件可跳过本步骤）。

命令示例：

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

### 1.3 生成 SFT 数据
我们使用 `GPT-5` 生成 SFT 数据，脚本为 `data/sft_data_generation.py`。

- 先在 `data/constant.py` 中配置相应 API / model（如 `GPT_MODEL`、base_url 等）。
- 再运行 `sft_data_generation.py` 生成训练 json/jsonl。

## 2. SFT

我们使用 **LLaMA-Factory** 进行 SFT。

### 2.1 安装 LLaMA-Factory
请按 `LLaMA-Factory/README.md` 安装依赖。

### 2.2 放置数据并配置 `dataset_info.json`
- 将你生成好的 SFT 数据（或我们提供的数据）放到：
  - `LLaMA-Factory/data/`（或 `llamafactory/data/`）
- 修改对应 `dataset_info.json`（如 `LLaMA-Factory/data/dataset_info.json`）中的数据路径与字段映射。

### 2.3 启动 SFT

```bash
cd LLaMA-Factory
llamafactory-cli train ./examples/train_full/qwen3.yaml
```

## 3. RL

我们使用 **verl + verl-tool** 进行强化学习。

### 3.1 安装 verl 与 verl-tool
按以下文档安装：
- `verl-tool/README.md`
- `verl-tool/verl/README.md`

### 3.2 模型部署（示例：8xA800）
- 示例策略：将工具视觉模型（如 Qwen3-VL）部署在 `GPU 6,7`。
- 主策略模型/参考模型等按你的训练配置分配其余 GPU。

### 3.3 工具配置
你需要修改工具配置文件：
- `verl-tool/verl_tool/servers/tools/utils/caption_videoqa_config.example.json`
- `verl-tool/verl_tool/servers/tool_init_config.example.json`

将其中的：
- `caption_dir` / `video_dir`
- `api_key` / `base_url` / `videoqa_model`
- `config_path`
替换成你自己的实际路径与服务地址。

### 3.4 工具连通性测试
可用示例脚本测试工具是否正常：
- `verl-tool/examples/server_test.sh`

测试通过后再开始 RL 训练。

## 4. Evaluation

评测前需要准备以下数据集对应的 caption：
- LVBench
- VideoMME-long
- MLVU

使用 `eval/evaluation.py` 进行测试。你需要先部署：
- VL 工具模型（给 `video_qa`）
- reasoning 模型

同样可以部署多个 reasoning 后端并通过 Nginx 统一映射提升吞吐。

命令示例：

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

可使用 `cli.py` 进行在线测试（工具调用 + 多轮推理）。

命令示例：

```bash
python cli.py \
  --video_path /path/to/video.mp4 \
  --question "What is the man doing in this video?" \
  --reasoning_base_url http://127.0.0.1:25600/v1 \
  --caption_base_url http://127.0.0.1:9081/v1 \
  --videoqa_base_url http://127.0.0.1:9081/v1
```

---

如果你希望，我可以再补一版英文 README，或给每个阶段加上「最小可复现目录结构 + 常见报错排查」。
