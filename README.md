# LongVideo-R1: Smart Navigation for Low-cost Long Video Understanding
This is the official implementaion of paper '[***LongVideo-R1: Smart Navigation for Low-cost Long Video Understanding***]()', which is accepted in ***CVPR 2026***.

​                                                   [[📖 Paper]()] [[🤗 LongVideo-R1-Qwen2.5](https://huggingface.co/ChurchillQAQ/LongVideo-R1-Qwen2.5)] [[🤗 LongVideo-R1-Qwen3]()]  [[🤗 LongVideo-R1-Data](https://huggingface.co/datasets/ChurchillQAQ/LongVideo-R1-Data)] 


## Abstract

This paper addresses the critical and underexplored challenge of long video understanding with low computational budgets. We propose LongVideo-R1, an active, reasoning-equipped multimodal large language model (MLLM) agent designed for efficient video context navigation, avoiding the redundancy of exhaustive search. At the core of LongVideo-R1 lies a reasoning module that leverages high-level visual cues to infer the most informative video clip for subsequent processing. During inference, the agent initiates traversal from top-level visual summaries and iteratively refines its focus, immediately halting the exploration process upon acquiring sufficient knowledge to answer the query. To facilitate training, we first extract hierarchical video captions from CGBench, a video corpus with grounding annotations, and guide GPT-5 to generate 33K high-quality chain-of-thought-with-tool trajectories. The LongVideo-R1 agent is fine-tuned upon the Qwen-3-8B model through a two-stage paradigm: supervised fine-tuning (SFT) followed by reinforcement learning (RL), where RL employs a specifically designed reward function to maximize selective and efficient clip navigation. Experiments on multiple long video benchmarks validate the effectiveness of name, which enjoys superior tradeoff between QA accuracy and efficiency. 

## Model Download

We provide two versions of the model:

1. [**LongVideo-R1-Qwen2.5**](https://huggingface.co/ChurchillQAQ/LongVideo-R1-Qwen2.5): obtained by performing SFT and RL using **Qwen3-8B** as the reasoning model, **Qwen2.5-VL-72B** as the caption model, and **Qwen2.5-VL-32B** as the video_qa model.
2. **LongVideo-R1-Qwen3**: obtained by performing SFT and RL using **Qwen3-8B** as the reasoning model, and **Qwen3-VL-32B** as both the caption model and video_qa model. **This model will be released in a few days**.

**LongVideo-R1-Qwen3** delivers better performance.


## 📚 Contents Link

- [Data Generation](./assets/docs/install.md)
- [Two Stage Training](./assets/docs/sync_design.md)
- [Evaluation](./assets/docs/asyncRL.md)


## Cli Demo
Use `cli.py` for online testing (tool use + multi-round reasoning).

LongVideo-R1, caption model, and video_qa model should be deployed in vllm serve mode.

Example command:

```bash
#Deploy the reasoning model.
MODEL_PATH="path/to/LongVideo-R1"
PORT=25600

echo "Starting GPU 0 vLLM serve (Port $PORT)..."
CUDA_VISIBLE_DEVICES=0 vllm serve $MODEL_PATH \
--tensor-parallel-size 1 \
--max-model-len 32768 \
--gpu-memory-utilization 0.85 \
--host 127.0.0.1 \
--port $PORT \
--served-model-name longvideor1 

#Deploy the caption and video_qa model.
MODEL_PATH="path/to/Qwen3-VL-32B-Instruct"
PORT=9081
GPU_PAIR="6,7"

echo "Starting GPU ${GPU_PAIR} vLLM serve (Port $PORT)..."
CUDA_VISIBLE_DEVICES=$GPU_PAIR vllm serve $MODEL_PATH \
--tensor-parallel-size 2 \
--max-model-len 16384 \
--gpu-memory-utilization 0.85 \
--host 127.0.0.1 \
--port $PORT \
--mm-processor-cache-gb 0 \
--served-model-name Qwen3-VL-32B 
```


```bash
python cli.py \
  --video_path /path/to/video.mp4 \
  --question "What is the man doing in this video?" \
  --reasoning_base_url http://127.0.0.1:25600/v1 \
  --caption_base_url http://127.0.0.1:9081/v1 \
  --videoqa_base_url http://127.0.0.1:9081/v1
```



## Acknowledgement

This project is based on LLaMA-Factory ([paper](https://arxiv.org/pdf/2403.13372), [code](https://github.com/hiyouga/LlamaFactory)), verl-tool([paper](https://arxiv.org/pdf/2509.01055), [code](https://github.com/TIGER-AI-Lab/verl-tool)), verl([code](https://github.com/verl-project/verl)), thanks for their excellent works.

## Citation
If you find LongVideo-R1 useful for your research and applications, please consider giving us a star ⭐ and citing it by the following BibTeX entry:
```

```
