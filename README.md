# LongVideo-R1: Smart Navigation for Low-cost Long Video Understanding
This is the official implementaion of paper '[***LongVideo-R1: Smart Navigation for Low-cost Long Video Understanding***]()', which is accepted in ***CVPR 2026***.

[[📖 Paper]()] [[🤗 LongVideo-R1-model]()] [[🤗 LongVideo-R1-data]()] 


## Abstract

This paper addresses the critical and underexplored challenge of long video understanding with low computational budgets. We propose LongVideo-R1, an active, reasoning-equipped multimodal large language model (MLLM) agent designed for efficient video context navigation, avoiding the redundancy of exhaustive search. At the core of LongVideo-R1 lies a reasoning module that leverages high-level visual cues to infer the most informative video clip for subsequent processing. During inference, the agent initiates traversal from top-level visual summaries and iteratively refines its focus, immediately halting the exploration process upon acquiring sufficient knowledge to answer the query. To facilitate training, we first extract hierarchical video captions from CGBench, a video corpus with grounding annotations, and guide GPT-5 to generate 33K high-quality chain-of-thought-with-tool trajectories. The LongVideo-R1 agent is fine-tuned upon the Qwen-3-8B model through a two-stage paradigm: supervised fine-tuning (SFT) followed by reinforcement learning (RL), where RL employs a specifically designed reward function to maximize selective and efficient clip navigation. Experiments on multiple long video benchmarks validate the effectiveness of name, which enjoys superior tradeoff between QA accuracy and efficiency. 

## 📚 Contents Link

- [Data Generation](./data_generation.md)
- [Two Stage Training](./training.md)
- [Evaluation](./evaluation.md)


## Cli Demo
Use `cli.py` for online testing (tool use + multi-round reasoning).

LongVideo-R1, caption model, and video_qa model should be deployed in vllm serve mode.

Example command:

```bash
python cli.py \
  --video_path /path/to/video.mp4 \
  --question "What is the man doing in this video?" \
  --reasoning_base_url http://127.0.0.1:25600/v1 \
  --caption_base_url http://127.0.0.1:9081/v1 \
  --videoqa_base_url http://127.0.0.1:9081/v1
```

## Acknowledgement

This project is based on LLaMA-Factory ([paper](https://arxiv.org/pdf/2403.13372)), [code](https://github.com/hiyouga/LlamaFactory)), verl-tool([paper](https://arxiv.org/pdf/2509.01055), [code](https://github.com/TIGER-AI-Lab/verl-tool)), verl([code](https://github.com/verl-project/verl)), thanks for their excellent works.

## Citation
If you find LongVideo-R1 useful for your research and applications, please consider giving us a star ⭐ and citing it by the following BibTeX entry:
```

```
