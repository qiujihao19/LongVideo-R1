## Evaluation

### 1. Prepare Caption

Before evaluation, using `LongVideo-R1/data/videocaption_generation.py`prepare captions for:

- [LVBench](https://huggingface.co/datasets/AIWinter/LVBench)
- [VideoMME-Long](https://huggingface.co/datasets/lmms-lab/Video-MME)
- [MLVU](https://huggingface.co/datasets/sy1998/MLVU_dev)

Notice, `MIN_CUBE_WIDTH` in `LongVideo-R1/data/constant.py` should be 3 for MLVU.

You can download our generated caption from [Huggingface](https://huggingface.co/datasets/CG-Bench/CG-Bench/tree/main). 

### 2. VLLM Serve Prepare

You need to deploy:

- A VL tool model (for `video_qa`), for example, Qwen3-VL-32B-Instruct.
- The reasoning model LongVideo-R1
- You can edit hyperparameters  in `LongVideo-R1/eval/constant.py`

- You can also deploy multiple reasoning backends and unify them with Nginx for higher throughput.

### 3. Evaluation

Use `LongVideo-R1/eval/evaluation.py` to evaluate.

```bash
Example command:

python eval/evaluation.py \
  --eval_data_file /path/to/eval.json \
  --eval_dataset lvbench \
  --caption_base_path /path/to/caption_base \
  --video_base_path /path/to/video_base \
  --save_base_path /path/to/save_dir \
  --max_rounds 30 \
  --max_workers 36
```
