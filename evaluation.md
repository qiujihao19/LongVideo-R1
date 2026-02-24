## Evaluation

### 1. Prepare Caption

Before evaluation, using `LongVideo-R1/data/videocaption_generation.py`prepare captions for:

- [LVBench](https://huggingface.co/datasets/AIWinter/LVBench)
- [VideoMME-Long](https://huggingface.co/datasets/lmms-lab/Video-MME)
- [MLVU](https://huggingface.co/datasets/sy1998/MLVU_dev)

Notice, `MIN_CUBE_WIDTH` in `LongVideo-R1/data/constant.py` should be 3 for MLVU.

You can download our generated caption from [Huggingface](https://huggingface.co/datasets/ChurchillQAQ/LongVideo-R1-Data). 

### 2. VLLM Serve Prepare

You need to deploy:

- A VL tool model (for `video_qa`), for example, Qwen3-VL-32B-Instruct.
- The reasoning model LongVideo-R1
- You can edit hyperparameters  in `LongVideo-R1/eval/constant.py`

- You can also deploy multiple reasoning backends and unify them with Nginx for higher throughput.

```
Example commdand:
#Deploy the reasoning model
#Use nginx to bind ports 8081 to 8086 to port 25600.
MODEL_PATH="path/to/LongVideo-R1"
BASE_PORT=8081

for i in {0..5}; do
  PORT=$((BASE_PORT + i))
  echo "Starting GPU $i vLLM serve (Port $PORT)..."
  CUDA_VISIBLE_DEVICES=$i nohup vllm serve $MODEL_PATH \
    --tensor-parallel-size 1 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.85 \
    --host 127.0.0.1 \
    --port $PORT \
    --served-model-name longvideor1 > vllm_gpu${i}.log 2>&1 &
done

#Deploy the video_qa model
BASE_PORT=9081
MODEL_PATH="path/to/Qwen3-VL-32B-Instruct"

for i in 6; do
  PORT=$((BASE_PORT + i - 6))   
  GPU_PAIR="$i,$((i+1))"
  echo "Starting GPU ${GPU_PAIR} vLLM serve (Port $PORT)..."
  
  CUDA_VISIBLE_DEVICES=$GPU_PAIR nohup vllm serve $MODEL_PATH \
    --tensor-parallel-size 2 \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.8 \
    --host 127.0.0.1 \
    --port $PORT \
    --mm-processor-cache-gb 0 \
    --served-model-name Qwen3-VL-32B > vllm_gpus${i}-${i+1}.log 2>&1 &
done
```



### 3. Evaluation

Use `LongVideo-R1/eval/evaluation.py` to evaluate.

We provide all of our evaluation results on [Huggingface](https://huggingface.co/datasets/ChurchillQAQ/LongVideo-R1-Data)

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

We found that the evaluation results on H800 and A800 differ: the results on A800 are about 1 point lower on average than those on H800.
