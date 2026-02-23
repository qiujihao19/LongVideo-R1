import argparse
import ast
import base64
import hashlib
import json
import os
import re
import time
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from decord import VideoReader, cpu
from openai import OpenAI
from PIL import Image


# Defaults aligned with eval/constant.py
DEFAULT_REASONING_MODEL = "longvideor1"
DEFAULT_VIDEOQA_MODEL = "Qwen2.5-VL-32B"
DEFAULT_CAPTION_MODEL = "Qwen2.5-VL-32B"
DEFAULT_REASONING_BASE_URL = "http://127.0.0.1:25600/v1"
DEFAULT_VIDEOQA_BASE_URL = "http://127.0.0.1:9081/v1"
DEFAULT_CAPTION_BASE_URL = "http://127.0.0.1:9081/v1"
DEFAULT_API_KEY = "111111"

VIDEOQA_IMAGE_MAX_LENGTH = 1024
VIDEOQA_FRAME_NUM = 16
SEGMENT_LEN = 16
MIN_CUBE_WIDTH = 4
MAX_CUBE_WIDTH = 8
MIN_SAMPLE_FRAME_NUM = 32
HIGH_LEVEL_CAPTION_WORDS = 400
LOW_LEVEL_CAPTION_WORDS = 200


SYSTEM_PROMPT = '''
You are a reasoning assistant designed to answer questions about a long video through hierarchical captions. 
The video is organized into three levels of temporal granularity:
1. **High-level**: The video is divided into {width} major segments.  
   - Each segment contains one caption summarizing its content.  
2. **Medium-level**: Each High-level segment is further divided into {width} sub-segments.  
   - Each sub-segment contains one caption.  
3. **Low-level**: Each Medium-level segment is further divided into {width} finer sub-segments.  
   - Each sub-segment contains one caption.
### Task
You will be asked a question about the video.  
At the beginning, you are given **only the High-level captions**.
Your goal is to answer the question as accurately as possible.
---
### Reasoning and Tool Usage
1. **Reason first:**  
   Before taking any action, carefully analyze whether the current information (captions you already have) is sufficient to answer the question.
2. **If sufficient:**  
   Directly provide your final answer inside `<answer></answer>` tags.
3. **If insufficient:**  
   Identify which part(s) of the video might contain the needed information.  
   Then use one of the following tools:
   - **To obtain finer captions:**
     ```
     <tool>get_caption((high_segment_id, medium_segment_id, low_segment_id))</tool>
     ```
     - Each of the three IDs is an integer from 1 to {width}.
     - To request a **Medium-level** caption, provide `(high_segment_id, medium_segment_id)` only.  
     - To request a **Low-level** caption, provide the full triplet `(high_segment_id, medium_segment_id, low_segment_id)`.
   - **To query visual information from the actual video segment:**
     ```
     <tool>video_qa((high_segment_id, medium_segment_id, low_segment_id), query)</tool>
     ```
     - This tool sends the **corresponding Low-level video segment** to a specialized video QA module.  
     - The `query` should specify **what exact information** you need (e.g., “what color is the person’s shirt?”, “what object is on the table?”).  
     - You may **only use `video_qa`** after you have already retrieved the corresponding Low-level caption for that segment.
---
### Output Format
Your reasoning and actions must follow this structure exactly:
<think>Your internal reasoning process here. Analyze what information you have, what is missing, and which part might be relevant.</think>
<tool>(get_caption or video_qa call here, if needed)</tool>  
or  
<think>...</think>
<answer>Your final answer here (only when you are confident the information is sufficient).</answer>
'''

VIDEO_QA_SYSTEM_PROMPT = '''
You are a video question-answering model. You will be given a video segment and a question. Your task is to summarize the information relevant to the question based on the provided video segment.
You must think first before answering. Your thinking process should be in <reasoning></reasoning> tags. Your answer should be in <answer></answer> tags.
If the question is not related to the video, just answer "<answer>The question is not related to the video segment.</answer>".
'''

CAPTION_SYSTEM_PROMPT = '''
    ### Task:
        You are a video understanding expert. Please create a detailed description with timestamp information for the current video clip (which contains multiple frames arranged in chronological order).
    ### Input Video Frames:
        You are given {num_frame} uniformly sampled frames from the video, along with the timestamp (in seconds) of each frame in the entire video.
    ### Description Guidelines:
    -Dialogue Description Guidelines:
    1)In addition to video frames, subtitle information for this video segment is also provided.
    2)The output description must faithfully include the given subtitle content. Do not invent or add dialogues that are not provided. Avoid redundant repetition, maintain the original order of the lines, and ensure the sentences flow smoothly.
    -Whenever reasonable, include the provided timestamps in your description.
    1)For multiple frames with short intervals that depict the same continuous action, you may merge them into a single description.
    2)For example:This video begins at 0.0s with a scene featuring two individuals seated outdoors, engaging in a conversation. The subtitles indicate they are discussing the impact of the pandemic on their ability to shoot videos at a bar. By 14.0s, the dialogue shifts to their newfound regular appearances on a show called Scam Nation. At 28.0s, the conversation turns to the promotion of a product named Kraken, encouraging viewers to visit a website for purchase ...
    
    ### Output Format:
    Your response should be in the following format, wrapped with <caption></caption> tags: "<caption>This clip (video) XXX</caption>".
    The description should be strictly controlled within {num_words} words.
'''


CAPTION_SYSTEM_PROMPT_LOW = '''
    ### Task:
        You are a video understanding expert. Please create a detailed description with timestamp information for the current video clip (which contains multiple frames arranged in chronological order).
    ### Input Video Frames:
        You are given {num_frame} uniformly sampled frames from the video, along with the timestamp (in seconds) of each frame in the entire video.
    ### Description Guidelines:
    -Dialogue Description Guidelines:
    1)In addition to video frames, subtitle information for this video segment is also provided.
    2)The output description must faithfully include the given subtitle content. Do not invent or add dialogues that are not provided. Avoid redundant repetition, maintain the original order of the lines, and ensure the sentences flow smoothly.
    -Whenever reasonable, include the provided timestamps in your description.
    1)For multiple frames with short intervals that depict the same continuous action, you may merge them into a single description.
    2)For example:This video begins at 0.0s with a scene featuring two individuals seated outdoors, engaging in a conversation. The subtitles indicate they are discussing the impact of the pandemic on their ability to shoot videos at a bar. By 14.0s, the dialogue shifts to their newfound regular appearances on a show called Scam Nation. At 28.0s, the conversation turns to the promotion of a product named Kraken, encouraging viewers to visit a website for purchase ...
    -Add more detailed descriptions if necessary.
    1)Enrich the caption with contextual details such as location, environment, lighting, weather, people's clothing, and atmosphere, while keeping it visually accurate.
    ### Output Format:
    Your response should be in the following format, wrapped with <caption></caption> tags: "<caption>This clip (video) XXX</caption>".
    The description should be strictly controlled within {num_words} words.
'''



def extract_tags(text: str, tags: List[str]) -> Dict[str, str]:
    results = {}
    for tag in tags:
        match = re.search(fr"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
        if match:
            results[tag] = match.group(1).strip()
    return results


def parse_function_call(call_str: str):
    match = re.match(r"(\w+)\((.*)\)", call_str.strip())
    if not match:
        return None, None, False

    func_name = match.group(1)
    arg_str = match.group(2).strip()
    try:
        args = ast.literal_eval(arg_str) if arg_str else ()
        return func_name, args, True
    except Exception:
        return func_name, arg_str, False


def encode_image(image: np.ndarray, max_length: int) -> str:
    pil_image = Image.fromarray(image)
    w, h = pil_image.size
    if w > max_length or h > max_length:
        scale = min(max_length / h, max_length / w)
        pil_image = pil_image.resize((round(w * scale), round(h * scale)))

    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def get_video_duration_cuberoot_from_meta(
    num_frames: int,
    fps: float,
    segment_len: int = SEGMENT_LEN,
    min_width: int = MIN_CUBE_WIDTH,
    max_width: int = MAX_CUBE_WIDTH,
) -> int:
    duration = num_frames / fps
    num_segments = duration / segment_len
    width = round(num_segments ** (1 / 3))
    width = max(min_width, min(max_width, width))
    return width


class OnlineCaptionCache:
    def __init__(self, cache_path: str, video_path: str, width: int, num_frames: int):
        self.cache_path = cache_path
        self.video_path = os.path.abspath(video_path)
        self.width = width
        self.num_frames = num_frames
        self.data = self._load_or_init()

    def _load_or_init(self):
        if os.path.exists(self.cache_path):
            with open(self.cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            meta = data.get("meta", {})
            if (
                meta.get("video_path") == self.video_path
                and int(meta.get("width", -1)) == self.width
                and int(meta.get("num_frames", -1)) == self.num_frames
            ):
                for key in ["high", "medium", "low"]:
                    data.setdefault(key, {})
                return data

        return {
            "meta": {
                "video_path": self.video_path,
                "width": self.width,
                "num_frames": self.num_frames,
            },
            "high": {},
            "medium": {},
            "low": {},
        }

    def _save(self):
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)

    def get(self, level: str, key: str):
        return self.data[level].get(key)

    def put(self, level: str, key: str, value: dict):
        self.data[level][key] = value
        self._save()


class LongVideoDemo:
    def __init__(self, args):
        self.video_path = args.video_path
        self.max_rounds = args.max_rounds
        self.decode_threads = args.decode_threads

        # Reuse one VideoReader instance to avoid repeated open/decode init overhead.
        self.vr = VideoReader(self.video_path, ctx=cpu(0), num_threads=self.decode_threads)
        self.fps = self.vr.get_avg_fps()
        self.num_frames = len(self.vr)

        self.reasoning_client = OpenAI(api_key=args.api_key, base_url=args.reasoning_base_url)
        self.caption_client = OpenAI(api_key=args.api_key, base_url=args.caption_base_url)
        self.video_client = OpenAI(api_key=args.api_key, base_url=args.videoqa_base_url)

        self.reasoning_model = args.reasoning_model
        self.caption_model = args.caption_model
        self.videoqa_model = args.videoqa_model

        self.width = get_video_duration_cuberoot_from_meta(self.num_frames, self.fps)
        self.video_duration = round(self.num_frames / self.fps, 1)
        cache_file = self._build_cache_file(args.cache_dir, self.video_path)
        self.cache = OnlineCaptionCache(cache_file, self.video_path, self.width, self.num_frames)

    def _process_video_frames(self, frame_ids: np.ndarray) -> Tuple[np.ndarray, List[float]]:
        frames = self.vr.get_batch(frame_ids).asnumpy()
        frame_time = [round(i / self.fps, 1) for i in frame_ids]
        return frames, frame_time

    @staticmethod
    def _build_cache_file(cache_dir: str, video_path: str) -> str:
        vp = os.path.abspath(video_path)
        stem = Path(vp).stem
        suffix = hashlib.md5(vp.encode("utf-8")).hexdigest()[:8]
        return os.path.join(cache_dir, f"{stem}_{suffix}.json")

    def _segment_key(self, ids: Tuple[int, ...]) -> Tuple[str, str]:
        if len(ids) == 1:
            return "high", str(ids[0])
        if len(ids) == 2:
            return "medium", f"{ids[0]},{ids[1]}"
        if len(ids) == 3:
            return "low", f"{ids[0]},{ids[1]},{ids[2]}"
        raise ValueError(f"Invalid segment id length: {ids}")

    def _validate_ids(self, ids: Tuple[int, ...]):
        for v in ids:
            if not (1 <= int(v) <= self.width):
                raise ValueError(f"segment id {ids} out of range [1, {self.width}]")

    def _get_caption_frame_ids(self, ids: Tuple[int, ...]) -> Tuple[np.ndarray, int]:
        all_indices = np.arange(self.num_frames)
        if len(ids) == 1:
            h = ids[0]
            chunks = np.array_split(all_indices, self.width)
            part = chunks[h - 1]
            sample_num = MIN_SAMPLE_FRAME_NUM * 4
            scale_factor = 4
        elif len(ids) == 2:
            h, m = ids
            idx = (h - 1) * self.width + (m - 1)
            chunks = np.array_split(all_indices, self.width ** 2)
            part = chunks[idx]
            sample_num = MIN_SAMPLE_FRAME_NUM * 2
            scale_factor = 2
        elif len(ids) == 3:
            h, m, l = ids
            idx = (h - 1) * self.width * self.width + (m - 1) * self.width + (l - 1)
            chunks = np.array_split(all_indices, self.width ** 3)
            part = chunks[idx]
            sample_num = MIN_SAMPLE_FRAME_NUM
            scale_factor = 1
        else:
            raise ValueError(f"Unsupported ids: {ids}")

        frame_ids = np.linspace(part[0], part[-1], sample_num, dtype=int)
        return frame_ids, scale_factor

    def _get_videoqa_frame_ids(self, ids: Tuple[int, int, int]) -> np.ndarray:
        h, m, l = ids
        all_indices = np.arange(self.num_frames)
        high_chunks = np.array_split(all_indices, self.width)
        med_chunks = np.array_split(high_chunks[h - 1], self.width)
        low_chunks = np.array_split(med_chunks[m - 1], self.width)
        part = low_chunks[l - 1]
        return np.linspace(part[0], part[-1], VIDEOQA_FRAME_NUM, dtype=int)

    def _format_caption_messages(self, b64_frames: List[str], frame_time: List[float], system_prompt: str):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": []},
        ]
        for t, img in zip(frame_time, b64_frames):
            messages[-1]["content"].append({"type": "text", "text": f"Frame at {t}s"})
            messages[-1]["content"].append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img}"},
            })
        return messages

    def _format_videoqa_messages(self, b64_frames: List[str], frame_time: List[float], query: str):
        messages = [
            {"role": "system", "content": VIDEO_QA_SYSTEM_PROMPT},
            {"role": "user", "content": [{"type": "text", "text": query}]},
        ]
        for t, img in zip(frame_time, b64_frames):
            messages[-1]["content"].append({"type": "text", "text": f"Video frame at {t} seconds:"})
            messages[-1]["content"].append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img}"},
            })
        return messages

    @staticmethod
    def _init_timing_stats() -> Dict[str, float]:
        return {
            "reasoning_model_seconds": 0.0,
            "reasoning_model_calls": 0,
            "get_caption_model_seconds": 0.0,
            "get_caption_model_calls": 0,
            "video_qa_model_seconds": 0.0,
            "video_qa_model_calls": 0,
        }

    def get_caption(self, ids: Tuple[int, ...], timing_stats: Dict[str, float] = None):
        self._validate_ids(ids)
        level, key = self._segment_key(ids)
        cached = self.cache.get(level, key)
        if cached is not None:
            print(f"[get_caption] cache hit: {level} {ids}")
            return cached["caption"], cached["frame_time"]

        print(f"[get_caption] generating online: {level} {ids}")
        frame_ids, scale_factor = self._get_caption_frame_ids(ids)
        frames, frame_time = self._process_video_frames(frame_ids)

        max_len = max(140, round(512 / np.sqrt(scale_factor)))
        b64_frames = [encode_image(frame, max_len) for frame in frames]
        if level == "low":
            messages = self._format_caption_messages(b64_frames, frame_time, system_prompt=CAPTION_SYSTEM_PROMPT_LOW.format(num_frame=len(frame_ids), num_words=LOW_LEVEL_CAPTION_WORDS))
        else:
            messages = self._format_caption_messages(b64_frames, frame_time, system_prompt=CAPTION_SYSTEM_PROMPT.format(num_frame=len(frame_ids), num_words=HIGH_LEVEL_CAPTION_WORDS))

        model_start = time.time()
        resp = self.caption_client.chat.completions.create(
            model=self.caption_model,
            messages=messages,
            max_tokens=1024,
        )
        model_elapsed = time.time() - model_start
        print(f"[Timing][get_caption:model_only] {model_elapsed:.3f}s")
        if timing_stats is not None:
            timing_stats["get_caption_model_seconds"] += model_elapsed
            timing_stats["get_caption_model_calls"] += 1

        content = resp.choices[0].message.content
        parsed = extract_tags(content, ["caption"])
        caption = parsed.get("caption", content)

        self.cache.put(level, key, {
            "caption": caption,
            "frame_time": frame_time,
            "frame_ids": frame_ids.tolist(),
        })
        return caption, frame_time

    def video_qa(self, ids: Tuple[int, int, int], query: str, timing_stats: Dict[str, float] = None) -> str:
        self._validate_ids(ids)
        frame_ids = self._get_videoqa_frame_ids(ids)
        frames, frame_time = self._process_video_frames(frame_ids)
        b64_frames = [encode_image(frame, VIDEOQA_IMAGE_MAX_LENGTH) for frame in frames]

        messages = self._format_videoqa_messages(b64_frames, frame_time, query)
        model_start = time.time()
        resp = self.video_client.chat.completions.create(
            model=self.videoqa_model,
            messages=messages,
        )
        model_elapsed = time.time() - model_start
        print(f"[Timing][video_qa:model_only] {model_elapsed:.3f}s")
        if timing_stats is not None:
            timing_stats["video_qa_model_seconds"] += model_elapsed
            timing_stats["video_qa_model_calls"] += 1

        content = resp.choices[0].message.content
        parsed = extract_tags(content, ["answer"])
        return parsed.get("answer", content)

    def _build_messages(self, question: str, history_model: List[str], history_tool: List[tuple], timing_stats: Dict[str, float] = None):
        duration = self.cache.data.get("meta", {}).get("duration")
        if duration is None:
            duration = self.video_duration
            self.cache.data["meta"]["duration"] = duration
            self.cache._save()

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT.format(width=self.width)},
            {"role": "user", "content": [{"type": "text", "text": question}]},
        ]
        messages[-1]["content"].append({
            "type": "text",
            "text": f"The total duration of the video is {duration} seconds.",
        })

        for h in range(1, self.width + 1):
            caption, frame_time = self.get_caption((h,), timing_stats=timing_stats)
            messages[-1]["content"].append({
                "type": "text",
                "text": f"High-level Caption {h} from {frame_time[0]} to {frame_time[-1]}: {caption}",
            })

        for i, model_resp in enumerate(history_model):
            messages.append({
                "role": "assistant",
                "content": [{"type": "text", "text": model_resp}],
            })
            item = history_tool[i]
            if item[0] == "get_caption":
                caption, frame_time, ids = item[1], item[2], item[3]
                if len(ids) == 2:
                    t = f"Medium-level Caption from tool get_caption(({ids[0]},{ids[1]})). From {frame_time[0]} to {frame_time[-1]}: {caption}"
                else:
                    t = f"Low-level Caption from tool get_caption(({ids[0]},{ids[1]},{ids[2]})). From {frame_time[0]} to {frame_time[-1]}: {caption}"
                messages.append({"role": "user", "content": [{"type": "text", "text": t}]})
            else:
                answer = item[1]
                messages.append({
                    "role": "user",
                    "content": [{"type": "text", "text": f"<information>{answer}</information>"}],
                })

        return messages

    def answer_question(self, question: str):
        history_model = []
        history_tool = []
        video_qa_results = []
        timing_stats = self._init_timing_stats()

        for round_id in range(1, self.max_rounds + 1):
            messages = self._build_messages(question, history_model, history_tool, timing_stats=timing_stats)
            if round_id == self.max_rounds:
                messages.append({
                    "role": "user",
                    "content": [{"type": "text", "text": "This is the last round. You must provide the final answer."}],
                })

            model_start = time.time()
            resp = self.reasoning_client.chat.completions.create(
                model=self.reasoning_model,
                messages=messages,
            )
            model_elapsed = time.time() - model_start
            print(f"[Timing][reasoning:model_only][round={round_id}] {model_elapsed:.3f}s")
            timing_stats["reasoning_model_seconds"] += model_elapsed
            timing_stats["reasoning_model_calls"] += 1

            content = resp.choices[0].message.content
            print('Round:', round_id)
            print(content)
            parsed = extract_tags(content, ["think", "tool", "answer"])
            history_model.append(content)

            if "answer" in parsed:
                return {
                    "answer": parsed["answer"],
                    "history": history_model,
                    "video_qa_results": video_qa_results,
                    "timing": timing_stats,
                }

            if "tool" not in parsed:
                history_model.pop()
                continue

            func_name, args, legal = parse_function_call(parsed["tool"])
            if not legal:
                history_model.pop()
                continue

            if func_name == "get_caption":
                if not isinstance(args, (tuple, list)) or len(args) not in [2, 3]:
                    history_model.pop()
                    continue
                ids = tuple(int(x) for x in args)
                caption, frame_time = self.get_caption(ids, timing_stats=timing_stats)
                history_tool.append(("get_caption", caption, frame_time, ids, parsed["tool"]))
            elif func_name == "video_qa":
                if (
                    not isinstance(args, (tuple, list))
                    or len(args) != 2
                    or not isinstance(args[0], (tuple, list))
                    or len(args[0]) != 3
                ):
                    history_model.pop()
                    continue
                ids = tuple(int(x) for x in args[0])
                query = str(args[1])
                answer = self.video_qa(ids, query, timing_stats=timing_stats)
                video_qa_results.append(answer)
                history_tool.append(("video_qa", answer, (ids[0], ids[1], ids[2], query), parsed["tool"]))
            else:
                history_model.pop()
                continue

        return {
            "answer": None,
            "history": history_model,
            "video_qa_results": video_qa_results,
            "timing": timing_stats,
        }


def parse_args():
    parser = argparse.ArgumentParser(description="LongVideo-R1 online demo CLI")
    # parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--video_path", type=str, default="/media/Disk2/Dataset/CG-Bench/cg_videos_720p/-DAp0zXkd6A.mp4")
    parser.add_argument("--question", type=str, default='What is the man doing in this video?', help="Single question. If omitted, enter interactive mode")
    parser.add_argument("--cache_dir", type=str, default=".caption_cache")
    parser.add_argument("--max_rounds", type=int, default=30)

    parser.add_argument("--api_key", type=str, default=os.getenv("OPENAI_API_KEY", DEFAULT_API_KEY))

    parser.add_argument("--reasoning_base_url", type=str, default=DEFAULT_REASONING_BASE_URL)
    parser.add_argument("--reasoning_model", type=str, default=DEFAULT_REASONING_MODEL)

    parser.add_argument("--caption_base_url", type=str, default=DEFAULT_CAPTION_BASE_URL)
    parser.add_argument("--caption_model", type=str, default=DEFAULT_CAPTION_MODEL)

    parser.add_argument("--videoqa_base_url", type=str, default=DEFAULT_VIDEOQA_BASE_URL)
    parser.add_argument("--videoqa_model", type=str, default=DEFAULT_VIDEOQA_MODEL)
    parser.add_argument("--decode_threads", type=int, default=16, help="decord VideoReader decode threads")
    return parser.parse_args()


def main():
    args = parse_args()
    demo = LongVideoDemo(args)

    print(f"[Init] width={demo.width}, cache={demo.cache.cache_path}")
    start_time = time.time()

    if args.question is not None:
        result = demo.answer_question(args.question)
        print("\n[Question]", args.question)
        print("[Answer]", result["answer"])
        timing = result.get("timing", {})
        print("[Model Timing Summary] reasoning: {:.3f}s / {} calls, get_caption: {:.3f}s / {} calls, video_qa: {:.3f}s / {} calls".format(
            timing.get("reasoning_model_seconds", 0.0),
            int(timing.get("reasoning_model_calls", 0)),
            timing.get("get_caption_model_seconds", 0.0),
            int(timing.get("get_caption_model_calls", 0)),
            timing.get("video_qa_model_seconds", 0.0),
            int(timing.get("video_qa_model_calls", 0)),
        ))
        end_time = time.time()
        print(f"[Time] {end_time - start_time:.2f}s")
        return
    

    print("Enter question for this video (empty line to exit).")
    while True:
        q = input("\nQuestion> ").strip()
        if not q:
            break
        result = demo.answer_question(q)
        print("[Answer]", result["answer"])
        timing = result.get("timing", {})
        print("[Model Timing Summary] reasoning: {:.3f}s / {} calls, get_caption: {:.3f}s / {} calls, video_qa: {:.3f}s / {} calls".format(
            timing.get("reasoning_model_seconds", 0.0),
            int(timing.get("reasoning_model_calls", 0)),
            timing.get("get_caption_model_seconds", 0.0),
            int(timing.get("get_caption_model_calls", 0)),
            timing.get("video_qa_model_seconds", 0.0),
            int(timing.get("video_qa_model_calls", 0)),
        ))


if __name__ == "__main__":
    main()
