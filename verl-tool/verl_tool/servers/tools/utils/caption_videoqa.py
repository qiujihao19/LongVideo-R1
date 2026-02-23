import json
import os
import re
import ast
import numpy as np
from PIL import Image
from io import BytesIO
import base64
from openai import OpenAI
from decord import VideoReader, cpu
from typing import Any, Dict, Optional

IMAGE_MAX_LENGTH = 1024
IMAGE_SAMPLE_FRAME_NUM = 16

VIDEO_QA_SYSTEM_PROMPT = '''
You are a video question-answering model. You will be given a video segment and a question. Your task is to summarize the information relevant to the question based on the provided video segment.
You must think first before answering. Your thinking process should be in <reasoning></reasoning> tags. Your answer should be in <answer></answer> tags.
If the question is not related to the video, just answer "<answer>The question is not related to the video segment.</answer>".
'''


DEFAULT_TOOL_CONFIG: Dict[str, Any] = {
    "openai": {
        "api_key": "1111111",
        "base_url": "http://127.0.0.1:9081/v1",
        "videoqa_model": "Qwen2.5-VL-32B",
    },
    "data_source_paths": {
        "videocaption_cgbench": {
            "caption_dir": "path/to/cgbench_caption",
            "video_dir": "path/to/cgbench_video",
        },
        "videocaption_videomme": {
            "caption_dir": "path/to/videomme_caption",
            "video_dir": "path/to/videomme_video",
        },
    },
}

_TOOL_CONFIG_CACHE: Dict[str, Dict[str, Any]] = {}
_OPENAI_CLIENT_CACHE: Dict[str, OpenAI] = {}


def _deep_merge(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            _deep_merge(dst[key], value)
        else:
            dst[key] = value
    return dst


def _build_client_cache_key(api_key: str, base_url: str) -> str:
    return f"{api_key}@@{base_url}"


def _get_openai_client(cfg: Dict[str, Any]) -> OpenAI:
    api_cfg = cfg["openai"]
    key = _build_client_cache_key(api_cfg["api_key"], api_cfg["base_url"])
    if key not in _OPENAI_CLIENT_CACHE:
        _OPENAI_CLIENT_CACHE[key] = OpenAI(
            api_key=api_cfg["api_key"],
            base_url=api_cfg["base_url"],
        )
    return _OPENAI_CLIENT_CACHE[key]


def load_tool_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load tool configuration from JSON file.
    If config_path is None, use default hardcoded fallback.
    """
    cache_key = config_path or "__default__"
    if cache_key in _TOOL_CONFIG_CACHE:
        return _TOOL_CONFIG_CACHE[cache_key]

    cfg = json.loads(json.dumps(DEFAULT_TOOL_CONFIG))
    if config_path:
        with open(config_path, "r", encoding="utf-8") as f:
            file_cfg = json.load(f)
        cfg = _deep_merge(cfg, file_cfg)

    _TOOL_CONFIG_CACHE[cache_key] = cfg
    return cfg


def parse_function_call(call_str):
    """
    解析形如 function_name((9,1,2)) 的字符串
    返回函数名和参数（转为 Python 对象）
    """
    func_pattern = r"(\w+)\((.*)\)"
    match = re.match(func_pattern, call_str.strip())
    arg_legal = False
    if match:
        func_name = match.group(1)
        arg_str = match.group(2).strip()
        try:
            # 直接解析
            args = ast.literal_eval(arg_str) if arg_str else ()
            arg_legal = True
        except Exception as e:
            print(f"参数解析失败: {arg_str}，错误: {e}")
            args = arg_str  # 保留原字符串
            arg_legal = False
        return func_name, args, arg_legal
        # return func_name, args
    else:
        return None, None, False
    
def extract_and_validate(text):
    """
    提取 <think>、<answer>、<tool> 标签内容，并验证是否符合4种合法格式之一。
    
    参数：
        text (str): 输入字符串
    返回：
        dict: 若格式合法，返回各部分内容的字典；否则返回 {'error': '说明'}
    """
    tags = ['think', 'answer', 'tool']
    results = {}

    # 提取每个标签的内容（只取第一个）
    for tag in tags:
        pattern = fr'<{tag}>(.*?)</{tag}>'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            results[tag] = match.group(1).strip()

    return results

def encode_image(image):
    pil_image = Image.fromarray(image)
    w, h = pil_image.size
    # print(f"Original image size: {w}x{h}")
    if w > IMAGE_MAX_LENGTH or h > IMAGE_MAX_LENGTH:
        scale = min(IMAGE_MAX_LENGTH / h, IMAGE_MAX_LENGTH / w)
        new_size = (int(w * scale), int(h * scale))
        pil_image = pil_image.resize(new_size)
    # print(f"Resized image size: {pil_image.size[0]}x{pil_image.size[1]}")
    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

def get_video_duration_cuberoot(path: str) -> float:
    segment_len = 16
    vr = VideoReader(path)
    num_frames = len(vr)
    fps = vr.get_avg_fps()
    duration = num_frames / fps  
    num_segments = duration / segment_len
    width = round(num_segments ** (1/3))
    width = max(4, min(8, width))
    return width, num_frames    

def process_video_wid(video_path, frame_ids):
    video = VideoReader(video_path, ctx=cpu(0), num_threads=16)
    avg_fps = video.get_avg_fps()
    # frame_list = list(range(0, len(video), interval))
    frame_list = frame_ids
    video = video.get_batch(frame_list).asnumpy()
    video_time = [round(i / avg_fps, 1) for i in frame_list]
    return video, video_time

def sample_video_woid(video):    
    sampled_video = video  
    sampled_video = [encode_image(frame) for frame in sampled_video]
    return sampled_video

def prepare_video_wid(video_path, segment_id):
        # 1. Pre-process video into a hierarchical structure
    print("--- Preparing Video Data ---")
    # video_frames, video_time = process_video(video_path, fps=2)
    width, num_frames = get_video_duration_cuberoot(video_path)
    high_id, medium_id, low_id = segment_id
    sample_frame_num = IMAGE_SAMPLE_FRAME_NUM

    high_indices = np.array_split(range(num_frames), width)
    high_indices_chunk = high_indices[high_id - 1]
    medium_indices = np.array_split(high_indices_chunk, width)
    medium_indices_chunk = medium_indices[medium_id - 1]
    low_indices = np.array_split(medium_indices_chunk, width)
    selected_low_indices = low_indices[low_id - 1]
    low_frame_ids = np.linspace(selected_low_indices[0], selected_low_indices[-1], sample_frame_num, dtype=int)
    selected_frames, video_time = process_video_wid(video_path, low_frame_ids)
    low_segment_data = {'video': sample_video_woid(selected_frames), 
                        'time': video_time}

    print("Video data prepared successfully.")
    return low_segment_data


def video_qa_format_message(video, video_time, user_prompt, system_prompt=VIDEO_QA_SYSTEM_PROMPT):
    messages = [{"role": "system", "content": system_prompt}, 
    {"role": "user", "content": [{"type": "text", "text": user_prompt}]}]
    
    for frame_time, frame in zip(video_time, video):
        messages[-1]["content"].append({
            "type": "text",
            "text": f"Video frame at {frame_time} seconds:"
        })
        messages[-1]["content"].append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{frame}"
            }
        })
    return messages


    
def get_caption_observation(
    caption_path: str,
    segment_id: str,
):
    if not os.path.exists(caption_path):
        return None
    with open(caption_path, 'r') as f:
        caption_data = json.load(f)
    if len(caption_data['captions']) == 3:
        caption_data['captions'].insert(0, [])
    high_data = caption_data['captions'][1]
    medium_data = caption_data['captions'][2]
    
    low_data = caption_data['captions'][3]
    width = caption_data['width']

    try:
        if len(segment_id) == 2:
            high_segment_id, medium_segment_id = map(int, segment_id)
            assert 1 <= high_segment_id <= width and 1 <= medium_segment_id <= width, "IDs must be in [1, width]"
            caption = medium_data[(high_segment_id - 1) * width + medium_segment_id - 1]["caption"]
            frame_time = medium_data[(high_segment_id - 1) * width + medium_segment_id - 1]['frame_time']
            caption = f'Medium-level Caption from tool get_caption(({segment_id[0]},{segment_id[1]})). From {frame_time[0]} to {frame_time[-1]}:{caption}'
            # caption = f'Medium-level Caption from tool get_caption(({segment_id[0]},{segment_id[1]},)):{caption}'
        elif len(segment_id) == 3:
            high_segment_id, medium_segment_id, low_segment_id = map(int, segment_id)
            assert 1 <= high_segment_id <= width and 1 <= medium_segment_id <= width and 1 <= low_segment_id <= width, "IDs must be in [1, width]"
            caption = low_data[(high_segment_id - 1) * width * width + (medium_segment_id - 1) * width + low_segment_id - 1]["caption"]
            frame_time = low_data[(high_segment_id - 1) * width * width + (medium_segment_id - 1) * width + low_segment_id - 1]['frame_time']
            caption = f'Low-level Caption from tool get_caption(({segment_id[0]},{segment_id[1]},{segment_id[2]})). From {frame_time[0]} to {frame_time[-1]}:{caption}'
        else:
            return None
        return caption
    except Exception as e:
        return None 

def get_video_qa_observation(
    video_path: str,
    segment_id: str,
    query: str,
    tool_cfg: Optional[Dict[str, Any]] = None,
):  
    if not os.path.exists(video_path) or len(segment_id) != 3:
        return None
    try:
        cfg = tool_cfg or load_tool_config(None)
        client = _get_openai_client(cfg)
        videoqa_model = cfg["openai"]["videoqa_model"]

        low_segment_data = prepare_video_wid(video_path, segment_id)
        messages = video_qa_format_message(low_segment_data['video'], low_segment_data['time'], query)
        response = client.chat.completions.create(
            model=videoqa_model,
            messages=messages,
            max_tokens=1024,
        )
        video_qa_observation = response.choices[0].message.content
        video_qa_parsed = extract_and_validate(video_qa_observation)
        assert "answer" in video_qa_parsed, "Video QA did not return an answer"
        video_qa_answer = video_qa_parsed["answer"]
        video_qa_answer = f'<information>{video_qa_answer}</information>'
        return video_qa_answer
    except Exception as e:
        return None

def get_tool_observation(
    parsed_tool_calls: str,
    video_uid: str,
    data_source: str,
    width: int,
    fps: float,
    tool_cfg: Optional[Dict[str, Any]] = None,
):  
    cfg = tool_cfg or load_tool_config(None)
    func_name, args, arg_legal = parse_function_call(parsed_tool_calls)
    if not arg_legal or len(args) > 3:
        return None
    source_map = cfg.get("data_source_paths", {})
    source_cfg = source_map.get(data_source)
    if not source_cfg:
        return None
    if func_name == 'get_caption':
        caption_dir = source_cfg.get("caption_dir")
        if not caption_dir:
            return None
        caption_path = f'{caption_dir}/{video_uid}.json'
        segment_id = args
        caption = get_caption_observation(caption_path, segment_id)
        return caption
    elif func_name == 'video_qa':
        video_dir = source_cfg.get("video_dir")
        if not video_dir:
            return None
        video_path = f'{video_dir}/{video_uid}.mp4'
        segment_id, query = args
        video_qa_observation = get_video_qa_observation(video_path, segment_id, query, tool_cfg=cfg)
        return video_qa_observation
    else:
        return None
        


