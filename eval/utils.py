import base64
from PIL import Image
from io import BytesIO
from decord import VideoReader, cpu
import numpy as np 
from argparse import ArgumentParser
import re
import ast
import os
from constant import *

def process_video_wid(video_path, frame_ids):
    video = VideoReader(video_path, ctx=cpu(0), num_threads=16)
    avg_fps = video.get_avg_fps()
    # frame_list = list(range(0, len(video), interval))
    frame_list = frame_ids
    video = video.get_batch(frame_list).asnumpy()
    video_time = [round(i / avg_fps, 1) for i in frame_list]
    return video, video_time

def encode_image(image_path):
    pil_image = Image.fromarray(image_path)
    w, h = pil_image.size
    # print(f"Original image size: {w}x{h}")
    if w > VIDEOQA_IMAGE_MAX_LENGTH or h > VIDEOQA_IMAGE_MAX_LENGTH:
        scale = min(VIDEOQA_IMAGE_MAX_LENGTH / h, VIDEOQA_IMAGE_MAX_LENGTH / w)
        new_size = (int(w * scale), int(h * scale))
        pil_image = pil_image.resize(new_size)
    # print(f"Resized image size: {pil_image.size[0]}x{pil_image.size[1]}")
    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str


def sample_video_woid(video):    
    sampled_video = video  
    sampled_video = [encode_image(frame) for frame in sampled_video]


def extract_and_validate(text):
    tags = ['think', 'answer', 'tool']
    results = {}
    for tag in tags:
        pattern = fr'<{tag}>(.*?)</{tag}>'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            results[tag] = match.group(1).strip()

    return results

def parse_function_call(call_str):
    func_pattern = r"(\w+)\((.*)\)"
    match = re.match(func_pattern, call_str.strip())
    arg_legal = False
    if match:
        func_name = match.group(1)
        arg_str = match.group(2).strip()
        try:
            args = ast.literal_eval(arg_str) if arg_str else ()
            arg_legal = True
        except Exception as e:
            print(f"Error parse function call: {arg_str}, {e}")
            args = arg_str  
            arg_legal = False
        return func_name, args, arg_legal
    else:
        return None, None, False


def format_message(video_caption, video_time, user_prompt, system_prompt, history_model, history_caption, emphasize=None):
    messages = [{"role": "system", "content": system_prompt}, 
    {"role": "user", "content": [{"type": "text", "text": user_prompt}]}]
    
    messages[-1]["content"].append({
                "type": "text",
                "text": f"The total duration of the video is {video_time} seconds."
            })   
    for idx, caption_data in enumerate(video_caption):
        caption = caption_data['caption']
        time = caption_data['frame_time']
        messages[-1]["content"].append({
                    "type": "text",
                    "text": f"High-level Caption {idx+1} from {time[0]} to {time[-1]}:{caption}"
                })
    assert len(history_model) == len(history_caption)
    if len(history_model) == 0:
        return messages
    for i in range(len(history_model)):
        content = [{"type": "text", "text": history_model[i]}]
        messages.append({
            "role": "assistant",
            "content": content
        })
        cap_time = history_caption[i]
        if cap_time[0] == 'get_caption':
            caption = cap_time[1]
            frame_time = cap_time[2]
            segment_id = cap_time[3]

            if len(segment_id) == 3:
                content = [{
                    "type": "text",
                    "text": f'Low-level Caption from tool get_caption(({segment_id[0]},{segment_id[1]},{segment_id[2]})). From {frame_time[0]} to {frame_time[-1]}:{caption}'
                }]   
            elif len(segment_id) == 2:
                content = [{
                    "type": "text",
                    "text": f'Medium-level Caption from tool get_caption(({segment_id[0]},{segment_id[1]},)). From {frame_time[0]} to {frame_time[-1]}:{caption}'
                }]  
            else:
                AssertionError("Segment ID length must be 2 or 3.")
            messages.append({
                "role": "user",
                "content": content
            })
        elif cap_time[0] == 'video_qa':
            video_qa_content = cap_time[1]
            content = [{
                "type": "text",
                "text": f'<information>{video_qa_content}</information>'
            }]  
            messages.append({
                "role": "user",
                "content": content
            })
        else:
            AssertionError("Unknown caption type.")

        
    if emphasize:
        messages[-1]["content"].append({
            "type": "text",
            "text": emphasize
        })
    return messages

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

def get_video_duration_cuberoot(path, segment_len=16, min_width=4, max_width=8):
    vr = VideoReader(path)
    num_frames = len(vr)
    fps = vr.get_avg_fps()
    duration = num_frames / fps  
    num_segments = duration / segment_len
    width = round(num_segments ** (1/3))
    width = max(min_width, min(max_width, width))
    return width, num_frames  



def prepare_video_wid(video_path, segment_id, width):
        # 1. Pre-process video into a hierarchical structure
    print("--- Preparing Video Data ---")
    # video_frames, video_time = process_video(video_path, fps=2)
    _ , num_frames = get_video_duration_cuberoot(video_path)
    high_id, medium_id, low_id = segment_id
    sample_frame_num = VIDEOQA_FRAME_NUM
    # Divide into 4 medium segments
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

def get_video_qa_observation(
    video_path: str,
    segment_id: str,
    query: str,
    width: int,
    client,
    model
):  

    image_list = os.listdir(video_path)
    image_list = sorted(image_list, key=lambda x: int(x.split('_')[0]))
    image_chunk = np.array_split(image_list, width ** 3)
    high_segment_id, medium_segment_id, low_segment_id = map(int, segment_id)
    image_chunk = image_chunk[(high_segment_id - 1) * width * width + (medium_segment_id - 1) * width + low_segment_id - 1]
    frame_time = [int(img.split('_')[1].split('.')[0]) for img in image_chunk]
    image_chunk = [os.path.join(video_path, img) for img in image_chunk]
    video = [encode_image(img) for img in image_chunk]
    messages = video_qa_format_message(video, frame_time, query)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=1024,
    )
    video_qa_observation = response.choices[0].message.content

    return video_qa_observation
