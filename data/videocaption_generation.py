from openai import OpenAI
from tqdm import tqdm

from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import json
import numpy as np
from constant import *
from utils import *




# define message formatting function
def format_message(video, video_time, sub_data, system_prompt):
    messages = [{"role": "system", "content": system_prompt.format(num_frame=len(video))},
                {"role": "user", "content": []}]

    for frame_time, frame in zip(video_time, video):
        if sub_data:
            messages[-1]["content"].append({
                "type": "text",
                "text": f"{frame_time}s. Subtitle: {','.join(sub_data[frame_time])}."
            })
        else:
            messages[-1]["content"].append({
                "type": "text",
                "text": f"{frame_time}s."
            })    
        messages[-1]["content"].append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{frame}"
            }
        })
    return messages



def process_level(
    level_data: dict,
    subs,
    client,
    model,
):  
    if subs:
        sub_data = load_subtitles(subs, level_data['time'])
    else:
        sub_data = None
    if len(level_data['video']) <= 32:
        messages = format_message(level_data['video'], level_data['time'], sub_data, CAPTION_SYSTEM_PROMPT_LOW.format(num_frame=len(level_data['video']), num_words=LOW_LEVEL_CAPTION_WORDS))
    else:
        messages = format_message(level_data['video'], level_data['time'], sub_data, CAPTION_SYSTEM_PROMPT.format(num_frame=len(level_data['video']), num_words=HIGH_LEVEL_CAPTION_WORDS))
    result = {}
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=2048
    )
    content = response.choices[0].message.content
    result = extract_and_validate(content)
    if 'caption' not in result:
        content = "<caption> " + content + " </caption>"

    return content




def process_segment(idx, segment_data: dict, subs, client, video_path, model: str = 'Qwen3-VL-32B'):
    print(f"--- Processing Level {segment_data['level']} Segment {idx + 1} ---")
    # segment_data['time'] = [float(t) for t in segment_data['time']]

    segment_data['frame_ids'] = segment_data['frame_ids'].tolist()
    video, video_time = process_video_wid(video_path, segment_data['frame_ids'])
    segment_data['video'] = sample_video_woid(video, segment_data['scale_factor'])
    segment_data['time'] = video_time
    caption = process_level(
        level_data=segment_data,
        subs=subs,
        client=client,
        model=model,
    )
    return {
        'segment_id': idx + 1,
        'level': segment_data['level'],
        'caption': caption,
        'frame_time': segment_data['time'],
        'frame_ids': segment_data['frame_ids']
    }

def caption_generation(video_path: str, subs, client, max_workers: int = 32, model: str = 'Qwen3-VL-32B'):

    print("--- Preparing Video Data ---")  
    width, num_frames = get_video_duration_cuberoot(video_path, segment_len=SEGMENT_LEN, min_width=MIN_CUBE_WIDTH, max_width=MAX_CUBE_WIDTH)
    print(f"Video width: {width}")
    sample_frame_num = MIN_SAMPLE_FRAME_NUM
    scale_factor = SCALE_FACTOR

    high_video_segment_data = []
    medium_video_segment_data = []
    low_video_segment_data = []

    # high level
    print("--- Preparing High Level Video Data ---")
    high_sample_frame_num = sample_frame_num * (scale_factor ** 2)
    part_frame_ids = np.array_split(range(num_frames), width)
    for idx, part_id in enumerate(part_frame_ids):
        frame_ids = np.linspace(part_id[0], part_id[-1], high_sample_frame_num, dtype=int)
        high_video_segment_data.append({
            'scale_factor': scale_factor ** 2,
            'frame_ids': frame_ids,
            'level': 'high'
        })
    
    # medium level
    print("--- Preparing Medium Level Video Data ---")
    medium_sample_frame_num = sample_frame_num * scale_factor
    part_frame_ids = np.array_split(range(num_frames), width**2)
    for idx, part_id in enumerate(part_frame_ids):
        frame_ids = np.linspace(part_id[0], part_id[-1], medium_sample_frame_num, dtype=int)
        medium_video_segment_data.append({
            'scale_factor': scale_factor,
            'frame_ids': frame_ids,
            'level': 'medium'
        })
    
    # low level
    print("--- Preparing Low Level Video Data ---")
    low_sample_frame_num = sample_frame_num
    part_frame_ids = np.array_split(range(num_frames), width**3)
    for idx, part_id in enumerate(part_frame_ids):
        frame_ids = np.linspace(part_id[0], part_id[-1], low_sample_frame_num, dtype=int)
        low_video_segment_data.append({
            'scale_factor': 1,
            'frame_ids': frame_ids,
            'level': 'low'
        })

    print("Video data prepared successfully.")
    
    caption_list = []
    for segment_group in [
        high_video_segment_data,
        medium_video_segment_data,
        low_video_segment_data
    ]:
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(process_segment, idx, segment_data, subs, client, video_path, model): idx
                for idx, segment_data in enumerate(segment_group)
            }
            for future in as_completed(future_to_idx):
                results.append(future.result())
        results = sorted(results, key=lambda x: x['segment_id'])
        caption_list.append(results)

    return caption_list, width

def split_data(data, idx, all_num):
    interval = round(len(data) / all_num)
    start = idx * interval
    end = (idx + 1)* interval
    return data[start:end] 


from argparse import ArgumentParser
def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--base_path', type=str, required=True, help='Path to the video data directory')
    parser.add_argument('--save_base_path', type=str, required=True, help='Path to the save caption base directory')
    parser.add_argument('--subtitle_path', type=str, required=True, help='Path to the subtitle directory')
    parser.add_argument('--use_subtitle', type=int, default=1, help='Use subtitle or not')
    parser.add_argument('--api_key', type=str, required=True, help='Your Qwen API key')
    parser.add_argument('--base_url', type=str, required=True, help='Your Qwen API base url')
    parser.add_argument('--model', type=str, default='Qwen3-VL-32B', help='Your caption model name')
    parser.add_argument('--max_workers', type=int, default=32, help='Max workers for parallel processing')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    base_path = args.base_path
    subtitle_path = args.subtitle_path
    use_subtitle = args.use_subtitle
    save_base_path = args.save_base_path
    model = args.model
    api_key = args.api_key
    base_url = args.base_url
    max_workers = args.max_workers
    
    
    client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
    
    os.makedirs(save_base_path, exist_ok=True)

    video_list = os.listdir(base_path)
    video_list = [video_name for video_name in video_list if video_name.endswith('.mp4')]
    video_list = sorted(video_list)

    has_finish_list = os.listdir(save_base_path)
    has_finish_list = [item.split('.')[0] for item in has_finish_list]

    video_list = [videoid for videoid in video_list if videoid.split('.')[0] not in has_finish_list]

    for video_name in tqdm(video_list):
        videoid = video_name.split('.')[0]
        video_path = os.path.join(base_path, video_name)
        subtitle_file = os.path.join(subtitle_path, videoid + '.srt')
        if use_subtitle == 1 and os.path.exists(subtitle_file):
            subs = read_srt(subtitle_file)
        else:
            subs = None
        caption_list, width = caption_generation(video_path, subs, client, max_workers=max_workers, model=model)
        save_data = {'video_name': video_name, 'width': width, 'captions': caption_list}
        save_path = os.path.join(save_base_path, videoid + '.json')
        with open(save_path, 'w') as f:
            json.dump(save_data, f)
