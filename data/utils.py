import base64
from PIL import Image
from io import BytesIO
from decord import VideoReader, cpu
import pysrt
import math
import re
import ast

# parse function call from string
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
            args = arg_str  
            arg_legal = False
        return func_name, args, arg_legal
    else:
        return None, None, False
    
    
def extract_and_validate(text, tags = ['caption']):
    results = {}
    for tag in tags:
        pattern = fr'<{tag}>(.*?)</{tag}>'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            results[tag] = match.group(1).strip()

    return results


def process_video_wid(video_path, frame_ids):
    video = VideoReader(video_path, ctx=cpu(0), num_threads=16)
    avg_fps = video.get_avg_fps()
    frame_list = frame_ids
    video = video.get_batch(frame_list).asnumpy()
    video_time = [round(i / avg_fps, 1) for i in frame_list]
    return video, video_time

def encode_image(image, max_length):
    pil_image = Image.fromarray(image)
    w, h = pil_image.size
    if w > max_length or h > max_length:
        scale = min(max_length / h, max_length / w)
        new_size = (round(w * scale), round(h * scale))
        pil_image = pil_image.resize(new_size)
    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

def sample_video_woid(video, scale_factor=1, max_length=512, min_length=140):    
    sampled_video = video  
    down_max_length = round(max_length / math.sqrt(scale_factor))
    if down_max_length < min_length:
        down_max_length = min_length
    sampled_video = [encode_image(frame, down_max_length) for frame in sampled_video]
    return sampled_video


def get_video_duration_cuberoot(path, segment_len=16, min_width=4, max_width=8):
    vr = VideoReader(path)
    num_frames = len(vr)
    fps = vr.get_avg_fps()
    duration = num_frames / fps  
    num_segments = duration / segment_len
    width = round(num_segments ** (1/3))
    width = max(min_width, min(max_width, width))
    return width, num_frames   


def read_srt(file_path):
    subs = pysrt.open(file_path, encoding="utf-8")
    results = []
    for sub in subs:
        if '<' and '>' in sub.text:
            clean_text = re.sub(r"<.*?>", "", sub.text).strip()
        else:
            clean_text = sub.text.strip()
        results.append({
            "index": sub.index,
            "start": sub.start.ordinal / 1000.0,  
            "end": sub.end.ordinal / 1000.0,
            "text": clean_text
        })
    return results


def load_subtitles(subs: list, video_time: list):
    subtitle_data = {}
    for time in video_time:
        sub_data = []
        for sub in subs:
            if sub['start'] <= time <= sub['end']:
                sub_data.append(sub['text'])
        subtitle_data[time] = sub_data
    return subtitle_data

# merge intervals with gap <= 5 seconds
def merge_intervals(intervals):
    if not intervals: return []
    if len(intervals) == 1: return intervals
    intervals = sorted(intervals, key=lambda x: x[0])
    id_list = [0 for _ in range(len(intervals))]
    id_init = 0
    for i in range(len(intervals) - 1):
        if intervals[i+1][0] - intervals[i][1] <= 5:
            id_list[i+1] = id_init
        else:
            id_init += 1
            id_list[i+1] = id_init
    
    merged = []
    for i in range(id_init + 1):
        cur_intervals = [intervals[j] for j in range(len(intervals)) if id_list[j] == i]
        merged.append([min([x[0] for x in cur_intervals]), max([x[1] for x in cur_intervals])])
    return merged

# compute timestamp id
def get_id(std, duration, width):
    L_high = duration / (width)
    L_med  = duration / (width**2)
    L_low  = duration / (width**3)
    high_id = int(std // L_high) + 1
    med_id = int(std % L_high // L_med) + 1
    low_id = int(std % L_high % L_med // L_low) + 1
    return (high_id, med_id, low_id)


# get covered ids between std_id and end_id
def get_covered_ids(std_id, end_id, width):
    covered = []
    for h in range(std_id[0], end_id[0]+1):
        med_start = std_id[1] if h == std_id[0] else 1
        med_end = end_id[1] if h == end_id[0] else width
        for m in range(med_start, med_end+1):
            low_start = std_id[2] if (h == std_id[0] and m == std_id[1]) else 1
            low_end = end_id[2] if (h == end_id[0] and m == end_id[1]) else width
            for l in range(low_start, low_end+1):
                covered.append((h, m, l))
    return covered


# get options segment ids between all std_id and end_id
def get_options_segment_ids(duration, clues, width):
    options_segment_ids = []
    for clue in clues:
        std, end = clue
        std_id = get_id(std, duration, width)
        end_id = get_id(end, duration, width)
        covered_ids = get_covered_ids(std_id, end_id, width)
        options_segment_ids = options_segment_ids + covered_ids
    
    return options_segment_ids