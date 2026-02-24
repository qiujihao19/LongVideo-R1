from openai import OpenAI
from decord import VideoReader, cpu
import numpy as np 
import argparse
import json
from collections import deque
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils import *
from constant import *

def format_message(video_caption, video_time, user_prompt, system_prompt, history_model, history_caption, emphasize=None, hints=[]):
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
    if len(history_model) == 0 and len(hints) == 0:
        return messages
    if len(history_model) == 0 and len(hints) > 0:
        for hint in hints:
            messages[-1]["content"].append({
                "type": "text",
                "text": hint[1]
            })
        return messages
    hint_ids = []
    used_ids = []
    if len(hints) > 0:
        hint_ids = [i[2] for i in hints]
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
        
        for hint in hints:
            if hint[2] == i:
                messages[-1]["content"].append({
                    "type": "text",
                    "text": hint[1]
                })
                used_ids.append(i)
    assert len(hint_ids) == len(used_ids)
      
    if emphasize:
        messages[-1]["content"].append({
            "type": "text",
            "text": emphasize
        })
    return messages



def video_qa_format_message(video, video_time, user_prompt, system_prompt):
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


def prepare_video_wid(video_path, segment_id):
    print("--- Preparing Video Data ---")
    width, num_frames = get_video_duration_cuberoot(video_path)
    high_id, medium_id, low_id = segment_id
    sample_frame_num = VIDEOQA_FRAME_NUM
    high_indices = np.array_split(range(num_frames), width)
    high_indices_chunk = high_indices[high_id - 1]
    medium_indices = np.array_split(high_indices_chunk, width)
    medium_indices_chunk = medium_indices[medium_id - 1]
    low_indices = np.array_split(medium_indices_chunk, width)
    selected_low_indices = low_indices[low_id - 1]
    low_frame_ids = np.linspace(selected_low_indices[0], selected_low_indices[-1], sample_frame_num, dtype=int)
    selected_frames, video_time = process_video_wid(video_path, low_frame_ids)
    low_segment_data = {'video': sample_video_woid(selected_frames, max_length=VIDEOQA_IMAGE_MAX_LENGTH), 
                        'time': video_time}

    print("Video data prepared successfully.")
    return low_segment_data

def judge_answer(pred, answer, gt_times, false_count, duration, width):
    if pred and pred == answer:
        return True, None
    
    gt_times = merge_intervals(gt_times) 
    options_segment_ids = get_options_segment_ids(duration, gt_times, width)
    if false_count == 3:
        return False, HINT_PROMPT_VQ
    if false_count > 3:
        return True, "Max try times"
    
    hint_ids = list(set([ids[:false_count + 1] for ids in options_segment_ids]))
    hint_messages = HINT_PROMPT.format(segment_id=hint_ids)
    return False, hint_messages
    
def run_multistep_vqa(file_path, video_path, user_prompt, duration, da, client, videoqa_client, max_rounds=20):
    
    # user_prompt = f'{PREFIX}\n{user_prompt}\n\n{SUFFIX}'
    user_prompt = f'{PREFIX}\n{user_prompt}\n'
    with open(file_path, 'r') as f:
        data = json.load(f)
    width = data['width']
    high_data = data['captions'][0]
    medium_data = data['captions'][1]
    low_data = data['captions'][2]
    gt = da['right_answer']
    gt_time = da['clue_intervals']

    queue = deque()
    queue.append({
        "video_caption": high_data,
        "video_time": duration,
        "system_prompt": GPT_PROMPT.format(width=width),
    })

    round_cnt = 0
    history_model = []
    history_caption = []
    video_qa_results = []
    hints = []
    false_count = 0
    
    while queue and round_cnt < max_rounds:
        task = queue.popleft()
        system_prompt = task["system_prompt"]
        video_time = task["video_time"]
        video_caption = task.get("video_caption", None)

        messages = format_message(
            video_caption, video_time,
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            history_model=history_model,
            history_caption=history_caption,
            emphasize=task.get("emphasize", None),
            hints=hints,
        )
        
        response = client.chat.completions.create(
            model=GPT_MODEL,
            messages=messages
        )
        content = response.choices[0].message.content
        parsed = extract_and_validate(content, tags=["answer", "think", "tool"])

        round_cnt += 1
        print(f"\n[Round {round_cnt}]")
        
        history_model.append(content)
        
        
        if "answer" in parsed and "think" in parsed:
            is_right, hint_messages = judge_answer(parsed["answer"], gt, gt_time, false_count, duration, width)
            if is_right:
                print("Right answer: ", parsed['answer']) if hint_messages is None else print("Max retry, False answer: ", parsed['answer'])
                return parsed["answer"], history_model, video_qa_results
            else:
                false_count += 1
            queue.append({
                "video_caption": high_data,
                "video_time": duration,
                "system_prompt": GPT_PROMPT.format(width=width),
                "emphasize": None
            })
            history_model.pop() 
            round_cnt -= 1
            hints.append(('hint', hint_messages, max(0, round_cnt - 1)))
            continue                
        
        if "answer" in parsed and "think" not in parsed:
            print("No think found in the response, regenerate the response.")
            # return None, history_model
            queue.append({
                "video_caption": high_data,
                "video_time": duration,
                "system_prompt": GPT_PROMPT.format(width=width),
                "emphasize": "You must think first before answering or calling the tool."
            })
            history_model.pop()  
            round_cnt -= 1
            continue


        if "tool" in parsed and "think" in parsed:
            func_name, args, arg_legal = parse_function_call(parsed["tool"])
            if not arg_legal:
                print('Tool Call Error:', parsed["tool"])
                queue.append({
                    "video_caption": high_data,
                    "video_time": duration,
                    "system_prompt": GPT_PROMPT.format(width=width),
                })
                history_model.pop()
                round_cnt -= 1
                continue

            print('Tool Call:', parsed["tool"])
            if func_name == "get_caption":
                segment_id = args
                assert len(segment_id) in [2, 3], "Segment ID must be length 2 or 3."
                if len(segment_id) == 2:
                    high_segment_id, medium_segment_id = map(int, segment_id)
                    assert 1 <= high_segment_id <= width and 1 <= medium_segment_id <= width, "IDs must be in [1, width]"
                    history_caption.append(('get_caption', medium_data[(high_segment_id - 1) * width + medium_segment_id - 1]["caption"], medium_data[(high_segment_id - 1) * width + medium_segment_id - 1]['frame_time'], (high_segment_id, medium_segment_id), parsed["tool"]))
                elif len(segment_id) == 3:
                    high_segment_id, medium_segment_id, low_segment_id = map(int, segment_id)
                    assert 1 <= high_segment_id <= width and 1 <= medium_segment_id <= width and 1 <= low_segment_id <= width, "IDs must be in [1, width]"
                    history_caption.append(('get_caption', low_data[(high_segment_id - 1) * width * width + (medium_segment_id - 1) * width + low_segment_id - 1]["caption"], low_data[(high_segment_id - 1) * width * width + (medium_segment_id - 1) * width + low_segment_id - 1]['frame_time'], (high_segment_id, medium_segment_id, low_segment_id), parsed["tool"]))
                queue.append({
                    "video_caption": high_data,
                    "video_time": duration,
                    "system_prompt": GPT_PROMPT.format(width=width),
                })
            elif func_name == "video_qa":

                segment_id, query = args
                high_segment_id, medium_segment_id, low_segment_id = map(int, segment_id)
                assert 1 <= high_segment_id <= width and 1 <= medium_segment_id <= width and 1 <= low_segment_id <= width, "IDs must be in [1, width]"

                low_video_data = prepare_video_wid(video_path, segment_id)
                low_segment = low_video_data
                sampled_video = low_segment['video']
                sampled_time = low_segment['time']
                video_qa_messages = video_qa_format_message(sampled_video, sampled_time, query, VIDEO_QA_SYSTEM_PROMPT)
                video_qa_response = videoqa_client.chat.completions.create(
                    model=VIDEOQA_MODEL,
                    messages=video_qa_messages
                )
                video_qa_content = video_qa_response.choices[0].message.content
                video_qa_results.append(video_qa_content)
                video_qa_parsed = extract_and_validate(video_qa_content, tags=["answer", "think"])
                assert "answer" in video_qa_parsed, "Video QA did not return an answer"
                video_qa_answer = video_qa_parsed["answer"]
                history_caption.append(('video_qa', video_qa_answer, (high_segment_id, medium_segment_id, low_segment_id, query), parsed["tool"]))
                queue.append({
                    "video_caption": high_data,
                    "video_time": duration,
                    "system_prompt": GPT_PROMPT.format(width=width),
                })
            else:
                print(f"Unknown function call: {func_name}, regenerate the response.")
                queue.append({
                    "video_caption": high_data,
                    "video_time": duration,
                    "system_prompt": GPT_PROMPT.format(width=width),
                })
                history_model.pop()  
        else:
            print("No valid <answer> or <tool> found in the response, regenerate the response.")
            # return None, history_model
            queue.append({
                "video_caption": high_data,
                "video_time": duration,
                "system_prompt": GPT_PROMPT.format(width=width),
                "emphasize": "You must think first before answering or calling the tool."
            })
            history_model.pop() 
    print("Max rounds reached without answer.")
    return None, history_model, video_qa_results
  


def process_one_entry(da, client, videoqa_client, caption_base_path, video_base_path, max_rounds=20):
    videoid = da['video_uid']
    file_path = f"{caption_base_path}/{videoid}.json"
    video_path = f"{video_base_path}/{videoid}.mp4"
    question = da['question']
    option = da['choices']
    option = [f"{chr(65+i)}. {opt}" for i, opt in enumerate(option)]
    duration = da['duration'] 
    option = '\n'.join(option)
    user_prompt = f"{question}\n{option}\n"

    try:
        result, history, video_qa_results = run_multistep_vqa(
            file_path=file_path,
            video_path=video_path,
            user_prompt=user_prompt,
            duration=duration,
            da=da,
            client=client,
            videoqa_client=videoqa_client,
            max_rounds=max_rounds
        )
        da['pred'] = result
        da['output'] = history
        da['video_qa_results'] = video_qa_results
    except Exception as e:
        da['pred'] = None
        da['output'] = str(e)
        da['video_qa_results'] = []
        print(f"Error processing video {videoid}: {e}")

    return da


def run_all(file_path, client, videoqa_client, caption_base_path, video_base_path, save_base_path, max_workers=8, max_rounds=30):
    os.makedirs(save_base_path, exist_ok=True)
    with open(file_path, 'r') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} entries from {file_path}")
    file_name = os.path.basename(file_path).split('.')[0]
    save_file = f"{save_base_path}/{file_name}_{GPT_MODEL}.jsonl"
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                process_one_entry, da, client, videoqa_client, caption_base_path, video_base_path, max_rounds
            ): da for da in data
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing {file_name}"):
            result = future.result()
            results.append(result)

            with open(save_file, 'a') as f:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')

    print(f"Saved results to {save_file}")
        
def parse_args():
    parser = argparse.ArgumentParser(description="Run multi-step VQA on CG-Bench videos.")
    parser.add_argument("--file_path", type=str, required=True, help="Data file path.")
    parser.add_argument("--video_base_path", type=str, required=True, help="Path to the video directory.")
    parser.add_argument("--caption_base_path", type=str, required=True, help="Path to the caption directory.")
    parser.add_argument("--save_base_path", type=str, required=True, help="Path to save the results.")
    parser.add_argument("--max_workers", type=int, default=8, help="Number of threads to use.")
    parser.add_argument("--max_rounds", type=int, default=30, help="Maximum number of rounds for VQA.")
    parser.add_argument("--gpt_api_key", type=str, required=True, help="GPT API key.")
    parser.add_argument("--gpt_api_base", type=str, default="https://api.openai.com/v1", help="GPT API base URL.")
    parser.add_argument("--qwen_api_key", type=str, required=True, help="Local Qwen API key. For example, '111111'.")
    parser.add_argument("--qwen_api_base", type=str, required=True, help="Local Qwen API base URL. For example, 'http://127.0.0.1:8000/v1'.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    file_path = args.file_path
    video_base_path = args.video_base_path
    caption_base_path = args.caption_base_path
    save_base_path = args.save_base_path
    max_workers = args.max_workers
    max_rounds = args.max_rounds
    gpt_api_key = args.gpt_api_key
    gpt_api_base = args.gpt_api_base
    qwen_api_key = args.qwen_api_key
    qwen_api_base = args.qwen_api_base

    client = OpenAI(
        api_key=gpt_api_key,
        base_url=gpt_api_base,
    )

    videoqa_client = OpenAI(
        api_key=qwen_api_key,
        base_url=qwen_api_base,
    )


    run_all(
        file_path=file_path,
        client=client,
        videoqa_client=videoqa_client,
        video_base_path=video_base_path,
        caption_base_path=caption_base_path,
        save_base_path=save_base_path,
        max_workers=max_workers,
        max_rounds=max_rounds
    )
