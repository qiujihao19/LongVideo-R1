from constant import *
from .utils import *
from openai import OpenAI
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
# define the OpenAI client

MAIN_CLIENT = OpenAI(
        api_key=API_KEY,
        base_url=REASONING_MODEL_BASE_URL
    )

VIDEO_CLIENT = OpenAI(
        api_key=API_KEY,
        base_url=VIDEOQA_BASE_URL)

def run_multistep_vqa(file_path, video_path, user_prompt, max_rounds=30):
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    width = data['width']
    high_data = data['captions'][0]
    medium_data = data['captions'][1]
    low_data = data['captions'][2]
    duration = high_data[-1]['frame_time'][-1]


    round_cnt = 0
    history_model = []
    history_caption = []
    video_qa_results = []
    
    messages = format_message(
        high_data, duration,
        user_prompt=user_prompt,
        system_prompt=SYSTEM_PROMPT.format(width=width),
        history_model=history_model,
        history_caption=history_caption,
    )
    
    while True:
        
        if round_cnt >= max_rounds:
            print("Max rounds reached without answer.")
            return None, history_model, video_qa_results
        
        if round_cnt == max_rounds - 1:
            print("This is the last round to answer the question.")
            messages.append({
                "role": "user",
                "content": [{
                    "type": "text",
                    "text": "This is the last round. You must provide the final answer in this round."
                }]
            })
        
        
        response = MAIN_CLIENT.chat.completions.create(
            model=REASONING_MODEL,
            messages=messages,
        )
        content = response.choices[0].message.content

        messages.append({
            "role": "assistant",
            "content": [{
                "type": "text",
                "text": content
            }]
        })
        
        parsed = extract_and_validate(content)

        round_cnt += 1
        print(f"\n[Round {round_cnt}]")

        # store the model response
        history_model.append(content)
        
        
        # case 1: the model has given the answer
        if "answer" in parsed and "think" in parsed:
            print("[Answer]:", parsed["answer"])
            return parsed["answer"], history_model, video_qa_results
        
        if "answer" in parsed and "think" not in parsed:
            print("No think found in the response, regenerate the response.")
            # return None, history_model
            history_model.pop()  # remove the invalid response
            messages.pop()
            messages.append({
                "role": "user",
                "content": "You must think first before answering or calling the tool."
            })
            continue

        # case 2: the model requests the tool
        if "tool" in parsed and "think" in parsed:
            func_name, args, arg_legal = parse_function_call(parsed["tool"])
            if not arg_legal:
                print('Tool Call Error:', parsed["tool"])
                messages.pop()
                history_model.pop()
                continue

            print('Tool Call:', parsed["tool"])
            if func_name == "get_caption":
                segment_id = args
                assert len(segment_id) in [2, 3], "Segment ID must be length 2 or 3."
                if len(segment_id) == 2:
                    high_segment_id, medium_segment_id = map(int, segment_id)
                    assert 1 <= high_segment_id <= width and 1 <= medium_segment_id <= width, "IDs must be in [1, width]"
                    history_caption.append(('get_caption', medium_data[(high_segment_id - 1) * width + medium_segment_id - 1]["caption"], medium_data[(high_segment_id - 1) * width + medium_segment_id - 1]['frame_time'], (high_segment_id, medium_segment_id), parsed["tool"]))
                    messages.append({
                        "role": "user",
                        "content": [{'type': "text",
                                     "text": f'Medium-level Caption from tool get_caption(({high_segment_id},{medium_segment_id})). From {medium_data[(high_segment_id - 1) * width + medium_segment_id - 1]["frame_time"][0]} to {medium_data[(high_segment_id - 1) * width + medium_segment_id - 1]["frame_time"][-1]}:{medium_data[(high_segment_id - 1) * width + medium_segment_id - 1]["caption"]}'}]
                    })
                elif len(segment_id) == 3:
                    high_segment_id, medium_segment_id, low_segment_id = map(int, segment_id)
                    assert 1 <= high_segment_id <= width and 1 <= medium_segment_id <= width and 1 <= low_segment_id <= width, "IDs must be in [1, width]"
                    # history_caption.append((low_data[(high_segment_id - 1) * width * width + (medium_segment_id - 1) * width + low_segment_id - 1]["caption"], low_data[(high_segment_id - 1) * width * width + (medium_segment_id - 1) * width + low_segment_id - 1]['frame_time'], (high_segment_id, medium_segment_id, low_segment_id), parsed["tool"]))
                    history_caption.append(('get_caption', low_data[(high_segment_id - 1) * width * width + (medium_segment_id - 1) * width + low_segment_id - 1]["caption"], low_data[(high_segment_id - 1) * width * width + (medium_segment_id - 1) * width + low_segment_id - 1]['frame_time'], (high_segment_id, medium_segment_id, low_segment_id), parsed["tool"]))
                    messages.append({
                        "role": "user",
                        "content": [{'type': "text",
                                     "text": f'Low-level Caption from tool get_caption(({high_segment_id},{medium_segment_id},{low_segment_id})). From {low_data[(high_segment_id - 1) * width * width + (medium_segment_id - 1) * width + low_segment_id - 1]["frame_time"][0]} to {low_data[(high_segment_id - 1) * width * width + (medium_segment_id - 1) * width + low_segment_id - 1]["frame_time"][-1]}:{low_data[(high_segment_id - 1) * width * width + (medium_segment_id - 1) * width + low_segment_id - 1]["caption"]}'}]
                    })
                    
            elif func_name == "video_qa":
                print(parsed["tool"])

                segment_id, query = args
                high_segment_id, medium_segment_id, low_segment_id = map(int, segment_id)
                assert 1 <= high_segment_id <= width and 1 <= medium_segment_id <= width and 1 <= low_segment_id <= width, "IDs must be in [1, width]"

                low_video_data = prepare_video_wid(video_path, segment_id, width)
                # get the low-level segment
                low_segment = low_video_data
                sampled_video = low_segment['video']
                sampled_time = low_segment['time']
                video_qa_messages = video_qa_format_message(sampled_video, sampled_time, query, VIDEO_QA_SYSTEM_PROMPT)
                video_qa_response = VIDEO_CLIENT.chat.completions.create(
                    model=VIDEOQA_MODEL,
                    messages=video_qa_messages
                )
                video_qa_content = video_qa_response.choices[0].message.content
                # history_model.append(video_qa_content)
                video_qa_results.append(video_qa_content)
                video_qa_parsed = extract_and_validate(video_qa_content)
                assert "answer" in video_qa_parsed, "Video QA did not return an answer"
                video_qa_answer = video_qa_parsed["answer"]
                history_caption.append(('video_qa', video_qa_answer, (high_segment_id, medium_segment_id, low_segment_id, query), parsed["tool"]))
                messages.append({
                    "role": "user",
                    "content": [{'type': "text",
                                 "text": f'<information>{video_qa_answer}</information>'}]
                })
            else:
                print(f"Unknown function call: {func_name}, regenerate the response.")
                messages.pop()
                history_model.pop()  # remove the invalid response
        else:
            print("No valid <answer> or <tool> found in the response, regenerate the response.")
            # return None, history_model
            messages.pop()
            history_model.pop()  # remove the invalid response
            messages.append({
                "role": "user",
                "content": "You must think first before answering or calling the tool."
            })

  

def process_one_entry(da, eval_dataset, video_base_path, caption_base_path, max_rounds=20):
    if eval_dataset == 'lvbench':
        videoid = da['key']
        file_path = f"{caption_base_path}/{videoid}.json"
        video_path = f"{video_base_path}/{videoid}.mp4"
        question = da['question']

        user_prompt = f"{question}\n"
    elif eval_dataset == 'videomme_long':
        videoid = da['videoID']
        file_path = f"{caption_base_path}/{videoid}.json"
        video_path = f"{video_base_path}/{videoid}.mp4"
        question = da['question']
        option = da['options']
        option = '\n'.join(option)
        user_prompt = f"{question}\n{option}\n"
    elif eval_dataset == 'mlvu':
        videoid = da['video_name'].split('.')[0]
        file_path = f"{caption_base_path}/{videoid}.json"
        video_path = f"{video_base_path}/{videoid}.mp4"
        question = da['question']
        user_prompt = f"{question}\n"
    else:
        raise ValueError(f"Unknown eval_dataset: {eval_dataset}")

    try:
        result, history, video_qa_results = run_multistep_vqa(
            file_path=file_path,
            video_path=video_path,
            user_prompt=user_prompt,
            max_rounds=max_rounds
        )
        da['pred'] = result
        da['output'] = history
        da['video_qa_results'] = video_qa_results
    except Exception as e:
        print(f"Error processing videoID {videoid}: {e}")
        da['pred'] = None
        da['output'] = str(e)
        da['video_qa_results'] = []

    return da


def run_all(eval_data_file, eval_dataset, video_base_path, caption_base_path, save_file, max_workers=8, max_rounds=30):
    os.makedirs(os.path.dirname(save_file), exist_ok=True)

    with open(eval_data_file, 'r') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} entries from {eval_data_file}")

    results = []
    # for da in data:
    #     process_one_entry(da, video_base_path, caption_base_path, max_rounds)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                process_one_entry, da, eval_dataset, video_base_path, caption_base_path, max_rounds
            ): da for da in data
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing {eval_data_file}"):
            result = future.result()
            results.append(result)
            with open(save_file, 'a') as f:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')

    print(f"Saved results to {save_file}")

    with open(save_file, 'r') as f:
        lines = f.readlines()
        result_data = [json.loads(line) for line in lines]
    all_correct = 0
    for da in result_data:
        if da['pred'] == da['answer']:
            all_correct += 1
    print(f"Accuracy: {all_correct}/{len(result_data)} = {all_correct/len(result_data):.4f}")
    
        
def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--eval_data_file', type=str, help='Path to the eval data file')
    parser.add_argument('--eval_dataset', type=str, choices=['lvbench','videomme_long','mlvu'], help='eval dataset')
    parser.add_argument('--caption_base_path', type=str, help='Path to the caption base directory')
    parser.add_argument('--video_base_path', type=str, help='Path to the videoqa base directory')
    parser.add_argument('--save_base_path', type=str, help='Path to the save base directory')
    parser.add_argument('--max_rounds', type=int, default=30, help='Max rounds for reasoning')
    parser.add_argument('--max_workers', type=int, default=36, help='Max workers for parallel processing')
    return parser.parse_args()
    
if __name__ == "__main__":
    args = parse_args()
    eval_data_file = args.eval_data_file
    eval_dataset = args.eval_dataset
    caption_base_path = args.caption_base_path
    video_base_path = args.video_base_path
    save_base_path = args.save_base_path
    max_rounds = args.max_rounds
    save_file = f"{save_base_path}/{eval_dataset}_{max_rounds}.jsonl"

    run_all(
        eval_data_file=eval_data_file,
        eval_dataset=eval_dataset,
        video_base_path=video_base_path,
        caption_base_path=caption_base_path,
        save_file=save_file,
        max_workers=args.max_workers,
        max_rounds=max_rounds
    )
