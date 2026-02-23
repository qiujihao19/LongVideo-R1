# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging
import os
import tempfile
import json
import pandas as pd
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError

from verl.utils.hdfs_io import copy, makedirs

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Configuration constants
# no video qa
# VIDEO_TREE_CAPTION_SYSTEM_PROMPT = "\n## INSTRUCTIONS\nThe video is divided into four hierarchical levels: Most-coarse, High-level, Medium-level, and Low-level. Answer the given question. You must conduct reasoning inside <think> and </think> first every time before you get new information. After reasoning, if you find you lack some knowledge, you can call tool `get_caption((high_segment_id, medium_segment_id, low_segment_id))` to get 1 segments caption(`high_segment_id, medium_segment_id, low_segment_id` is from 1 to {width}). (high_segment_id, medium_segment_id, low_segment_id) is a triplet, when you get high_level_caption, the triplet should be `(high_segment_id,)`, when you get medium_level_caption, the triplet should be `(high_segment_id, medium_segment_id,)`and it will return the segment's caption. If you find no further external knowledge needed, you can provide the answer inside <answer> and </answer> after another thinking.\n"
# with video qa
# VIDEO_TREE_CAPTION_SYSTEM_PROMPT = "\n## INSTRUCTIONS\nThe video is divided into four hierarchical levels: Most-coarse, High-level, Medium-level, and Low-level. Answer the given question. You must conduct reasoning inside <think> and </think> first every time before you get new information. After reasoning, if you find you lack some knowledge, you can call tool `get_caption((high_segment_id, medium_segment_id, low_segment_id))` to get 1 segments caption(`high_segment_id, medium_segment_id, low_segment_id` is from 1 to {width}). (high_segment_id, medium_segment_id, low_segment_id) is a triplet, when you get high_level_caption, the triplet should be `(high_segment_id,)`, when you get medium_level_caption, the triplet should be `(high_segment_id, medium_segment_id,)`and it will return the segment's caption. If you think the question's answer can be found in some segments, but the caption is not detailed enough, use the tool `video_qa((high_segment_id, medium_segment_id, low_segment_id), query)` to get the relevant information of the query from the video segment. This tool can only be used after you have requested the low-level caption. If you find no further external knowledge needed, you can provide the answer inside <answer> and </answer> after another thinking.\n"
VIDEO_TREE_CAPTION_SYSTEM_PROMPT = '''
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

def process_single_row(row, current_split_name, row_index):
    """
    Process a single row of data for SearchR1-like format.

    Args:
        row: DataFrame row containing the original data
        current_split_name: Name of the current split (train/test)
        row_index: Index of the row in the DataFrame

    Returns:
        pd.Series: Processed row data in the required format
    """
    question = row.get("question")
    option = row.get("choices")
    right_answer = row.get("right_answer")
    video_uid = row.get("video_uid")
    clue_intervals = row.get("clue_intervals")
    width = row.get('width')
    duration = row.get('duration')
    fps = row.get('fps')
    
    option = [f"{chr(65+i)}. {opt}" for i, opt in enumerate(option)]
    option = '\n'.join(option)
    user_content = f"{question}\n{option}\n"

    caption_path = f'{args.caption_dir}/{video_uid}.json'
    with open(caption_path, 'r') as f:
        caption = json.load(f)
    
    # most_coarse_data = caption['captions'][0][0]
    high_caption = caption['captions'][0]
    high_caption_list = [f"High-level Caption {idx+1} from {data['frame_time'][0]} to {data['frame_time'][-1]}:{data['caption']}" for idx, data in enumerate(high_caption)]
    base_caption = "\n".join(high_caption_list)
    time_prompt = f"The total duration of the video is {duration} seconds."
    base_caption_prompt = f"{time_prompt}\n\n{base_caption}\n\n"
    user_prompt = f'{user_content}\n\n{base_caption_prompt}'

    # Build prompt structure

    prompt = [{"role": "system", "content": VIDEO_TREE_CAPTION_SYSTEM_PROMPT.format(width=width)}, {"role": "user", "content": user_prompt}]

    # Extract ground truth from reward_model or fallback to golden_answers
    reward_model_data = {
            'style': 'rule',
            'ground_truth': right_answer
        }
    if isinstance(reward_model_data, dict) and "ground_truth" in reward_model_data:
        ground_truth = reward_model_data.get("ground_truth")
    else:
        ground_truth = row.get("golden_answers", [])

    # Process data source
    data_source_tagged = "videocaption_cgbench"

    # Build tools kwargs structure
    # tools_kwargs = {
    #     "search": {
    #         "create_kwargs": {"ground_truth": ground_truth, "question": question, "data_source": data_source_tagged}
    #     }
    # }

    # Build complete extra_info structure
    extra_info = {
        "index": row_index,
        "need_tools_kwargs": True,
        "video_uid": video_uid,
        "split": current_split_name,
        "clue_intervals": clue_intervals,
        "width": width,
        "duration": duration,
        "fps": fps,
        "data_source": data_source_tagged,
        # "tools_kwargs": tools_kwargs,
    }

    return pd.Series(
        {
            "data_source": data_source_tagged,
            "prompt": prompt,
            "ability": 'get_caption',
            "reward_model": reward_model_data,
            "extra_info": extra_info,
            "metadata": row.get("metadata"),
        }
    )


def main():
    local_save_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_save_dir, exist_ok=True)

    processed_files = []

    for split in ["rl_data"]:
        parquet_filename = f"{split}.parquet"
        logger.info(f"Processing {split} split...")
        local_parquet_filepath = f'{args.data_dir}/{split}.parquet'
        # Load and process Parquet file
        df_raw = pd.read_parquet(local_parquet_filepath)
        logger.info(f"Loaded {len(df_raw)} rows from {parquet_filename}")

        def apply_process_row(row, split_name=split):
            return process_single_row(row, current_split_name=split_name, row_index=row.name)

        df_processed = df_raw.apply(apply_process_row, axis=1)

        # Save processed DataFrame
        output_file_path = os.path.join(local_save_dir, f"{split}.parquet")
        df_processed.to_parquet(output_file_path, index=False)
        logger.info(f"Saved {len(df_processed)} processed rows to {output_file_path}")
        processed_files.append(output_file_path)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process LongVideo-R1 rl data to training data.")
    parser.add_argument(
        "--data_dir", help="The directory of LongVideo-R1 rl data."
    )
    parser.add_argument(
        "--local_dir",
        help="Local directory to save the processed Parquet files.",
    )

    parser.add_argument(
        "--caption_dir", help="The directory of preprocessed captions.",
    )


    args = parser.parse_args()

    # System and user content configuration
    main()
