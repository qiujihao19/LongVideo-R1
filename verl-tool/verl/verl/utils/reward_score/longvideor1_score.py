# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 Search-R1 Contributors
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
# Adapted from https://github.com/PeterGriffinJin/Search-R1/blob/main/verl/utils/reward_score/qa_em.py

import random
import re
import ast
import string

def interval_length(iv):
    return max(0.0, iv[1] - iv[0])

def intersection(a, b):
    s = max(a[0], b[0])
    e = min(a[1], b[1])
    return max(0.0, e - s)

def total_intersection_with_gt(visited_intervals, gt_intervals):
    total = 0.0
    for v in visited_intervals:
        for g in gt_intervals:
            total += intersection(v, g)
    return total

def union_visited_length(visited_intervals):
    if not visited_intervals:
        return 0.0
    segs = sorted(visited_intervals, key=lambda x: x[0])
    merged = []
    cur_s, cur_e = segs[0]
    for s,e in segs[1:]:
        if s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))
    return sum(interval_length(x) for x in merged)

def gt_total_length(gt_intervals):
    return sum(interval_length(g) for g in gt_intervals)


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



def segment_to_range(seg, duration, width):
    L_high = duration / width
    L_med = duration / (width ** 2)
    L_low = duration / (width ** 3)
    if len(seg) == 1:
        start = (seg[0] - 1) * L_high
        return [start, min(duration, start + L_high)]
    elif len(seg) == 2:
        start = (seg[0] - 1) * L_high + (seg[1] - 1) * L_med
        return [start, min(duration, start + L_med)]
    elif len(seg) == 3:
        start = (seg[0] - 1) * L_high + (seg[1] - 1) * L_med + (seg[2] - 1) * L_low
        return [start, min(duration, start + L_low)]
    else:
        # raise ValueError("invalid seg length")
        return [0,0]


def extract_tool_call(solution_str):
    """Extract the tool call from the solution string."""
    tool_pattern = r"<tool>(.*?)</tool>"
    matches = re.findall(tool_pattern, solution_str, re.DOTALL)
    if matches:
        results = [m.strip() for m in matches]
    elif len(matches) < 1:
        results = None
    return results

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
    else:
        return None, None, False


def extract_solution(solution_str):
    """Extract the equation from the solution string."""

    answer_pattern = r"<answer>(.*?)</answer>"
    match = re.finditer(answer_pattern, solution_str, re.DOTALL)
    matches = list(match)

    # If there are 0  matches, return None
    if len(matches) < 1:
        return None

    # If there are 2 or more matches, return the last one
    return matches[-1].group(1).strip()


def compute_score_wtime_gt(solution_str, ground_truth, extra_info):
    """The scoring function for exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    w_repeat = -0.2
    w_f1 = 1.0
    w_answer = 1.0

    answer_reward = 0
    f1_reward = 0
    repeat_reward = 0
    
    answer = extract_solution(solution_str=solution_str)
    
    clue_intervals = extra_info.get('clue_intervals')
    width = extra_info.get('width')
    duration = extra_info.get('duration')

    if answer is None:
        answer_reward += 0
    else:
        if answer == ground_truth:
            answer_reward += 1
        else:
            answer_reward += 0
    
    try:
    
        tool_calls = extract_tool_call(solution_str=solution_str)
        tool_call_segments = []
        if tool_calls is None:
            rewards = {
                "score": w_answer * answer_reward,
                "answer_reward": answer_reward,
                "f1_reward": f1_reward,
                "repeat_reward": repeat_reward,
            }
            return rewards
        
        for tool_call in tool_calls:
            func_name, args, arg_legal = parse_function_call(tool_call)
            if func_name == 'get_caption':
                if arg_legal and len(args) < 4:
                    tool_call_segments.append(args)
        
        if len(tool_call_segments) == 0:
            rewards = {
                "score": w_answer * answer_reward,
                "answer_reward": answer_reward,
                "f1_reward": f1_reward,
                "repeat_reward": repeat_reward,
            }
            return rewards
        
        repeat_reward = (len(tool_call_segments) - len(set(tool_call_segments))) / len(set(tool_call_segments))

        merged_clue_intervals = merge_intervals(clue_intervals)
        visited_intervals = [segment_to_range(seg, duration, width) for seg in tool_call_segments]
        # 基础量
        gt_len = gt_total_length(merged_clue_intervals)  # 可能为0（保护）
        inter_len = total_intersection_with_gt(visited_intervals, merged_clue_intervals)
        visited_len = union_visited_length(visited_intervals)

        # Coverage
        cov = (inter_len / gt_len) if gt_len > 0 else 0.0
        # Precision
        prec = (inter_len / visited_len) if visited_len > 0 else 0.0
        # F1-style
        if cov + prec > 0:
            f1_reward = 2 * cov * prec / (cov + prec)
        else:
            f1_reward = 0.0
        rewards = {
                "score": w_answer * answer_reward + w_f1 * f1_reward + w_repeat * repeat_reward,
                "answer_reward": answer_reward,
                "f1_reward": f1_reward,
                "repeat_reward": repeat_reward,
            }
        return rewards
    
    except Exception as e:
        rewards = {
                "score": w_answer * answer_reward + w_f1 * f1_reward + w_repeat * repeat_reward,
                "answer_reward": answer_reward,
                "f1_reward": f1_reward,
                "repeat_reward": repeat_reward,
            }
        return rewards
    



def compute_score_only_answer(solution_str, ground_truth, extra_info):
    """The scoring function for exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """

    
    answer = extract_solution(solution_str=solution_str)
    
    if answer is None:
        return 0
    else:
        if answer == ground_truth:
            return 1
        else:
            return 0

def compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
    memory_limit_mb=None,
):
    """Compute the score for a given solution based on the data source.

    Args:
        data_source (str): The source dataset identifier which determines the scoring method.
        solution_str (str): The solution string to be evaluated.
        ground_truth (str): The ground truth answer for comparison.
        extra_info (dict, optional): Additional information that might be needed for scoring. Defaults to None.

    Returns:
        float: The computed score as a floating point number. If the result is a dictionary,
               it returns the dictionary instead.

    Raises:
        NotImplementedError: If the reward function is not implemented for the given data source.
    """
    if data_source == 'videocaption_cgbench':
        res = compute_score_wtime_gt(solution_str, ground_truth, extra_info)
    elif data_source == 'videocaption_videomme':
        res = compute_score_only_answer(solution_str, ground_truth, extra_info)
    else:
        raise NotImplementedError(f"Reward function is not implemented for {data_source=}")

    if isinstance(res, dict):
        return res
    elif isinstance(res, int | float | bool):
        return float(res)
    else:
        return float(res[0])

    
            
    
    
    

