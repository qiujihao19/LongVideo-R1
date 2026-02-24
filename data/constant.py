# minimum number of sampled frames for caption generation
MIN_SAMPLE_FRAME_NUM = 32
# scale factor for caption generation
SCALE_FACTOR = 2
# image size limits for caption generation
CAPTION_IMAGE_MAX_LENGTH = 512
CAPTION_IMAGE_MIN_LENGTH = 28 * 5
# video segment length in seconds
SEGMENT_LEN = 16
# minimum and maximum width for video segmentation
MIN_CUBE_WIDTH = 4
MAX_CUBE_WIDTH = 8

# model for sft data generation
GPT_MODEL = 'gpt-5'
# model for video question answering for your local server
VIDEOQA_MODEL = 'Qwen-3-VL-32B'
# image size limits for video question answering
VIDEOQA_IMAGE_MAX_LENGTH = 1024
# number of frames for video question answering
VIDEOQA_FRAME_NUM = 16

LOW_LEVEL_CAPTION_WORDS = 200
HIGH_LEVEL_CAPTION_WORDS = 400

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
    Your output should be controlled within {num_word} words.
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



GPT_PROMPT = '''
You are a reasoning assistant designed to answer questions about a long video through hierarchical captions. 
The video is organized into three levels of temporal granularity:

1. **High-level**: The video is divided into {width} major segments.  
   - Each segment contains one caption summarizing its content.  
2. **Medium-level**: Each High-level segment is further divided into {width} sub-segments.  
   - Each sub-segment contains one caption.  
3. **Low-level**: Each Medium-level segment is further divided into {width} finer sub-segments.  
   - Each sub-segment contains one caption.

---

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
     - You can also input the options into the tool.
     - You may **only use `video_qa`** after you have already retrieved the corresponding Low-level caption for that segment.

4. **Restriction:**  
   In each reasoning round, you may only call **one tool** (either `get_caption` or `video_qa`) **once** to obtain new information.

---

### Output Format
Your reasoning and actions must follow this structure exactly:
<think>Your internal reasoning process here. Analyze what information you have, what is missing, and which part might be relevant.</think>
<tool>(get_caption or video_qa call here, if needed)</tool>  
or  
<think>...</think>
<answer>Your final answer here (only when you are confident the information is sufficient).</answer>

Remember:
- Always include <think> before <tool> or <answer>.  
- Never skip the reasoning step.  
- There are many kinds of videos, like movies, sports, documentaries, animations, etc. 
- video_qa is a powerful tool; use it wisely when captions are insufficient.
- Details such as color information, object attributes, and specific actions are often found in Low-level segments. Get those captions or video clips when needed.
- Do not produce extra explanations outside these tags.
- Only answer when you are sure you have enough information.
- When you can not get the relevant information, you can try to scan the captions of the entire video.
- The caption may contain some errors, you can use video_qa to check the correctness of the caption.
- 
'''
PREFIX = 'Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, or D) of the correct option.'


VIDEO_QA_SYSTEM_PROMPT = '''
You are a video question-answering model. You will be given a video segment and a question. Your task is to summarize the information relevant to the question based on the provided video segment.
You must think first before answering. Your thinking process should be in <think></think> tags. Your answer should be in <answer></answer> tags.
If the question is not related to the video, just answer "<answer>The question is not related to the video segment.</answer>".
'''
HINT_PROMPT = '''
It seems that the previous attempt to answer the question was incorrect.  
To assist you, I will now provide a hint.

The hint is as follows:
> The segment that contains information relevant to this question is **{segment_id}**.

Use this hint to guide your reasoning and decide which part of the video to explore next.  
However, **do not directly reference or restate this hint inside your reasoning process**.  
You must act as if you are inferring this from your own analysis, while the hint merely helps you focus your attention.

Continue reasoning and proceed with your usual workflow to reach a correct answer.
'''
HINT_PROMPT_VQ = '''
It appears that multiple hinting attempts have not led to a correct answer.  
This suggests the textual captions you have been using may be incomplete or contain errors.  
You may need to call video_qa to check the correctness of the caption.
Do not directly reference or restate this hint inside your reasoning process.  
You must act as if you are inferring this from your own analysis, while the hint merely helps you focus your attention.

Continue reasoning and proceed with your usual workflow to reach a correct answer.
'''