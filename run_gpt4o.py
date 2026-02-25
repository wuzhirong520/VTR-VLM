
max_frame_num = 50
model_name = "gpt-4o-2024-08-06"
api_key="你的OpenAI API Key"
base_url="你的OpenAI API Base URL"
max_resolution = 640

import json
from utils.video_utils import get_video_frames_for_default_adaretake
from models.vlm.apicall import API_VLM_Model

while True:
    video_path = input("Video Path: ")
    question = input("Question: ")
    options = input("Options: ")
    if options == '':
        options = []
    else:
        options = json.loads(options)

    if len(options) == 0:
        full_prompt = question + " Keep the answer short and concise."
    else:
        full_prompt = '\n'.join([
            'Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, or D) of the correct option, with no text around it.',
            question, ' '.join([f"{chr(65 + i)}. {o}" for i, o in enumerate(options)])
        ])
    print("\n")

    model = API_VLM_Model(
        model_path=model_name,
        extra_model_config={
            "api_key": api_key,
            "base_url": base_url,
            "support_video_url" : False,
            "max_resolution": max_resolution
        }
    )
    frames = get_video_frames_for_default_adaretake(video_path, max_frame_num)  
    frames = model.get_video_url(frames)
    ans = model.infer(full_prompt, {"frames" : frames})
    print("Predict Answer:", ans)