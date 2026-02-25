


from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize
import torch
from qwen_vl_utils import smart_resize
import math
import imageio.v2 as imageio
import tempfile
import subprocess
import base64

AVAILABLE_VLM_MODELS = ["qwen2.5vl", "llava", "adaretake", "adaretake_llava", "apicall", "qwen3vl"]

def get_vlm_model(model_name, model_path, extra_model_config=None):

    if model_name not in AVAILABLE_VLM_MODELS:
        raise ValueError(f"Unsupported VLM model: {model_name}")

    if model_name == "qwen2.5vl":
        from .qwen2_5vl import Qwen2_5VL_VLM_Model
        return Qwen2_5VL_VLM_Model(model_path=model_path, extra_model_config=extra_model_config)
    elif model_name == "llava":
        from .llava import LLaVA_VLM_Model
        return LLaVA_VLM_Model(model_path=model_path, extra_model_config=extra_model_config)
    elif model_name == "adaretake":
        from .adaretake import AdaReTake_VLM_Model
        return AdaReTake_VLM_Model(model_path=model_path, extra_model_config=extra_model_config)
    elif model_name == "adaretake_llava":
        from .adaretake_llava import AdaReTake_LLaVA_VLM_Model
        return AdaReTake_LLaVA_VLM_Model(model_path=model_path, extra_model_config=extra_model_config)
    elif model_name == "apicall":
        from .apicall import API_VLM_Model
        return API_VLM_Model(model_path=model_path, extra_model_config=extra_model_config)
    elif model_name == "qwen3vl":
        from .qwen3vl import Qwen3VL_VLM_Model
        return Qwen3VL_VLM_Model(model_path=model_path, extra_model_config=extra_model_config)
    

import pulp
def solve_pixel_allocation(N, P1, P2, C):
    K = len(C)
    n = [pulp.LpVariable(f"n_{i}", lowBound=0, cat="Integer") for i in range(K)]

    prob1 = pulp.LpProblem("Stage1", pulp.LpMinimize)
    prob1 += 0  # No objective, just feasibility
    prob1 += sum(n) == N, "Total_N"
    prob1 += sum(n[i] * C[i] for i in range(K)) >= P1, "Budget_Lower_Bound"
    prob1 += sum(n[i] * C[i] for i in range(K)) <= P2, "Budget_Upper_Bound"
    prob1.solve(pulp.PULP_CBC_CMD(msg=False, options=[f"randomSeed=42"]))
    max_total = sum(pulp.value(n[i]) * C[i] for i in range(K))
    prob2 = pulp.LpProblem("Stage2", pulp.LpMinimize)
    y = [pulp.LpVariable(f"y_{i}", lowBound=0) for i in range(K)] 
    y_min = pulp.LpVariable("y_min")
    y_max = pulp.LpVariable("y_max")
    for i in range(K):
        prob2 += y[i] == n[i] * C[i]
        prob2 += y_min <= y[i]
        prob2 += y[i] <= y_max
    prob2 += sum(y) == max_total
    prob2 += sum(n) == N
    prob2.setObjective(y_max - y_min)
    prob2.solve(pulp.PULP_CBC_CMD(msg=False, options=[f"randomSeed=42"]))
    n_result = [int(pulp.value(n[i])) for i in range(K)]
    return n_result

def video_to_base64(file_path):
    video_format = file_path.split('.')[-1].lower()
    mime_type = f"video/{video_format}"
    with open(file_path, 'rb') as video_file:
        video_data = video_file.read()
    base64_encoded = base64.b64encode(video_data).decode('utf-8')
    result = f"data:{mime_type};base64,{base64_encoded}"
    return result

def resize_video_for_vlm(model_name, frames_info, dynamic_resolution_config = None):
    if model_name == "llava":
        return frames_info
    elif model_name == "adaretake_llava":
        video = torch.tensor(frames_info['frames']).permute(0, 3, 1, 2)
        nframes, _, height, width = video.shape
        longsize_resolution = 682
        if max(width, height) > longsize_resolution:
            resize_factor = longsize_resolution / max(width, height)
            resized_width, resized_height = int(width * resize_factor), int(height * resize_factor)
            video = resize(
                video,
                [resized_height, resized_width],
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            ).float()
        return [{"frames" : video,}]
    elif model_name == "apicall":
        ffmpeg_params = [
            "-vf", "scale='min(720,iw)':'min(720,ih)':force_original_aspect_ratio=decrease,pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-c:v", "libx264",
            "-preset", "slow",
            "-crf", "30",
            "-pix_fmt", "yuv420p"
        ]
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as temp_video_file:
            temp_video_path = temp_video_file.name
            with imageio.get_writer(temp_video_path, format='FFMPEG', mode='I', fps=2, ffmpeg_params = ffmpeg_params) as writer:
                for frame in frames_info['frames']:
                    writer.append_data(frame)
            # subprocess.run(['cp', temp_video_path, frames_info['video_save_path']])
            video_url = video_to_base64(temp_video_path)
        frames_info['frames'] = video_url
        return frames_info
    elif model_name in ("adaretake", "qwen2.5vl", "qwen3vl"):
        video = torch.tensor(frames_info['frames']).permute(0, 3, 1, 2)
        # print(video.shape)
        nframes, _, height, width = video.shape

        image_factor = 14 * 2
        if model_name == "qwen3vl":
            image_factor = 16 * 2

        if dynamic_resolution_config is None:
            if model_name == "adaretake":
                longsize_resolution = 448
                if max(width, height) > longsize_resolution:
                    resize_factor = longsize_resolution / max(width, height)
                    resized_width, resized_height = int(width * resize_factor), int(height * resize_factor)
                    video = resize(
                        video,
                        [resized_height, resized_width],
                        interpolation=InterpolationMode.BICUBIC,
                        antialias=True,
                    ).float()
                return [{"frames" : video,}]
            else:
                min_pixels = 4 * image_factor * image_factor
                total_pixels = 20480 * image_factor * image_factor
                max_pixels = max(min(768 * image_factor * image_factor, total_pixels / nframes * 2), int(min_pixels * 1.05))
                max_pixels = min(256 * image_factor * image_factor, max_pixels)
                resized_height, resized_width = smart_resize(
                    height,
                    width,
                    factor=image_factor,
                    min_pixels=min_pixels,
                    max_pixels=max_pixels,
                )
                video = resize(
                    video,
                    [resized_height, resized_width],
                    interpolation=InterpolationMode.BICUBIC,
                    antialias=True,
                ).float()
                return [{
                    "frames" : video, 
                    **({"video_metadata" : frames_info['video_metadata']} if model_name == "qwen3vl" else {}),
                }]
        else:
            C = []
            max_size, min_size, max_total_pixels = dynamic_resolution_config['max_size'], dynamic_resolution_config['min_size'], dynamic_resolution_config['max_total_pixels']
            for i in range(100,1,-1):
                h, w = i * image_factor, round(i * width / height)*image_factor
                if h > max_size or w > max_size or h < min_size or w < min_size or h > math.ceil(height/image_factor)*image_factor or w > math.ceil(width/image_factor)*image_factor:
                    continue
                C.append([h,w])
            C = sorted(C, key=lambda x: -x[0]*x[1])
            # print(C)
            dy_n_s = solve_pixel_allocation(nframes//2, int(max_total_pixels*0.9   * nframes / frames_info['expect_frames']), int(max_total_pixels* nframes / frames_info['expect_frames']), [c[0]//image_factor * c[1]//image_factor for c in C])
            dynamic_resolutions = []
            for i in range(len(dy_n_s)):
                if dy_n_s[i] == 0:
                    continue
                ddn = 20480 * image_factor * image_factor // (C[i][0] * C[i][1])
                for j in range(0, dy_n_s[i], ddn):
                    if j + ddn > dy_n_s[i]:
                        dynamic_resolutions.append([(dy_n_s[i] - j)*2, C[i][0], C[i][1]])
                    else:
                        dynamic_resolutions.append([ddn*2, C[i][0], C[i][1]])
                # dynamic_resolutions.append([dy_n_s[i]*2, C[i][0], C[i][1]])
            print(dynamic_resolutions)
            frames, frames_idx, frames_sim = frames_info['frames'], frames_info['frames_idx'], frames_info['frames_sim']
            frame_infos = [{
                "video" : video[i],
                "simarity" : frames_sim[i],
                "global_idx" : frames_idx[i],
            } for i in range(len(frames))]
            frame_infos = sorted(frame_infos, key=lambda x: -x['simarity'])

            resized_frames = []

            last_frame_num = 0
            for i in range(len(dynamic_resolutions)):
                part_frames = sorted(frame_infos[last_frame_num:last_frame_num+dynamic_resolutions[i][0]], key = lambda x : x['global_idx'])
                last_frame_num+=dynamic_resolutions[i][0]
                mvideo = torch.stack([f['video'] for f in part_frames])
                resized_height, resized_width = dynamic_resolutions[i][1], dynamic_resolutions[i][2]
                mvideo = resize(
                    mvideo,
                    [resized_height, resized_width],
                    interpolation=InterpolationMode.BICUBIC,
                    antialias=True,
                ).float()
                resized_frames.append({
                    "shape" : [resized_height, resized_width],
                    "frames" : mvideo,
                    "global_idx" : [f['global_idx'] for f in part_frames],
                    **({"fps" : frames_info['fps']} if model_name == "qwen3vl" else {}),
                })
            
            return resized_frames
    