
import numpy as np
from PIL import Image

# import ffmpeg
# def get_video_info(video_path : str):
#     probe = ffmpeg.probe(video_path)
#     video_info = next(stream for stream in probe['streams'] if stream['codec_type'] == 'video')
#     total_duration = float(video_info['duration'])
#     width = int(video_info['width'])
#     height = int(video_info['height'])
#     fps = eval(video_info['avg_frame_rate'])
#     return total_duration, fps, width, height

# def get_video_frames_by_fps(video_path : str, fps : float, width : int, height : int):
#     frames = (
#         ffmpeg.input(video_path)
#         .filter("fps", fps=fps)
#         .output("pipe:", format="rawvideo", pix_fmt="rgb24")
#         .run(capture_stdout=True, capture_stderr=True) 
#     )
#     frames = np.frombuffer(frames[0], np.uint8).reshape(-1, height, width, 3)
#     return frames

import decord

def get_video_frames_by_fps(video_path : str, fps : float):
    # vr = decord.VideoReader(video_path, ctx=decord.cpu(0), num_threads=1)
    vr = decord.VideoReader(video_path)
    total_frames = len(vr)
    num_frames = round(total_frames / vr.get_avg_fps() * fps)
    frame_indices = np.linspace(0, total_frames - 1, num_frames).astype(int)
    frame_indices = frame_indices.tolist()
    frames = vr.get_batch(frame_indices).asnumpy() #[N,H,W,C]
    return frames, frame_indices

def get_video_frames_by_indices(video_path : str, frame_indices):
    # vr = decord.VideoReader(video_path, ctx=decord.cpu(0), num_threads=1)
    vr = decord.VideoReader(video_path)
    # print(video_path, len(vr), vr.get_avg_fps())
    return vr.get_batch(frame_indices).asnumpy(), vr.get_avg_fps(), len(vr)/vr.get_avg_fps()

def get_video_frames_for_default_llava(video_path, max_frames_num=64,fps=1,force_sample=False):
    if max_frames_num == 0:
        return np.zeros((1, 336, 336, 3))
    # vr = decord.VideoReader(video_path, ctx=decord.cpu(0), num_threads=1)
    vr = decord.VideoReader(video_path)
    total_frame_num = len(vr)
    video_time = total_frame_num / vr.get_avg_fps()
    fps = round(vr.get_avg_fps()/fps)
    frame_idx = [i for i in range(0, len(vr), fps)]
    frame_time = [i/fps for i in frame_idx]
    if len(frame_idx) > max_frames_num or force_sample:
        sample_fps = max_frames_num
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        frame_time = [i/vr.get_avg_fps() for i in frame_idx]
    frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    time_instruciton = f"The video lasts for {video_time:.2f} seconds, and {len(spare_frames)} frames are uniformly sampled from it. These frames are located at {frame_time}.Please answer the following questions related to this video."
    return spare_frames, time_instruciton

def get_video_frames_for_default_qwen2_5vl(video_path : str):
    # vr = decord.VideoReader(video_path, ctx=decord.cpu(0), num_threads=1)
    vr = decord.VideoReader(video_path)
    total_frames = len(vr)
    from qwen_vl_utils.vision_process import smart_nframes
    num_frames = smart_nframes({}, total_frames, vr.get_avg_fps())
    # print(nframes)
    frame_indices = np.linspace(0, total_frames - 1, num_frames).astype(int)
    frames = vr.get_batch(frame_indices).asnumpy()
    return frames


def get_video_frames_for_default_qwen3vl(video_path : str):
    # vr = decord.VideoReader(video_path, ctx=decord.cpu(0), num_threads=1)
    vr = decord.VideoReader(video_path)
    total_frames = len(vr)
    from qwen_vl_utils.vision_process import smart_nframes
    num_frames = smart_nframes({}, total_frames, vr.get_avg_fps())
    # print(nframes)
    frame_indices = np.linspace(0, total_frames - 1, num_frames).astype(int)
    frames = vr.get_batch(frame_indices).asnumpy()

    video_metadata = dict(
        fps=vr.get_avg_fps(),
        frames_indices=frame_indices.tolist(),
        total_num_frames=total_frames,
        video_backend="decord",
    )

    return frames, video_metadata



def get_video_frames_for_default_adaretake(video_path : str, max_num_frames = 2048, sample_fps = 2):
    # vr = decord.VideoReader(video_path, ctx=decord.cpu(0), num_threads=1)
    vr = decord.VideoReader(video_path)
    total_frames = len(vr)
    num_frames = min(round(total_frames / vr.get_avg_fps() * sample_fps), max_num_frames)
    frame_indices = np.linspace(0, total_frames - 1, num_frames).astype(int)
    frames = vr.get_batch(frame_indices).asnumpy()
    return frames