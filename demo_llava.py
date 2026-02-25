import multiprocessing as mp
if mp.get_start_method() != "spawn":
    mp.set_start_method("spawn", force=True)   
import argparse
from tqdm import tqdm
import time
import sys
import signal
import os
import yaml
from utils.log_utils import log
import warnings
import json
warnings.filterwarnings("ignore")

def embed_worker(vtr_name, vtr_path, device, recv_queue, embed_queue):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
    from models import get_vtr_model
    vtr_model = get_vtr_model(vtr_name, vtr_path, "cuda")
    
    while True:
        data = recv_queue.get()
        if data is None:
            break
        video_embedding = vtr_model.get_video_embedding(data['frames']).detach().cpu()
        embed_queue.put({
            "idx" : data['idx'],
            "video_embedding": video_embedding,
        })
        

def terminate_processes_and_exit(signal_number, frame):
    print("\nCleaning Subprocesses")
    for p in mp.active_children():
        p.terminate()
        p.join()
    sys.exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, terminate_processes_and_exit)
    signal.signal(signal.SIGTERM, terminate_processes_and_exit)

    
    vtr_config = {
        "frame_num": 16,
        "fps": 1,
        "name": "pe",
        "model_path": "/mnt/bn/wzr/models/PE-Core-G14-448/PE-Core-G14-448.pt",
        "top_k": 16,
        "top_k_per": 4,
        "top_k_per_list": "[2,4,8]"
    }
    qof_config = {
        "option_generate_prompt" : "Please generate four options (A/B/C/D) for the question : ",
        "generate_num" : 3,
        "scale" : 2,
    }
    model_config = {
        "name": "llava",
        "model_path": "/mnt/bn/wzr/models/LLaVA-Video-7B-Qwen2"
    }
    dynamic_resolution_config = None


    import torch
    recv_queue = mp.Queue()
    embed_queue = mp.Queue()
    embed_worker_processes = []
    for deivce in range(torch.cuda.device_count()):
        p = mp.Process(target=embed_worker, args=(
            vtr_config['name'],
            vtr_config['model_path'],
            deivce,
            recv_queue,
            embed_queue
        ))
        p.start()
        embed_worker_processes.append(p)
    

    log("Start Loading Models")
    from models import get_vtr_model
    vtr_model = get_vtr_model(vtr_config['name'], vtr_config['model_path'], "cpu")
    from models import get_vlm_model
    vlm_model = get_vlm_model(model_name=model_config['name'], model_path=model_config['model_path'])
    log("Finish Loading Models")

    from models import get_vtr_results
    from utils.video_utils import get_video_frames_by_indices
    from models.vlm import resize_video_for_vlm

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

        
   
        log("Start Loading Video using Decord, Video Path: ", video_path)
        from utils.video_utils import get_video_frames_by_fps
        total_frames, frame_indices = get_video_frames_by_fps(video_path, vtr_config['fps'])
        log("Finish Loading Frames: ", len(total_frames), "Duration:", len(total_frames)/vtr_config['fps']/60, "minutes")
        total_frame_num = len(total_frames)
        frame_idx_s = []
        all_video_segments = []
        for j in range(0, total_frame_num, vtr_config['frame_num']):
            if j + vtr_config['frame_num'] > total_frame_num:
                j = total_frame_num - vtr_config['frame_num']
            frame_idx_s.append(list(range(j, j+vtr_config['frame_num'])))
            if j + vtr_config['frame_num'] == total_frame_num:
                break
        for idx, frame_idx in enumerate(frame_idx_s):
            all_video_segments.append({
                "idx" : idx,
                "frames": total_frames[frame_idx],
            })
        log("Start Embedding Video Segments, Total Segments: ", len(all_video_segments))
        pbar = tqdm(range(len(all_video_segments)))
        for segment in all_video_segments:
            recv_queue.put(segment)
        embed_results = []
        for _ in range(len(all_video_segments)):
            embed_results.append(embed_queue.get())
            pbar.update(1)
        pbar.close()
        embed_results = sorted(embed_results, key=lambda x: x['idx'])
        video_embeddings = [x['video_embedding'] for x in embed_results]
        video_embeddings = torch.cat(video_embeddings)
        log("Finish Embedding Video Segments, Video Embedding Shape: ", video_embeddings.shape)

        if len(options) == 0:
            log("Start Generating Options")
            similarity = []
            generate_step = 0
            for vtr in [question]:
                text_embedding = vtr_model.get_text_embedding(vtr).detach().cpu()
                sim_matrix = text_embedding @ video_embeddings.transpose(0,1)
                similarity.append(sim_matrix[0])
            if len(similarity[0]) < qof_config['generate_num'] * vtr_config['top_k']:
                generate_step = qof_config['generate_num'] - int(len(similarity[0]) / vtr_config['top_k'])
            
            gen_options = []
            import numpy as np
            pbar  = tqdm(range(qof_config['generate_num']))
            while generate_step < qof_config['generate_num']:
                partition = np.sort(np.argsort(similarity[0])[-qof_config['generate_num']*vtr_config['top_k']:])
                partition = partition[generate_step::qof_config['generate_num']]
                vtr_results = get_vtr_results(frame_indices, similarity, vtr_config, partition)
                frames_info = {}
                frames_info['frames'], frames_info['fps'], frames_info['duration'] = get_video_frames_by_indices(video_path, vtr_results['frames_idx'])
                frames_info["frames_idx"] = vtr_results['frames_idx']
                frames_info["frames_sim"] = vtr_results['frames_sim']
                frames_info['expect_frames'] = vtr_config['top_k'] * vtr_config['top_k_per']
                frames = resize_video_for_vlm(model_config['name'], frames_info, dynamic_resolution_config)
                option_str = vlm_model.infer(qof_config['option_generate_prompt'] + question, frames)
                for opt in option_str.split("\n"):
                    if len(opt) > 2 and (opt[1] == "." or opt[1] == ")" or opt[1] == ":") and opt[0] in  ('A','B','C','D','E'):
                        gen_options.append(opt[2:].strip())
                generate_step += 1
                pbar.update(1)
            pbar.close()
            gen_options = list(set(gen_options))
            if len(gen_options) > 0:
                vtr_texts = [question + " " + o for o in gen_options]
                similarity = []
                for vtr in vtr_texts:
                    text_embedding = vtr_model.get_text_embedding(vtr).detach().cpu()
                    sim_matrix = text_embedding @ video_embeddings.transpose(0,1)
                    similarity.append(sim_matrix[0])
                partition = np.sort(np.argsort(similarity[0])[-int(vtr_config['top_k']*qof_config['scale']):])
            else:
                similarity = similarity
                partition = None
            log("Finish Generate Options: \n", gen_options)
        else:
            log("Use Provided Options: \n", options)
            vtr_texts = [question + " " + o for o in options]
            similarity = []
            for vtr in vtr_texts:
                text_embedding = vtr_model.get_text_embedding(vtr)
                sim_matrix = text_embedding @ video_embeddings.transpose(0,1)
                similarity.append(sim_matrix[0])
            partition = None

        print("")

        log("Start Sampling Frames")
        vtr_results = get_vtr_results(frame_indices, similarity, vtr_config, partition)
        frames_info = {}
        frames_info['frames'], frames_info['fps'], frames_info['duration'] = get_video_frames_by_indices(video_path, vtr_results['frames_idx'])
        frames_info["frames_idx"] = vtr_results['frames_idx']
        frames_info["frames_sim"] = vtr_results['frames_sim']
        frames_info['expect_frames'] = vtr_config['top_k'] * vtr_config['top_k_per']
        frames = resize_video_for_vlm(model_config['name'], frames_info, dynamic_resolution_config)
        log("Finish Sampling Frames, Frame Idx: ", frames_info['frames_idx'])

        print("\n")

        log("Start Inferencing: \n" + full_prompt+ "\n")
        ans = vlm_model.infer(full_prompt, frames)
        log("Finish Inferencing: \n" + ans)
        print("\n")


    for _ in embed_worker_processes:
        recv_queue.put(None)
    for p in embed_worker_processes:
        p.join()

    