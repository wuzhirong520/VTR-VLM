import multiprocessing as mp
# import torch.multiprocessing as mp
if mp.get_start_method() != "spawn":
    mp.set_start_method("spawn", force=True)   
import argparse
from tqdm import tqdm
import time
import os
import sys
import signal
import yaml
from utils.log_utils import log

def video_worker(vtr_name, vtr_path, recv_queue, video_queue):
    print(f"video_worker and {vtr_name}")
    from models import get_vtr_results
    from utils.video_utils import get_video_frames_by_indices
    from models.vlm import resize_video_for_vlm
    import numpy as np
    import copy
    import torch
    import time
    import json
    from PIL import Image

    if vtr_name != 'none':
        from models import get_vtr_model
        vtr_model = get_vtr_model(vtr_name, vtr_path, "cpu")
        # vtr_model = get_vtr_model(vtr_name, vtr_path, "cuda")

    while True:
        data = recv_queue.get()
        if data is None:
            break

        debug_time_1 = time.time()

        if data['config']['enable_vtr']:

            if data['config']['qof']['mode'] == 'ques_opts':
                video_embeddings = torch.load(data['embd_save_path'], weights_only=True, map_location="cpu")
                if len(data['QA']['opts']) > 0:
                    vtr_texts = [data['QA']['ques'] + " " + o for o in data['QA']['opts']]
                else:
                    vtr_texts = [data['QA']['ques']]
                similarity = []
                for vtr in vtr_texts:
                    text_embedding = vtr_model.get_text_embedding(vtr).detach().cpu()
                    sim_matrix = text_embedding @ video_embeddings.transpose(0,1)
                    similarity.append(sim_matrix[0])
                if 'scale' in data['config']['qof']:
                    text_embedding = vtr_model.get_text_embedding(data['QA']['ques']).detach().cpu()
                    sim0 = (text_embedding @ video_embeddings.transpose(0,1))[0]
                    partition = np.sort(np.argsort(sim0)[-int(data['config']['vtr']['top_k']*data['config']['qof']['scale']):])
                else:
                    partition = None
            elif data['config']['qof']['mode'] == 'ques_opts_global':
                video_embeddings = torch.load(data['embd_save_path'], weights_only=True, map_location="cpu")
                if len(data['QA']['opts']) > 0:
                    vtr_texts = [data['QA']['ques'] + " " + o for o in data['QA']['opts']]
                else:
                    vtr_texts = [data['QA']['ques']]

                sss = 0.5
                global_summary_embedding = vtr_model.get_text_embedding(data['QA']['global_summary']).detach().cpu()

                similarity = []
                for vtr in vtr_texts:
                    text_embedding = vtr_model.get_text_embedding(vtr).detach().cpu()
                    text_embedding = (1-sss) * text_embedding + sss * global_summary_embedding
                    sim_matrix = text_embedding @ video_embeddings.transpose(0,1)
                    similarity.append(sim_matrix[0])


                if 'scale' in data['config']['qof']:
                    text_embedding = vtr_model.get_text_embedding(data['QA']['ques']).detach().cpu()
                    sim0 = (text_embedding @ video_embeddings.transpose(0,1))[0]
                    partition = np.sort(np.argsort(sim0)[-int(data['config']['vtr']['top_k']*data['config']['qof']['scale']):])
                else:
                    partition = None
            elif data['config']['qof']['mode'] == 'opts_only':
                video_embeddings = torch.load(data['embd_save_path'], weights_only=True, map_location="cpu")
                if len(data['QA']['opts']) > 0:
                    vtr_texts = [o for o in data['QA']['opts']]
                else:
                    vtr_texts = [data['QA']['ques']]
                similarity = []
                for vtr in vtr_texts:
                    text_embedding = vtr_model.get_text_embedding(vtr).detach().cpu()
                    sim_matrix = text_embedding @ video_embeddings.transpose(0,1)
                    similarity.append(sim_matrix[0])
                if 'scale' in data['config']['qof']:
                    text_embedding = vtr_model.get_text_embedding(data['QA']['ques']).detach().cpu()
                    sim0 = (text_embedding @ video_embeddings.transpose(0,1))[0]
                    partition = np.sort(np.argsort(sim0)[-int(data['config']['vtr']['top_k']*data['config']['qof']['scale']):])
                else:
                    partition = None
            elif data['config']['qof']['mode'] == 'pre_gen_opts':
                video_embeddings = torch.load(data['embd_save_path'], weights_only=True, map_location="cpu")
                if len(data['QA']['gen_opts']) > 0:
                    vtr_texts = [data['QA']['ques'] + " " + o for o in data['QA']['gen_opts']]
                else:
                    vtr_texts = [data['QA']['ques']]
                similarity = []
                for vtr in vtr_texts:
                    text_embedding = vtr_model.get_text_embedding(vtr).detach().cpu()
                    sim_matrix = text_embedding @ video_embeddings.transpose(0,1)
                    similarity.append(sim_matrix[0])
                if 'scale' in data['config']['qof']:
                    text_embedding = vtr_model.get_text_embedding(data['QA']['ques']).detach().cpu()
                    sim0 = (text_embedding @ video_embeddings.transpose(0,1))[0]
                    partition = np.sort(np.argsort(sim0)[-int(data['config']['vtr']['top_k']*data['config']['qof']['scale']):])
                else:
                    partition = None
            elif data['config']['qof']['mode'] == 'ques_only':
                video_embeddings = torch.load(data['embd_save_path'], weights_only=True, map_location="cpu")
                similarity = []
                vtr_text = [data['QA']['ques']]
                for vtr in vtr_text:
                    text_embedding = vtr_model.get_text_embedding(vtr).detach().cpu()
                    sim_matrix = text_embedding @ video_embeddings.transpose(0,1)
                    similarity.append(sim_matrix[0])
                partition = None
            elif data['config']['qof']['mode'] == 'gt':
                video_embeddings = torch.load(data['embd_save_path'], weights_only=True, map_location="cpu")
                similarity = []
                vtr = data['QA']['ques'] + " " + data['QA']['opts'][ord(data['QA']['gt_ans'])-ord('A')]
                text_embedding = vtr_model.get_text_embedding(vtr).detach().cpu()
                sim_matrix = text_embedding @ video_embeddings.transpose(0,1)
                similarity.append(sim_matrix[0])
                partition = None
            elif data['config']['qof']['mode'] == 'generate_opts':
                genearte_num = data['config']['qof']['generate_num']

                if 'generate_step' not in data:
                    video_embeddings = torch.load(data['embd_save_path'], weights_only=True, map_location="cpu")
                    similarity = []
                    for vtr in [data['QA']['ques']]:
                        text_embedding = vtr_model.get_text_embedding(vtr).detach().cpu()
                        sim_matrix = text_embedding @ video_embeddings.transpose(0,1)
                        similarity.append(sim_matrix[0])
                    data['generate_step'] = 0
                    data['options_generated'] = []
                    data['similarity'] = similarity

                    if len(data['similarity'][0]) < genearte_num * data['config']['vtr']['top_k']:
                        data['generate_step'] = genearte_num - int(len(data['similarity'][0]) / data['config']['vtr']['top_k'])

                # if len(data['similarity'][0]) < genearte_num * data['config']['vtr']['top_k']:
                #     data['generate_step'] = genearte_num
                
                if data['generate_step'] < genearte_num:
                    print(data['generate_step'], data['options_generated'])
                    similarity = data['similarity']

                    # partition = np.argsort(similarity[0])[-genearte_num*data['config']['vtr']['top_k']:]
                    # partition = partition[data['generate_step']::genearte_num]
                    # partition = np.argsort(partition)

                    partition = np.sort(np.argsort(similarity[0])[-genearte_num*data['config']['vtr']['top_k']:])
                    partition = partition[data['generate_step']::genearte_num]

                    # partition = partition[data['generate_step']*data['config']['vtr']['top_k'] : (data['generate_step']+1) *data['config']['vtr']['top_k']]
                else:
                    print("Final", data['generate_step'], data['options_generated'])
                    video_embeddings = torch.load(data['embd_save_path'], weights_only=True, map_location="cpu")
                    data['options_generated'] = list(set(data['options_generated']))
                    data['QA']['gen_opts'] = data['options_generated']
                    if len(data['options_generated']) > 0:
                        vtr_texts = [data['QA']['ques'] + " " + o for o in data['options_generated']]
                        similarity = []
                        for vtr in vtr_texts:
                            text_embedding = vtr_model.get_text_embedding(vtr).detach().cpu()
                            sim_matrix = text_embedding @ video_embeddings.transpose(0,1)
                            similarity.append(sim_matrix[0])
                        partition = np.sort(np.argsort(data['similarity'][0])[-int(data['config']['vtr']['top_k']*data['config']['qof']['scale']):])
                    else:
                        similarity = data['similarity']
                        partition = None

            # elif data['config']['qof']['mode'] == 'generate_opts':
            #     if 'generate_step' not in data:
            #         data['generate_step'] = 0
            #         data['options_generated'] = []
            #         data['partition'] = None
            #     data['options_generated'] = list(set(data['options_generated']))
            #     data['QA']['gen_opts'] = data['options_generated']
            #     print(data['generate_step'], data['options_generated'])
            #     video_embeddings = torch.load(data['embd_save_path'], weights_only=True, map_location="cpu")
            #     if len(data['options_generated']) > 0:
            #         vtr_texts = [data['QA']['ques'] + " " + o for o in data['options_generated']]
            #     else:
            #         vtr_texts = [data['QA']['ques']]
            #     similarity = []
            #     for vtr in vtr_texts:
            #         text_embedding = vtr_model.get_text_embedding(vtr).detach().cpu()
            #         sim_matrix = text_embedding @ video_embeddings.transpose(0,1)
            #         similarity.append(sim_matrix[0])
            #     partition = data['partition']
            #     data['partition'] = np.sort(np.argsort(similarity[0])[-int(data['config']['vtr']['top_k']*data['config']['qof']['scale']):])

            vtr_results = get_vtr_results(data['frame_indices'], similarity, data['config']['vtr'], partition)
            frames_info = {}
            frames_info['frames'], frames_info['fps'], frames_info['duration'] = get_video_frames_by_indices(data['video_path'], vtr_results['frames_idx'])
            frames_info["frames_idx"] = vtr_results['frames_idx']
            frames_info["frames_sim"] = vtr_results['frames_sim']
            frames_info['expect_frames'] = data['config']['vtr']['top_k'] * data['config']['vtr']['top_k_per']

            if data['config']['enable_dynamic_resolution']:
                dynamic_resolution_config = data['config']['dynamic_resolution']
            else:
                dynamic_resolution_config = None

            frames = resize_video_for_vlm(data['config']['vlm']['name'], frames_info, dynamic_resolution_config)

        else:
            frames_info = {}
            if data['config']['vlm']['name'] == "llava":
                from utils.video_utils import get_video_frames_for_default_llava
                frames_info['frames'], frames_info['time_instruction'] = get_video_frames_for_default_llava(data['video_path'])
            elif data['config']['vlm']['name'] == "qwen2.5vl":
                from utils.video_utils import get_video_frames_for_default_qwen2_5vl
                frames_info['frames'] = get_video_frames_for_default_qwen2_5vl(data['video_path'])  
            elif data['config']['vlm']['name'] == "qwen3vl":
                from utils.video_utils import get_video_frames_for_default_qwen3vl
                frames_info['frames'], frames_info['video_metadata'] = get_video_frames_for_default_qwen3vl(data['video_path'])  
            elif data['config']['vlm']['name'] == "adaretake":
                from utils.video_utils import get_video_frames_for_default_adaretake
                frames_info['frames'] = get_video_frames_for_default_adaretake(data['video_path'])  
            elif data['config']['vlm']['name'] == "adaretake_llava":
                from utils.video_utils import get_video_frames_for_default_adaretake
                frames_info['frames'] = get_video_frames_for_default_adaretake(data['video_path'], 1024)  
            elif data['config']['vlm']['name'] == "apicall":
                from utils.video_utils import get_video_frames_for_default_adaretake
                frames_info['frames'] = get_video_frames_for_default_adaretake(data['video_path'], 640)  
            frames = resize_video_for_vlm(data['config']['vlm']['name'], frames_info, None)


        if data['config']['enable_vtr'] and 'save_similarity_infos' in data['config'] and data['config']['save_similarity_infos']:
            if 'generate_step' not in data or data['generate_step'] == genearte_num:
                similarity_infos = {
                    "similarity": np.array(similarity).tolist(),
                    "frame_indices": np.array(data['frame_indices']).tolist(),
                    "selected_frames_idx" : np.array(frames_info["frames_idx"]).tolist(),
                    "selected_frames_sim" : np.array(frames_info["frames_sim"]).tolist(),
                    "selected_frames_res" : [
                        [ f['shape'], np.array(f['global_idx']).tolist() ] for f in frames
                    ]
                }
                os.makedirs(os.path.join(data['config']['save_res_dir'], f"similarity_infos"), exist_ok=True)
                with open(os.path.join(data['config']['save_res_dir'], f"similarity_infos/{data['QA']['id']:05d}.json"), "w") as dump_file:
                    json.dump(similarity_infos, dump_file, ensure_ascii=False)


        if data['config']['dump_vtr_videos']:
            if 'generate_step' not in data or data['generate_step'] == genearte_num:
                import imageio.v2 as imageio
                os.makedirs(os.path.join(data['config']['save_res_dir'], f"dumps/{data['QA']['id']:05d}/images"), exist_ok=True)
                with open(os.path.join(data['config']['save_res_dir'], f"dumps/{data['QA']['id']:05d}/info.json"), "w") as dump_file:
                    json.dump(data['QA'], dump_file, indent=4, ensure_ascii=False)
                debug_video_save_path = os.path.join(data['config']['save_res_dir'], f"dumps/{data['QA']['id']:05d}/video.mp4")
                if isinstance(frames, dict):
                    dump_frames = frames['frames']
                else:
                    if len(frames) == 1:
                        dump_frames = frames[0]['frames'].permute(0,2,3,1).numpy()
                    else:
                        from torchvision.transforms import InterpolationMode
                        from torchvision.transforms.functional import resize
                        dump_frames = []
                        resized_height, resized_width = frames[0]['shape']
                        for fi in range(len(frames)):
                            if fi==0:
                                fff = frames[0]['frames'].permute(0,2,3,1).numpy()
                            else:
                                fff = resize(
                                    frames[fi]['frames'],
                                    [resized_height, resized_width],
                                    interpolation=InterpolationMode.BICUBIC,
                                    antialias=True,
                                ).permute(0,2,3,1).numpy()
                            for fii in range(len(fff)):
                                dump_frames.append((fff[fii], frames[fi]['global_idx'][fii]))
                        dump_frames = sorted(dump_frames, key=lambda x: x[1])
                        dump_frames = [f[0] for f in dump_frames]
                    dump_frames = np.clip(dump_frames,0,255).astype(np.uint8)
                ffmpeg_params = [
                    # "-vf", "scale='min(720,iw)':'min(720,ih)':force_original_aspect_ratio=decrease,pad=ceil(iw/2)*2:ceil(ih/2)*2",
                    "-c:v", "libx264",
                    "-preset", "slow",
                    "-crf", "30",
                    "-pix_fmt", "yuv420p"
                ]
                with imageio.get_writer(debug_video_save_path, format='FFMPEG', mode='I', fps=2, ffmpeg_params = ffmpeg_params) as writer:
                    for frame_idx, frame in enumerate(dump_frames):
                        Image.fromarray(frame).save(os.path.join(data['config']['save_res_dir'], f"dumps/{data['QA']['id']:05d}/images/{frame_idx:05d}.jpg"))
                        writer.append_data(frame)

        debug_time_2 = time.time()
        
        if data['config']['enable_vtr'] and data['config']['qof']['mode'] == 'generate_opts' and data['generate_step'] < data['config']['qof']['generate_num']:   
            data["preprocess_time"] = data["preprocess_time"] + [debug_time_2 - debug_time_1]
            data["frames_info"] = frames
            video_queue.put(data)
        else:
            new_data = {
                "config" : data['config'],
                "video_path" : data['video_path'],
                "QA" : data['QA'],
                "frames_info" : frames,
                "preprocess_time" : data["preprocess_time"] + [debug_time_2 - debug_time_1],
                "inference_time" : data["inference_time"]
            }
            video_queue.put(new_data)

def vlm_worker(model_name, model_path, extra_model_config, devices,recv_queue, video_queue, vlm_queue):
    # import debugpy
    # debugpy.listen(("::", 34579), in_process_debug_adapter=True)
    # print("🟡 Waiting for VSCode debugger to attach... ")
    # debugpy.wait_for_client()
    # print("🟢 Debugger attached!")

    print(f"vlm_worker {model_name},{devices}")
    if isinstance(devices, int):
        devices = [devices]
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(d) for d in devices])
    from models import get_vlm_model
    vlm_model = get_vlm_model(model_name, model_path, extra_model_config)

    while True:
        data = video_queue.get()
        if data is None:
            break

        if data['config']['enable_vtr'] and data['config']['qof']['mode'] == 'generate_opts' and 'generate_step' in data and data['generate_step'] < data['config']['qof']['generate_num']:   
            print("VLM ", data['generate_step'], data['options_generated'])
            debug_time_1 = time.time()
            option_str = vlm_model.infer(data['config']['qof']['option_generate_prompt'] + data['QA']['ques'], data['frames_info'])
            print(option_str)
            debug_time_2 = time.time()
            options = []
            for opt in option_str.split("\n"):
                if len(opt) > 2 and (opt[1] == "." or opt[1] == ")" or opt[1] == ":") and opt[0] in  ('A','B','C','D','E'):
                    options.append(opt[2:].strip())
            data['options_generated'] = data['options_generated'] + options
            data['generate_step'] = data['generate_step'] + 1
            data["inference_time"] = data["inference_time"] + [debug_time_2 - debug_time_1]
            data.pop('frames_info')
            recv_queue.put(data)
        else:
            debug_time_1 = time.time()
            ans = vlm_model.infer(data['QA']['full_prompt'], data['frames_info'])
            debug_time_2 = time.time()
            new_data = {
                "config" : data['config'],
                "video_path" : data['video_path'],
                "QA" : data['QA'],
                "ans" : ans,
                "preprocess_time" : data["preprocess_time"],
                "inference_time" : data["inference_time"] + [debug_time_2 - debug_time_1]
            }
            vlm_queue.put(new_data)

def send_worker(host, port, vlm_queue):
    from utils.network_utils import send_request
    while True:
        data = vlm_queue.get()
        if data is None:
            break
        request = {
            "cmd" : "put",
            "data" : data
        }
        response = send_request(host, port, request)
        if response is None or response['status']!='success':
            log("Failed to Send Result...")
            time.sleep(5)
            continue

def terminate_processes_and_exit(signal_number, frame):
    print("\nCleaning Subprocesses")
    for p in mp.active_children():
        p.terminate()
        p.join()
    sys.exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, terminate_processes_and_exit)
    signal.signal(signal.SIGTERM, terminate_processes_and_exit)

    parser = argparse.ArgumentParser()
    # parser.add_argument("--host", type=str, default="fdbd:dc02:21:13c::22")
    # parser.add_argument("--port", type=int, default=63153)
    parser.add_argument("--video_processor_num", type=int, default=16)
    args = parser.parse_args()

    with open("./configs/multinode/default.yaml", "r", encoding="utf-8") as yaml_file:
        multinode_config = yaml.safe_load(yaml_file)

    import torch
    total_num_gpus = torch.cuda.device_count()
    gpu_num_scale = total_num_gpus / 8
    args.video_processor_num = max(1, int(args.video_processor_num * gpu_num_scale))

    recv_queue = mp.Queue(maxsize=int(16*gpu_num_scale))
    video_queue = mp.Queue(maxsize=int(64*gpu_num_scale))
    vlm_queue = mp.Queue()

    video_worker_processes = []
    send_worker_process = mp.Process(target=send_worker, args=(multinode_config['host'], multinode_config['port'], vlm_queue))
    send_worker_process.start()

    vlm_worker_processes = []

    cur_config = None

    

    while True:
        if video_queue.qsize() > int(16*gpu_num_scale) and recv_queue.qsize() > int(8*gpu_num_scale):
            time.sleep(0.1)
            continue
        try:
            from utils.network_utils import send_request
            response = send_request(multinode_config['host'], multinode_config['port'], {"cmd" : "get"})
            if response is None or response['status']!='success':
                # log("Retry Network...")
                time.sleep(5)
                continue
            # log(response)

            config = response['config']

            if cur_config is None or  (
                cur_config['vtr']['name'] != config['vtr']['name'] or
                cur_config['vtr']['model_path'] != config['vtr']['model_path']
            ):
                log("Change VTR Model, Waiting to Terminate Old Model...")
                for _ in video_worker_processes:
                    recv_queue.put(None)
                for p in video_worker_processes:
                    p.join()
                log("Starting New VTR Model")
                video_worker_processes = []
                for _ in range(args.video_processor_num):
                    p = mp.Process(target=video_worker, args=(
                        config['vtr']['name'],
                        config['vtr']['model_path'],
                        recv_queue,
                        video_queue
                    ))
                    p.start()
                    video_worker_processes.append(p)

            if cur_config is None or (
                cur_config['vlm']['name'] != config['vlm']['name'] or
                cur_config['vlm']['model_path'] != config['vlm']['model_path'] or
                cur_config['vlm']['devices_list']!= config['vlm']['devices_list'] or
                cur_config['vlm']['extra_model_config']!= config['vlm']['extra_model_config']
            ):
                log("Change VLM Model, Waiting to Terminate Old Model...")
                for _ in vlm_worker_processes:
                    video_queue.put(None)
                for p in vlm_worker_processes:
                    p.join()
                log("Starting New VLM Model")
                vlm_worker_processes = []
                for deivces in config['vlm']['devices_list']:
                    if (isinstance(deivces, int) and deivces<total_num_gpus) or (isinstance(deivces, list) and max(deivces)<total_num_gpus):
                        p = mp.Process(target=vlm_worker, args=(
                            config['vlm']['name'],
                            config['vlm']['model_path'],
                            config['vlm']['extra_model_config'],
                            deivces,
                            recv_queue,
                            video_queue,
                            vlm_queue
                        ))
                        p.start()
                        vlm_worker_processes.append(p)
            
            cur_config = config


            item = {
                "config" : response['config'],
                "video_path" : response['data']['video_path'],
                "frame_indices" : response['data']['frame_indices'],
                "embd_save_path" : response['data']['embd_save_path'],
                "QA" : response['data']['QA'],
                "preprocess_time" : [],
                "inference_time" : []
            }
            recv_queue.put(item)

        except Exception as e:
            log("!!!!!!",e)
            time.sleep(5)
        