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
warnings.filterwarnings("ignore")

def vtr_worker(vtr_name, vtr_path, device, recv_queue, vtr_queue):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
    from models import get_vtr_model
    from utils.video_utils import get_video_frames_by_fps
    import torch
    vtr_model = get_vtr_model(vtr_name, vtr_path, "cuda")
    import time
    
    while True:
        data = recv_queue.get()
        if data is None:
            break
        debug_time_1 = time.time()
        frames, frame_indices = get_video_frames_by_fps(data['video_path'], data['config']['vtr']['fps'])
        debug_time_2 = time.time()
        video_embeds = vtr_model.get_all_video_embeds(frames, data['config']['vtr'])
        debug_time_3 = time.time()
        video_id = os.path.basename(data['video_path']).split('.')[0]
        embd_save_path = os.path.join(data['config']['save_res_dir'], f"embeds/{video_id}.npy")
        print(video_embeds.shape)
        torch.save(video_embeds, embd_save_path)
        data['embd_save_path'] = embd_save_path
        data['frame_indices'] = frame_indices
        data['time_info'] = {
            "read_video" : debug_time_2 - debug_time_1,
            "inference" : debug_time_3 - debug_time_2
        }
        vtr_queue.put(data)

def send_worker(host, port, vtr_queue):
    from utils.network_utils import send_request
    while True:
        data = vtr_queue.get()
        if data is None:
            break
        # print(data)
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
    # parser.add_argument("--port", type=int, default=51251)
    args = parser.parse_args()

    with open("./configs/multinode/default.yaml", "r", encoding="utf-8") as yaml_file:
        multinode_config = yaml.safe_load(yaml_file)

    recv_queue = mp.Queue(maxsize=8)
    vtr_queue = mp.Queue()

    send_worker_process = mp.Process(target=send_worker, args=(multinode_config['host'], multinode_config['port'], vtr_queue))
    send_worker_process.start()

    vtr_worker_processes = []

    cur_config = None
   
    while True:
        try:
            from utils.network_utils import send_request
            response = send_request(multinode_config['host'], multinode_config['port'], {"cmd" : "get"})
            if response is None or response['status'] != "success":
                time.sleep(5)
                continue
            # print(response)
            data = response['data']
            config = response['config']
            if cur_config is None or (
                cur_config['vtr']['name'] != config['vtr']['name'] or
                cur_config['vtr']['model_path'] != config['vtr']['model_path']
            ):
                log("Change VTR Model, Waiting to Terminate Old Model...")
                for _ in vtr_worker_processes:
                    recv_queue.put(None)
                for p in vtr_worker_processes:
                    p.join()
                log("Starting New VTR Model")
                vtr_worker_processes = []
                import torch
                for deivce in range(torch.cuda.device_count()):
                    p = mp.Process(target=vtr_worker, args=(
                        config['vtr']['name'],
                        config['vtr']['model_path'],
                        deivce,
                        recv_queue,
                        vtr_queue
                    ))
                    p.start()
                    vtr_worker_processes.append(p)
            
            cur_config = config
            data['config'] = config
            recv_queue.put(data)
            
        except Exception as e:
            log("!!!!!!",e)
            time.sleep(5)