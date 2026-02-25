import os
import json
import argparse
from datetime import datetime
from tqdm import tqdm
from queue import Queue
import yaml
from utils.log_utils import log

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/default_emb.yaml")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./logs/emb")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as yaml_file:
        config = yaml.safe_load(yaml_file)

    with open("./configs/multinode/default.yaml", "r", encoding="utf-8") as yaml_file:
        multinode_config = yaml.safe_load(yaml_file)
    
    benchmark_config_path = os.path.join("./configs/benchmarks", config['benchmark_config']+".yaml")
    with open(benchmark_config_path, "r", encoding="utf-8") as yaml_file:
        c = yaml.safe_load(yaml_file)
        c.update(config['benchmark'] if 'benchmark' in config else {})
        config['benchmark'] = c
    config.pop('benchmark_config')

    vtr_config_path = os.path.join("./configs/vtr_models", config['vtr_config']+".yaml")
    with open(vtr_config_path, "r", encoding="utf-8") as yaml_file:
        c = yaml.safe_load(yaml_file)
        c.update(config['vtr'] if 'vtr' in config else {})
        config['vtr'] = c
    config.pop('vtr_config')

    if args.resume is not None:
        save_res_dir = args.resume
        with open(os.path.join(save_res_dir, "config.json"), "r") as f:
            config = json.load(f)
    else:
        timestamp_str =  datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        config['save_res_dir'] = os.path.abspath(os.path.join(args.output_dir, timestamp_str))
        save_res_dir = config['save_res_dir']
        os.makedirs(save_res_dir, exist_ok=True)
        os.makedirs(os.path.join(save_res_dir,"embeds"), exist_ok=True)
        with open(os.path.join(save_res_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=4, ensure_ascii=False)

    save_res_path = os.path.join(save_res_dir, 'results.json')

    last_infer_results = {}
    if os.path.exists(save_res_path):
        with open(save_res_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                item = json.loads(line)
                for QA in item['QAs']:
                    last_infer_results[QA['id']]=item
                
    from benchmarks import get_benchmark
    benchmark = get_benchmark(config['benchmark'])

    annos = {}
    num_qas = 0
    for row in benchmark.rows:
        row_id = row['id']
        if row_id in last_infer_results:
            continue
        video_path = row['video_path']
        if video_path not in annos:
            annos[video_path] = {
                'video_path' : video_path,
                'QAs' : [],
            }
        annos[video_path]['QAs'].append({
            "id" : row_id,
            "ques" : row['ques'],
            "opts" : row['opts'],
            "gt_ans" : row['gt_ans'],
            "full_prompt" : row['full_prompt'],
        })
        num_qas += 1

    annos = list(annos.values())
    num_video =  len(annos)
    log("Video Num :", num_video, " ,QA Num :", num_qas)

    data_queue = Queue()
    for anno in annos:
        data_queue.put(anno)

    pbar = tqdm(range(num_video))

    recved_num = 0

    def handel_request(request):
        global recved_num
        if request['cmd']=='get':
            if data_queue.empty():
                response = {
                    "status" : "no more data",
                }
                if recved_num >= num_video:
                    response = None
            else:
                response = {
                    "status" : "success",
                    "config" : config,
                    "data" : data_queue.get()
                }
        elif request['cmd']=='put':
            if request['data']['config']['save_res_dir']!=save_res_dir:
                log("No this section, discard it.")
            else:
                request['data'].pop('config')
                with open(save_res_path, "a") as f:
                    f.write(json.dumps(request['data'], ensure_ascii=False)+"\n")
                pbar.update(1)
                recved_num+=1
            response = { "status" : "success",}
        return response

    from utils.network_utils import start_server, send_request

    start_server(handel_request, "::", multinode_config['port'])