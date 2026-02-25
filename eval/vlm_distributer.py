import os
import json
import argparse
from datetime import datetime
from tqdm import tqdm
from queue import Queue
import yaml
from utils.log_utils import log
import copy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./logs")
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

    if 'vtr_config' in config:
        vtr_config_path = os.path.join("./configs/vtr_models", config['vtr_config']+".yaml")
        with open(vtr_config_path, "r", encoding="utf-8") as yaml_file:
            c = yaml.safe_load(yaml_file)
            c.update(config['vtr'] if 'vtr' in config else {})
            config['vtr'] = c
        config.pop('vtr_config')

    vlm_config_path = os.path.join("./configs/vlm_models", config['vlm_config']+".yaml")
    with open(vlm_config_path, "r", encoding="utf-8") as yaml_file:
        c = yaml.safe_load(yaml_file)
        c.update(config['vlm'] if 'vlm' in config else {})
        config['vlm'] = c
        if 'extra_model_config' not in config['vlm']:
            config['vlm']['extra_model_config'] = {}
        extra_model_config = copy.deepcopy(config['vlm'])
        extra_model_config.pop('extra_model_config')
        extra_model_config.pop('name')
        extra_model_config.pop('model_path')
        extra_model_config.pop('devices_list')
        extra_model_config.update(config['vlm']['extra_model_config'])
        config['vlm']['extra_model_config'] = extra_model_config
    config.pop('vlm_config')

    if args.resume is not None:
        save_res_dir = args.resume
        with open(os.path.join(save_res_dir, "config.json"), "r") as f:
            config = json.load(f)
    else:
        timestamp_str =  datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        config['save_res_dir'] = os.path.abspath(os.path.join(args.output_dir, timestamp_str))
        save_res_dir = config['save_res_dir']
        os.makedirs(save_res_dir, exist_ok=True)
        with open(os.path.join(save_res_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=4, ensure_ascii=False)

    save_res_path = os.path.join(save_res_dir, 'results.json')

    last_infer_results = {}
    if os.path.exists(save_res_path):
        with open(save_res_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                item = json.loads(line)
                last_infer_results[item['QA']['id']]=item
                
    from benchmarks import get_benchmark
    benchmark = get_benchmark(config['benchmark'])

    if config['enable_vtr']:
        sim = {}
        with open(os.path.join(config['pre_calc_emb_path'], "results.json"), "r") as f:
            for line in f.readlines():
                data = json.loads(line)
                embd_save_path = os.path.join(config['pre_calc_emb_path'], "embeds", os.path.basename(data['embd_save_path']))
                sim[data['video_path']] = (data['frame_indices'], embd_save_path)


    if 'qof' in config and config['qof']['mode'] == 'pre_gen_opts':
        pre_gen_opts = {}
        with open(config['qof']['pre_gen_opts_path'], "r") as f:
            for line in f.readlines():
                data = json.loads(line)
                pre_gen_opts[data['QA']['id']] = data['QA']['gen_opts']

    annos = []
    for row in benchmark.rows:
        row_id = row['id']
        if row_id in last_infer_results:
            continue
        video_path = row['video_path']
        
        if config['enable_vtr']:
            frame_indices, embd_save_path = sim[video_path]
        else:
            frame_indices, embd_save_path = None, None
        annos.append({
            'video_path' : video_path,
            'frame_indices' : frame_indices,
            'embd_save_path' : embd_save_path,
            'QA' : {
                "id" : row_id,
                "ques" : row['ques'],
                "opts" : row['opts'],
                "gt_ans" : row['gt_ans'],
                "full_prompt" : row['full_prompt'],
                # "global_summary" : row['global_summary'],
            }
        })
        if 'qof' in config and config['qof']['mode'] == 'pre_gen_opts':
            annos[-1]['QA']['gen_opts'] = pre_gen_opts[row_id]

    data_queue = Queue()
    for anno in annos:
        data_queue.put(anno)

    pbar = tqdm(range(len(annos)), desc=f"SENT 0/{len(annos)}")

    recved_num = 0

    def handel_request(request):
        global recved_num
        if request['cmd']=='get':
            if data_queue.empty():
                response = {
                    "status" : "no more data",
                }
                if recved_num >= len(annos):
                    response = None
            else:
                r = int(str(pbar.desc).split(" ")[1].split("/")[0]) + 1
                pbar.set_description_str(f"SENT {r}/{len(annos)}")
                response = {
                    "status" : "success",
                    "config" : config,
                    "data" : data_queue.get()
                }
        elif request['cmd']=='put':
            if request['data']['config']['save_res_dir']!=config['save_res_dir']:
                log("No this section, discard it.")
            else:
                with open(save_res_path, "a") as f:
                    f.write(json.dumps(request['data'], ensure_ascii=False)+"\n")
                metrics = benchmark.metrics(save_res_path)
                with open(os.path.join(config['save_res_dir'], "metrics.json"), "w") as f:
                    json.dump(metrics, f, indent=4, ensure_ascii=False)
                pbar.update(1)
                recved_num+=1
            response = { "status" : "success",}
        return response

    from utils.network_utils import start_server, send_request

    start_server(handel_request, "::", multinode_config['port'])
