
from tqdm import tqdm
import os
from utils.log_utils import log
import warnings
import json
import torch
warnings.filterwarnings("ignore")
        
if __name__ == "__main__":
    video_embeddings_save_dir = "./pre_calc_emb/videoevalpro"
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

        
        log("Load Video Embeddings from Cache")
        embd_path = os.path.join(video_embeddings_save_dir, "embeds", os.path.basename(video_path)[:-4] + ".npy")
        video_embeddings = torch.load(embd_path, weights_only=True, map_location="cpu")
        with open(os.path.join(video_embeddings_save_dir, "results.json"), "r") as f:
            for line in f.readlines():
                item = json.loads(line)
                if os.path.basename(video_path) == os.path.basename(item['video_path']):
                    frame_indices = item['frame_indices']
                    break
        
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
