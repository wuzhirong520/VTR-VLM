import numpy as np
import torch
import os
import copy
import json

AVAILABLE_VTR_MODELS = ["siglip", "clip4clip", "clip", "pe", "llava_video_encoder"]

def get_vtr_model(model_name, model_path, device):

    if model_name not in AVAILABLE_VTR_MODELS:
        raise ValueError(f"Unsupported VTR model: {model_name}")

    if model_name == "siglip":
        from .siglip import Siglip_VTR_Model
        return Siglip_VTR_Model(model_path=model_path, device=device)
    elif model_name == "clip4clip":
        from .clip4clip import CLIP4Clip_VTR_Model
        return CLIP4Clip_VTR_Model(model_path=model_path, device=device)
    elif model_name == "clip":
        from .clip import Clip_VTR_Model
        return Clip_VTR_Model(model_path=model_path, device=device)
    elif model_name == "pe":
        from .pe import PE_VTR_Model
        return PE_VTR_Model(model_path=model_path, device=device)
    elif model_name == "llava_video_encoder":
        from .llava_video_encoder import LLavaVideoEncoder
        return LLavaVideoEncoder(model_path=model_path, device=device)

import pulp
solve_frame_cache = {}
def solve_frame_allocation(N=16, P=64, C=[2,4,6,8]):
    key = f"{N}_{P}_{'_'.join([str(c) for c in C])}"
    if key in solve_frame_cache:
        return solve_frame_cache[key]
    K = len(C)
    prob = pulp.LpProblem("Frame_Allocation", pulp.LpMinimize)
    n = [pulp.LpVariable(f"n_{i}", lowBound=0, cat="Integer") for i in range(K)]
    target = P / K
    d = [pulp.LpVariable(f"d_{i}", lowBound=0, cat="Continuous") for i in range(K)]
    prob += pulp.lpSum(n) == N, "Total_Segment_Num"
    prob += pulp.lpSum(n[i] * C[i] for i in range(K)) == P, "Total_Frame_Num"
    for i in range(K):
        prob += d[i] >= n[i] * C[i] - target
        prob += d[i] >= -(n[i] * C[i] - target)
    prob += pulp.lpSum(d)
    prob.solve(pulp.PULP_CBC_CMD(msg=False, options=[f"randomSeed=42"]))
    n_result = [int(pulp.value(n[i])) for i in range(K)]
    solve_frame_cache[key] = n_result
    print(N, P, C, n_result, pulp.LpStatus[prob.status])
    return n_result


def get_vtr_results(frame_indices, similarity, vtr_config, partition=None):
    # print(len(frame_indices), len(similarity), len(similarity[0]), len(sim_matrix))
    total_frame_num = len(frame_indices)
    frame_idx_s = []
    for j in range(0, total_frame_num, vtr_config['frame_num']):
        if j + vtr_config['frame_num'] > total_frame_num:
            j = total_frame_num - vtr_config['frame_num']
        frame_idx_s.append(list(range(j, j+vtr_config['frame_num'])))
        if j + vtr_config['frame_num'] == total_frame_num:
            break
    
    similarity = np.array([s.float() for s in similarity])
    frame_idx_s = np.array(frame_idx_s)

    if partition is not None:
        frame_idx_s = frame_idx_s[partition]
        similarity = similarity[:, partition]

    score = np.array([float(np.max(similarity[:, i])) for i in range(len(similarity[0]))])

    sim = [[i, score[i]] for i in range(len(score))]
    sim = sorted(sim, key=lambda x: x[1], reverse=True)

    sim = sim[:vtr_config['top_k']]

    # sim = sorted(sim, key = lambda x : x[0])

    top_k_per = vtr_config['top_k_per']
    if len(sim) < vtr_config['top_k']:
        top_k_per = min(round(vtr_config['top_k'] / len(sim) * top_k_per), vtr_config['top_k_per'] * 4)
        if top_k_per%2==1:
            top_k_per-=1

    top_k_per_list = [top_k_per for _ in range(len(sim))]
    if 'top_k_per_list' in vtr_config:

        if len(sim) == vtr_config['top_k']:
            top_k_per_list = []
            top_k_per_list_C = json.loads(vtr_config['top_k_per_list'])
            if isinstance(top_k_per_list_C[0], int):
                frame_alloc = solve_frame_allocation(len(sim), top_k_per * len(sim), top_k_per_list_C)
                for ii in range(len(top_k_per_list_C)):
                    top_k_per_list += [top_k_per_list_C[ii]] * frame_alloc[ii]
                top_k_per_list = sorted(top_k_per_list, reverse=True)
            else:
                for p,q in top_k_per_list_C:
                    top_k_per_list += [p]*q
                top_k_per_list = sorted(top_k_per_list, reverse=True)

    new_frame_sim = []
    frame_indices = np.array(frame_indices)
    for (i, s), top_k_per in zip(sim, top_k_per_list):
        new_idx = frame_indices[frame_idx_s[i]]
        if vtr_config['frame_num'] == 1:
            new_frame_sim.append((int(new_idx[0]), s))
            continue
        start, end = new_idx[0], new_idx[-1]
        ss = np.linspace(start, end, top_k_per).astype(int)
        for j in range(top_k_per):
            new_frame_sim.append((int(ss[j]), s))
    

    if 'global_sample' in vtr_config:
        half_global = vtr_config['global_sample'] // 2
        interval = (frame_indices[-1] - frame_indices[0]) // half_global
        global_sample_frames_ids = np.linspace(frame_indices[0], frame_indices[-1], half_global*2).astype(int)
        
        global_frames_not_in = []
        for ii in range(half_global):
            target_id = (global_sample_frames_ids[ii*2]+global_sample_frames_ids[ii*2+1])/2
            min_dis = -1
            for i in range(len(new_frame_sim)):
                x = abs(new_frame_sim[i][0] - target_id)
                if min_dis == -1 or x < min_dis:
                    min_dis = x
            if min_dis > interval:
                global_frames_not_in.append((global_sample_frames_ids[ii*2], -1.0-min_dis))
                global_frames_not_in.append((global_sample_frames_ids[ii*2+1], -1.0-min_dis))

        new_frame_sim = new_frame_sim[:len(new_frame_sim)-len(global_frames_not_in)] + global_frames_not_in
        print("global_frames_to_add : ", len(global_frames_not_in), "total frames:", len(new_frame_sim))
        
    new_frame_sim = sorted(new_frame_sim, key = lambda x : x[0])
    return {
        "frames_idx" : [v[0] for v in new_frame_sim],
        "frames_sim" : [v[1] for v in new_frame_sim],
    }