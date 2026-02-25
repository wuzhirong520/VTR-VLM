import torch
import numpy as np
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from qwen_vl_utils import smart_resize
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from .model import VLM_Model
import pulp

class Qwen2_5VL_VLM_Model(VLM_Model):
    def __init__(self, model_path, extra_model_config):
        self.vlm_model_path = model_path
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.vlm_model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(self.vlm_model_path)
    
    def infer(self, prompt, frames_info, generate_config = {'do_sample':False, 'temperature':0, 'top_p': 1}):
        if frames_info is None:
            messages = [{
                "role": "user",
                "content": [{"type": "text", "text": prompt},],
            }]
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(
                text=[text],
                images=None,
                videos=None,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.model.device)
            inputs = dict(inputs)
        else:
            messages = [{
                "role": "user",
                "content": [{"type": "video","video": "",},
                            {"type": "text", "text": prompt},],
            }]
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            if len(frames_info) == 1:
                video = frames_info[0]['frames']
                image_inputs, video_inputs = None, [video]
                inputs = self.processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )
                inputs = inputs.to(self.model.device)
                inputs = dict(inputs)
            else:
                multi_resolution_video_embed = {}
                for fs in range(len(frames_info)):
                    mvideo = frames_info[fs]['frames']
                    image_inputs, video_inputs = None, [mvideo]
                    old_inputs = self.processor(
                        text=[text],
                        images=image_inputs,
                        videos=video_inputs,
                        padding=True,
                        return_tensors="pt",
                    )
                    old_inputs = old_inputs.to(self.model.device)
                    with torch.no_grad():
                        old_inputs['pixel_values_videos'] = old_inputs['pixel_values_videos'].type(self.model.visual.dtype)
                        video_features = self.model.visual(old_inputs['pixel_values_videos'], old_inputs['video_grid_thw'])
                    # print(old_inputs['pixel_values_videos'].shape, video_features.shape, old_inputs['video_grid_thw'])
                    # print(old_inputs['input_ids'].shape)
                    grid_t = len(frames_info[fs]['frames']) // 2
                    grid_h = frames_info[fs]['shape'][0] //28
                    grid_w = frames_info[fs]['shape'][1] //28
                    video_features = video_features.reshape(grid_t, -1, video_features.shape[-1])
                    for j in range(grid_t):
                        idx = int(frames_info[fs]['global_idx'][j*2] + frames_info[fs]['global_idx'][j*2+1])//2
                        multi_resolution_video_embed[idx] = {
                            "embed" : video_features[j],
                            "h_index" : torch.arange(grid_h).view(-1, 1).expand(-1, grid_w).reshape(-1),
                            "w_index" : torch.arange(grid_w).view(1, -1).expand(grid_h, -1).reshape(-1),
                            "t_index" : torch.full((grid_h*grid_w,), idx, dtype=torch.int64),
                        }
                multi_resolution_video_embed = [[k,v] for k,v in multi_resolution_video_embed.items()]
                multi_resolution_video_embed = sorted(multi_resolution_video_embed, key = lambda x : x[0])
                multi_resolution_video_embed = [v for _,v in multi_resolution_video_embed]
                pre_calculated_visual_embedding = []
                t_index, h_index, w_index = [], [], []
                for i in range(len(multi_resolution_video_embed)):
                    t_index.append(torch.full_like(multi_resolution_video_embed[i]['t_index'], i))
                    h_index.append(multi_resolution_video_embed[i]['h_index'])
                    w_index.append(multi_resolution_video_embed[i]['w_index'])
                    pre_calculated_visual_embedding.append(multi_resolution_video_embed[i]['embed'])
                t_index = torch.cat(t_index)
                h_index = torch.cat(h_index)
                w_index = torch.cat(w_index)
                pre_calculated_visual_embedding = torch.cat(pre_calculated_visual_embedding)

                inputs = self.processor(text=[text],padding=True,return_tensors="pt")
                video_token_idx = int(torch.nonzero(inputs['input_ids'][0]==151656))
                input_ids = [inputs['input_ids'][0][:video_token_idx], torch.full(pre_calculated_visual_embedding.shape[:1], 151656).to(inputs['input_ids'].dtype).to(inputs['input_ids'].device), inputs['input_ids'][0][video_token_idx+1:]]
                input_ids = torch.cat(input_ids).unsqueeze(0)
                attention_mask = torch.ones_like(input_ids)
                front_text_position_ids = torch.arange(video_token_idx).view(1, -1).expand(3, -1)
                video_position_ids = torch.stack([t_index, h_index, w_index]) + video_token_idx
                back_text_position_ids = torch.arange(len(inputs['input_ids'][0]) - video_token_idx - 1).view(1, -1).expand(3, -1) + video_position_ids.max() + 1
                position_ids = torch.cat([front_text_position_ids, video_position_ids, back_text_position_ids], dim=1).unsqueeze(1)
                mrope_position_deltas = torch.tensor([position_ids.max() + 1 - len(input_ids[0])]).unsqueeze(1)

                inputs = {
                    "input_ids" : input_ids.to(self.model.device),
                    "attention_mask" : attention_mask.to(self.model.device),
                    "pre_calculated_visual_embedding" : pre_calculated_visual_embedding.to(self.model.device),
                    "position_ids" : position_ids.to(self.model.device),
                }
                self.model.model.rope_deltas = mrope_position_deltas.to(self.model.device)
                    

        generated_ids = self.model.generate(**inputs, 
                                            max_new_tokens=128, 
                                            do_sample=generate_config['do_sample'],
                                            temperature=generate_config['temperature'],
                                            top_p=generate_config['top_p'],)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs['input_ids'], generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        answer = output_text[0]
        return answer