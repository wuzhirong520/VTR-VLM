import torch
import numpy as np
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from qwen_vl_utils import smart_resize
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from .model import VLM_Model
import pulp

class Qwen3VL_VLM_Model(VLM_Model):
    def __init__(self, model_path, extra_model_config):
        self.vlm_model_path = model_path
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.vlm_model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(self.vlm_model_path)
    
    def infer(self, prompt, frames_info, generate_config = {'do_sample':False, 'temperature':0, 'top_p': 1}):
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
                video_metadata=[frames_info[0]['video_metadata']],
                do_resize=False,
                return_tensors="pt",
                do_sample_frames = False
            )
            inputs = inputs.to(self.model.device)
            inputs = dict(inputs)
        else:
            multi_resolution_video_embed = {}
            for fs in range(len(frames_info)):
                mvideo = frames_info[fs]['frames']
                old_inputs = self.processor.video_processor(
                    videos=[mvideo],
                    do_resize=False,
                    do_sample_frames = False,
                    return_tensors="pt"
                )
                old_inputs = old_inputs.to(self.model.device)
                with torch.no_grad():
                    old_inputs['pixel_values_videos'] = old_inputs['pixel_values_videos'].type(self.model.visual.dtype)
                    video_features, deepstack_video_features = self.model.visual(old_inputs['pixel_values_videos'], old_inputs['video_grid_thw'])
                grid_t = len(frames_info[fs]['frames']) // 2
                grid_h = frames_info[fs]['shape'][0] //32
                grid_w = frames_info[fs]['shape'][1] //32
                video_features = video_features.reshape(grid_t, -1, video_features.shape[-1])
                deepstack_video_features = [deepstack_video_features[i].reshape(grid_t, -1, deepstack_video_features[i].shape[-1]) for i in range(len(deepstack_video_features))]
                for j in range(grid_t):
                    idx = int(frames_info[fs]['global_idx'][j*2] + frames_info[fs]['global_idx'][j*2+1])//2
                    multi_resolution_video_embed[idx] = {
                        "embed" : video_features[j],
                        "deepstack_embed" : [deepstack_video_features[i][j] for i in range(len(deepstack_video_features))],
                        "h_index" : torch.arange(grid_h).view(-1, 1).expand(-1, grid_w).reshape(-1),
                        "w_index" : torch.arange(grid_w).view(1, -1).expand(grid_h, -1).reshape(-1),
                        "t_index" : torch.full((grid_h*grid_w,), 0, dtype=torch.int64),
                        "token_num" : grid_h*grid_w,
                        "timestamp" : f"<{idx/frames_info[fs]['fps']:.1f} seconds>"
                    }
            multi_resolution_video_embed = [[k,v] for k,v in multi_resolution_video_embed.items()]
            multi_resolution_video_embed = sorted(multi_resolution_video_embed, key = lambda x : x[0])
            multi_resolution_video_embed = [v for _,v in multi_resolution_video_embed]
            pre_calculated_visual_embedding = []
            pre_calculated_visual_embedding_deepstack = []
            t_index, h_index, w_index = [], [], []
            video_placeholder = ""
            for i in range(len(multi_resolution_video_embed)):
                t_index.append(torch.full_like(multi_resolution_video_embed[i]['t_index'], 0))
                h_index.append(multi_resolution_video_embed[i]['h_index'])
                w_index.append(multi_resolution_video_embed[i]['w_index'])
                pre_calculated_visual_embedding.append(multi_resolution_video_embed[i]['embed'])
                pre_calculated_visual_embedding_deepstack.append(multi_resolution_video_embed[i]['deepstack_embed'])
                video_placeholder += multi_resolution_video_embed[i]['timestamp']
                video_placeholder += (
                    self.processor.vision_start_token + "<|placeholder|>" * multi_resolution_video_embed[i]['token_num'] + self.processor.vision_end_token
                )
            pre_calculated_visual_embedding = torch.cat(pre_calculated_visual_embedding)
            pre_calculated_visual_embedding_deepstack = [torch.cat([pre_calculated_visual_embedding_deepstack[i][j] for i in range(len(pre_calculated_visual_embedding_deepstack))]) for j in range(len(pre_calculated_visual_embedding_deepstack[0]))]
            if f"{self.processor.vision_start_token}{self.processor.video_token}{self.processor.vision_end_token}" in text:
                final_text = text.replace(
                    f"{self.processor.vision_start_token}{self.processor.video_token}{self.processor.vision_end_token}", video_placeholder, 1
                )
            else:
                final_text = text.replace(self.processor.video_token, video_placeholder, 1)
            final_text = final_text.replace("<|placeholder|>", self.processor.video_token)

            text_inputs = self.processor.tokenizer(final_text)
            input_ids, attention_mask = torch.Tensor(text_inputs['input_ids']).long(), torch.Tensor(text_inputs['attention_mask']).long()

            position_ids = []
            last_vision_end_idx = 0
            vision_part_idx = 0
            last_t_idx = 0
            for i in range(len(input_ids)):
                if input_ids[i] != 151656: # vision pad token
                    last_t_idx += 1
                if input_ids[i] == 151652: # vision start token
                    position_ids.append(torch.arange(last_vision_end_idx, last_t_idx).view(1, -1).expand(3, -1))
                    position_ids.append(torch.stack([t_index[vision_part_idx], h_index[vision_part_idx], w_index[vision_part_idx]]) + last_t_idx)
                    vision_part_idx += 1
                    last_vision_end_idx = last_t_idx
            position_ids.append(torch.arange(last_vision_end_idx, last_t_idx).view(1, -1).expand(3, -1))
            position_ids = torch.cat(position_ids , dim=1).unsqueeze(1)
            mrope_position_deltas = torch.tensor([position_ids.max() + 1 - len(input_ids)]).unsqueeze(1)

            inputs = {
                "input_ids" : input_ids.unsqueeze(0).to(self.model.device),
                "attention_mask" : attention_mask.unsqueeze(0).to(self.model.device),
                "pre_calculated_visual_embedding" : pre_calculated_visual_embedding.to(self.model.device),
                "pre_calculated_visual_embedding_deepstack" : [v.to(self.model.device) for v in pre_calculated_visual_embedding_deepstack],
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