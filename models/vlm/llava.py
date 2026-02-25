import torch
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
import copy
from .model import VLM_Model

class LLaVA_VLM_Model(VLM_Model):
    def __init__(self, model_path, extra_model_config):
        self.vlm_model_path = model_path
        self.extra_model_config = extra_model_config
        # overwrite_config = {}
        # overwrite_config["mm_spatial_pool_stride"] = 2
        # overwrite_config["mm_spatial_pool_mode"] = 'average'
        # overwrite_config["mm_pooling_position"] = 'before'
        # overwrite_config["mm_newline_position"] = 'grid'
        model_name = "llava_qwen"
        self.torch_dtype="bfloat16"
        # self.add_time_instruction = True
        self.add_time_instruction = False
        if self.extra_model_config is not None:
            if "torch_dtype" in self.extra_model_config:
                self.torch_dtype = self.extra_model_config["torch_dtype"]
            if "add_time_instruction" in self.extra_model_config:
                self.add_time_instruction = self.extra_model_config["add_time_instruction"]
                
        self.tokenizer, self.model, self.processor, _ = load_pretrained_model(
            self.vlm_model_path, None, model_name, 
            torch_dtype=self.torch_dtype, 
            device_map="auto",
            # overwrite_config=overwrite_config
        )
    
    def infer(self, prompt, frames_info, generate_config = {'do_sample':False, 'temperature':0, 'top_p': 1}):
        if frames_info is None:
            conv = copy.deepcopy(conv_templates["qwen_1_5"])
            conv.append_message(conv.roles[0], prompt)
            conv.append_message(conv.roles[1], None)
            prompt_question = conv.get_prompt()
            input_ids = tokenizer_image_token(prompt_question, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to("cuda")
            if generate_config['do_sample']:
                cont = self.model.generate(
                    input_ids,
                    images=None,
                    modalities=["text"],
                    do_sample=generate_config['do_sample'],
                    temperature=generate_config['temperature'],
                    top_p=generate_config['top_p'],
                    max_new_tokens=128,
                )
            else:
                cont = self.model.generate(
                    input_ids,
                    images=None,
                    modalities=["text"],
                    max_new_tokens=128,
                    do_sample = False
                )
        else:
            frames = frames_info['frames']
            if self.torch_dtype == "bfloat16":
                video = self.processor.preprocess(frames, return_tensors="pt")["pixel_values"].to(torch.bfloat16).to("cuda")
            else:
                video = self.processor.preprocess(frames, return_tensors="pt")["pixel_values"].to(torch.float16).to("cuda")
            conv = copy.deepcopy(conv_templates["qwen_1_5"])
            if self.add_time_instruction:
                if "time_instruction" in frames_info and frames_info['time_instruction'] is not None:
                    full_prompt = "<image>\n" + frames_info['time_instruction'] + "\n" + prompt
                else:
                    full_prompt = f"<image>\nThe video lasts for {frames_info['duration']:.2f} seconds, and {len(frames_info['frames_idx'])} frames are sampled from it. "
                    full_prompt += f"These frames are located at "
                    full_prompt += ",".join([f"{idx/frames_info['fps']:.2f}s" for idx in frames_info['frames_idx']])
                    full_prompt += ".Please answer the following questions related to this video."
                    print(full_prompt)
                    full_prompt += "\n" + prompt
            else:
                full_prompt = "<image>\n" + prompt
            # conv = copy.deepcopy(conv_templates["chatml_direct"])
            # full_prompt = "<image>\n" + prompt
            conv.append_message(conv.roles[0], full_prompt)
            conv.append_message(conv.roles[1], None)
            prompt_question = conv.get_prompt()
            input_ids = tokenizer_image_token(prompt_question, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to("cuda")
            if generate_config['do_sample']:
                cont = self.model.generate(
                    input_ids,
                    images=[video],
                    modalities=["video"],
                    do_sample=generate_config['do_sample'],
                    temperature=generate_config['temperature'],
                    top_p=generate_config['top_p'],
                    max_new_tokens=128,
                )
            else:
                cont = self.model.generate(
                    input_ids,
                    images=[video],
                    modalities=["video"],
                    max_new_tokens=128,
                    do_sample = False
                )
        answer = self.tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()
        return answer