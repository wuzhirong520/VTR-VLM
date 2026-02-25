import torch
from transformers import SiglipTextModel, AutoTokenizer, SiglipVisionModel, AutoProcessor
from .model import VTR_Model

class Siglip_VTR_Model(VTR_Model):
    def __init__(self, model_path, device):
        super().__init__()
        self.device = device
        self.vtr_model_path = model_path
        self.vtr_text_model = SiglipTextModel.from_pretrained(self.vtr_model_path).to(device)
        self.vtr_tokenizer = AutoTokenizer.from_pretrained(self.vtr_model_path)
        self.vtr_video_model = SiglipVisionModel.from_pretrained(self.vtr_model_path).to(torch.bfloat16).to(device)
        self.vtr_processor = AutoProcessor.from_pretrained(self.vtr_model_path)

    def get_text_embedding(self, text):
        text_inputs = self.vtr_tokenizer([text], padding="max_length", return_tensors="pt")
        if text_inputs["input_ids"].shape[1] > 64:
            print("!!!!!!!!!!!!", text_inputs["input_ids"].shape)
            text_inputs["input_ids"] = text_inputs["input_ids"][:,:64]
        text_inputs["input_ids"] = text_inputs["input_ids"].to(self.device)
        with torch.no_grad():
            text_outputs = self.vtr_text_model(**text_inputs)
        text_outputs = text_outputs.pooler_output 
        text_embedding = text_outputs / text_outputs.norm(dim=-1, keepdim=True)
        return text_embedding 
    
    def get_video_embedding(self,
                            frames):
        video_split = self.vtr_processor(images = frames, return_tensors="pt")
        video_split['pixel_values'] = video_split['pixel_values'].to(torch.bfloat16).to(self.device)
        with torch.no_grad():
            visual_output = self.vtr_video_model(**video_split)
        visual_output = visual_output.pooler_output
        # visual_output = visual_output / visual_output.norm(dim=-1, keepdim=True)
        visual_output = torch.mean(visual_output.float(), dim=0)
        visual_output = visual_output / visual_output.norm(dim=-1, keepdim=True)
        video_embedding = visual_output.unsqueeze(0)
        return video_embedding