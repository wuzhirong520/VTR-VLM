import torch
from transformers import CLIPTokenizer, CLIPModel, AutoProcessor
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from PIL import Image
from .model import VTR_Model

class Clip_VTR_Model(VTR_Model):
    def __init__(self, model_path, device):
        super().__init__()
        self.device = device
        self.vtr_model_path = model_path
        self.vtr_model = CLIPModel.from_pretrained(self.vtr_model_path).to(device)
        self.vtr_tokenizer = CLIPTokenizer.from_pretrained(self.vtr_model_path)
        self.vtr_processor = AutoProcessor.from_pretrained(self.vtr_model_path)

    def get_text_embedding(self, text):
        text_inputs = self.vtr_tokenizer([text], padding="max_length", return_tensors="pt")
        if text_inputs["input_ids"].shape[1] > 77:
            print("!!!!!!!!!!!!", text_inputs["input_ids"].shape)
            text_inputs["input_ids"] = text_inputs["input_ids"][:,:77]
            text_inputs["attention_mask"] = text_inputs["attention_mask"][:,:77]
        text_inputs["input_ids"] = text_inputs["input_ids"].to(self.device)
        text_inputs["attention_mask"] = text_inputs["attention_mask"].to(self.device)
        with torch.no_grad():
            text_outputs = self.vtr_model.text_model(**text_inputs)[1]
            text_outputs = self.vtr_model.text_projection(text_outputs)
        text_embedding = text_outputs / text_outputs.norm(dim=-1, keepdim=True)
        return text_embedding
    
    def get_video_embedding(self,
                            frames):
        video_split = self.vtr_processor(images = frames, return_tensors="pt")
        video_split['pixel_values'] = video_split['pixel_values'].to(self.device)
        with torch.no_grad():
            visual_output = self.vtr_model.vision_model(**video_split)[1]
            visual_output = self.vtr_model.visual_projection(visual_output)
        visual_output = torch.mean(visual_output, dim=0)
        visual_output = visual_output / visual_output.norm(dim=-1, keepdim=True)
        video_embedding = visual_output.unsqueeze(0)
        return video_embedding 