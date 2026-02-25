import torch
from transformers import CLIPTokenizer, CLIPTextModelWithProjection, CLIPVisionModelWithProjection
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from PIL import Image
from .model import VTR_Model

class CLIP4Clip_VTR_Model(VTR_Model):
    def __init__(self, model_path, device):
        super().__init__()
        self.device = device
        self.vtr_model_path = model_path
        self.vtr_text_model = CLIPTextModelWithProjection.from_pretrained(self.vtr_model_path).to(device)
        self.vtr_video_model = CLIPVisionModelWithProjection.from_pretrained(self.vtr_model_path).to(device)
        self.vtr_tokenizer = CLIPTokenizer.from_pretrained(self.vtr_model_path)
        self.vtr_transforms = Compose([
            Resize((224,224), interpolation=Image.BICUBIC),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    def get_text_embedding(self, text):
        text_inputs = self.vtr_tokenizer(text=text, return_tensors="pt")
        if text_inputs["input_ids"].shape[1] > 77:
            print("!!!!!!!!!!!!", text_inputs["input_ids"].shape)
            text_inputs["input_ids"] = text_inputs["input_ids"][:,:77]
            text_inputs["attention_mask"] = text_inputs["attention_mask"][:,:77]
        text_ids, text_mask = text_inputs["input_ids"].to(self.device), text_inputs["attention_mask"].to(self.device)
        with torch.no_grad():
            text_outputs = self.vtr_text_model(input_ids=text_ids, attention_mask=text_mask)
        text_embedding = text_outputs[0] / text_outputs[0].norm(dim=-1, keepdim=True)
        return text_embedding
    
    def get_video_embedding(self,
                            frames):

        video_split = [self.vtr_transforms(Image.fromarray(f)) for f in frames]
        video_split = torch.stack(video_split).to(self.device)
        with torch.no_grad():
            visual_output = self.vtr_video_model(video_split)
        visual_output = visual_output["image_embeds"]
        visual_output = visual_output / visual_output.norm(dim=-1, keepdim=True)
        visual_output = torch.mean(visual_output, dim=0)
        visual_output = visual_output / visual_output.norm(dim=-1, keepdim=True)
        video_embedding = visual_output.unsqueeze(0)
        return video_embedding 