import torch
from .model import VTR_Model
from .perception_encoder import pe
from .perception_encoder.tokenizer import SimpleTokenizer
import torchvision.transforms.v2 as T
import torchvision.transforms.functional as F

class PE_VTR_Model(VTR_Model):
    def __init__(self, model_path, device):
        super().__init__()
        self.device = device
        self.vtr_model_path = model_path
        model_name = str(model_path).split("/")[-1].split(".")[0]
        self.clip_model = pe.CLIP.from_config(model_name, pretrained=True,checkpoint_path=model_path,pool_type="attn").to(device)
        self.image_size = self.clip_model.image_size
        self.context_length = self.clip_model.context_length
        self.tokenizer = SimpleTokenizer(context_length=self.context_length)
        self.image_transform = T.Compose([
            T.Resize((self.image_size, self.image_size), interpolation=F.InterpolationMode.BICUBIC),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def get_text_embedding(self, text):
        text_inputs = self.tokenizer(text).to(self.device)
        with torch.no_grad():
            text_embedding = self.clip_model.encode_text(text_inputs, normalize=True)
        return text_embedding 
    
    def get_video_embedding(self,
                            frames, return_image_embeddings=False):
        images = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0  # [N, C, H, W]
        images = images.to(self.device)
        frames = self.image_transform(images) # [N, C, H, W]
        with torch.no_grad():
            image_embeddings = self.clip_model.encode_image(frames, normalize=True) #[N, dim]
        if return_image_embeddings:
            return image_embeddings
        video_embedding = image_embeddings.unsqueeze(0).mean(dim=1) #[1, dim]
        return video_embedding