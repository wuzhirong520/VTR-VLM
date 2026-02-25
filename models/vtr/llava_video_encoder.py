from .llava_video_encoder_src.model.builder import load_pretrained_model
from .llava_video_encoder_src.mm_utils import tokenizer_image_token
from .llava_video_encoder_src.constants import IMAGE_TOKEN_INDEX
from .llava_video_encoder_src.conversation import conv_templates

# from llava_video_encoder_src.model.builder import load_pretrained_model
# from llava_video_encoder_src.mm_utils import tokenizer_image_token
# from llava_video_encoder_src.constants import IMAGE_TOKEN_INDEX
# from llava_video_encoder_src.conversation import conv_templates

import torch
import copy
import numpy as np
from PIL import Image
import decord
from safetensors import safe_open
import os
from transformers import AutoTokenizer, SiglipTextModel

from .model import VTR_Model
class LLavaVideoEncoder(VTR_Model):
    def __init__(self, model_path, device):
        super().__init__()
        self.device = device
        self.model_path = model_path
        self.siglip_model_path = "/mnt/bn/wzr/models/siglip-so400m-patch14-384"
        self.siglip_text_model = None
        self.llava_model = None

    def load_video_model(self):
        self.llava_tokenizer, self.llava_model, self.llava_processor, _ = load_pretrained_model(
            self.model_path, None, "llava_qwen", 
            torch_dtype="bfloat16", 
            device_map=self.device,
            # overwrite_config=overwrite_config
        )
        with safe_open(os.path.join(self.siglip_model_path, "model.safetensors"), framework="pt", device="cpu") as f:
            state_dict = {}
            for k in f.keys():
                if str(k).startswith("vision_model.head."):
                    state_dict.update({k[len("vision_model.head."):]: f.get_tensor(k)})
            # print(state_dict.keys())
            self.llava_model.get_model().get_vision_tower().vision_tower.vision_model.head.load_state_dict(state_dict)

    def load_text_model(self):
        self.siglip_tokenizer = AutoTokenizer.from_pretrained(self.siglip_model_path)
        self.siglip_text_model = SiglipTextModel.from_pretrained(self.siglip_model_path).to(self.device)

    def get_text_embedding(self, text):
        if self.siglip_text_model is None:
            self.load_text_model()
        text_inputs = self.siglip_tokenizer([text], padding="max_length", return_tensors="pt")
        if text_inputs["input_ids"].shape[1] > 64:
            print("!!!!!!!!!!!!", text_inputs["input_ids"].shape)
            text_inputs["input_ids"] = text_inputs["input_ids"][:,:64]
        text_inputs["input_ids"] = text_inputs["input_ids"].to(self.device)
        with torch.no_grad():
            text_outputs = self.siglip_text_model(**text_inputs)
        text_outputs = text_outputs.pooler_output 
        text_embedding = text_outputs / text_outputs.norm(dim=-1, keepdim=True)
        return text_embedding 

    
    def get_video_embedding(self,
                            frames, return_image_embeddings=False):
        if self.llava_model is None:
            self.load_video_model()
        with torch.no_grad():
            video = self.llava_processor.preprocess(frames, return_tensors="pt")["pixel_values"].to(torch.bfloat16).to(self.device)
            encoded_image_features, video_pooler_output = self.llava_model.get_model().get_vision_tower()(video)
            visual_output = torch.mean(video_pooler_output.float(), dim=0)
            visual_output = visual_output / visual_output.norm(dim=-1, keepdim=True)
            video_embedding = visual_output.unsqueeze(0)
        return video_embedding
    
if __name__ == "__main__":
    def get_video_frames_for_default_llava(video_path, max_frames_num=64,fps=1,force_sample=False):
        if max_frames_num == 0:
            return np.zeros((1, 336, 336, 3))
        # vr = decord.VideoReader(video_path, ctx=decord.cpu(0), num_threads=1)
        vr = decord.VideoReader(video_path)
        total_frame_num = len(vr)
        video_time = total_frame_num / vr.get_avg_fps()
        fps = round(vr.get_avg_fps()/fps)
        frame_idx = [i for i in range(0, len(vr), fps)]
        frame_time = [i/fps for i in frame_idx]
        if len(frame_idx) > max_frames_num or force_sample:
            sample_fps = max_frames_num
            uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
            frame_idx = uniform_sampled_frames.tolist()
            frame_time = [i/vr.get_avg_fps() for i in frame_idx]
        frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
        spare_frames = vr.get_batch(frame_idx).asnumpy()
        time_instruciton = f"The video lasts for {video_time:.2f} seconds, and {len(spare_frames)} frames are uniformly sampled from it. These frames are located at {frame_time}.Please answer the following questions related to this video."
        return spare_frames, time_instruciton
    
    model = LLavaVideoEncoder("/mnt/bn/wzr/models/LLaVA-Video-7B-Qwen2", "cuda")
    frames, _ = get_video_frames_for_default_llava("/mnt/bn/wzr/datasets/VideoMME/data/_8lBR0E_Tx8.mp4")
    video_embedding = model.get_video_embedding(frames)
    text_embedding = model.get_text_embedding("What is the action in this video?")
    print(video_embedding @ text_embedding.T)

# print(frames.shape)

# model = Model()
# model.model.get_model().get_vision_tower().vision_tower.vision_model.head.load_state_dict(state_dict)

    # answer = model.infer("What is the action in this video?", frames)

# from .model import VTR_Model
# class LLavaVideoEncoder(VTR_Model):
#     def __init__(self, model_path, device):
#         super().__init__()
#         self.device = device
#         self.tokenizer, self.model, self.processor, _ = load_pretrained_model(
#             model_path, None, "llava_qwen", 
#             torch_dtype="bfloat16", 
#             device_map=device,
#             # overwrite_config=overwrite_config
#         )
#     def get_video_feature_raw(self, images, modalities = ["video"]):
#         def unpad_image(tensor, original_size):
#             """
#             Unpads a PyTorch tensor of a padded and resized image.

#             Args:
#             tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
#             original_size (tuple): The original size of the image (height, width).

#             Returns:
#             torch.Tensor: The unpadded image tensor.
#             """
#             original_width, original_height = original_size
#             current_height, current_width = tensor.shape[1:]

#             # Compute aspect ratios
#             original_aspect_ratio = original_width / original_height
#             current_aspect_ratio = current_width / current_height

#             # Determine padding size and direction
#             if original_aspect_ratio > current_aspect_ratio:
#                 # Padding was added to the height
#                 scale_factor = current_width / original_width
#                 new_height = int(original_height * scale_factor)
#                 padding = (current_height - new_height) // 2
#                 unpadded_tensor = tensor[:, padding : current_height - padding, :]
#             else:
#                 # Padding was added to the width
#                 scale_factor = current_height / original_height
#                 new_width = int(original_width * scale_factor)
#                 padding = (current_width - new_width) // 2
#                 unpadded_tensor = tensor[:, :, padding : current_width - padding]

#             return unpadded_tensor

#         images = [images]

#         if type(images) is list:
#                 images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]

#         video_idx_in_batch = []
#         for _ in range(len(modalities)):
#             if modalities[_] == "video":
#                 video_idx_in_batch.append(_)

#         images_list = []
#         for image in images:
#             if image.ndim == 4:
#                 images_list.append(image)
#             else:
#                 images_list.append(image.unsqueeze(0))

#         concat_images = torch.cat([image for image in images_list], dim=0)
#         split_sizes = [image.shape[0] for image in images_list]
#         encoded_image_features = self.model.encode_images(concat_images)
#         # image_features,all_faster_video_features = self.encode_multimodals(concat_images, video_idx_in_batch, split_sizes)

#         # This is a list, each element is [num_images, patch * patch, dim]
#         # rank_print(f"Concat images : {concat_images.shape}")
#         encoded_image_features = torch.split(encoded_image_features, split_sizes)
#         image_features = []
#         for idx, image_feat in enumerate(encoded_image_features):
#             if idx in video_idx_in_batch:
#                 image_features.append(self.model.get_2dPool(image_feat))
#             else:
#                 image_features.append(image_feat)
#         # image_features = self.encode_multimodals(concat_images, video_idx_in_batch, split_sizes)
#         # rank_print(f"Encoded image feats : {[x.shape for x in image_features]}")
#         # image_features = torch.split(image_features, split_sizes, dim=0)
#         mm_patch_merge_type = getattr(self.model.config, "mm_patch_merge_type", "flat")
#         image_aspect_ratio = getattr(self.model.config, "image_aspect_ratio", "square")
#         mm_newline_position = getattr(self.model.config, "mm_newline_position", "one_token")

#         if mm_patch_merge_type == "flat":
#             image_features = [x.flatten(0, 1) for x in image_features]

#         elif mm_patch_merge_type.startswith("spatial"):
#             new_image_features = []
#             for image_idx, image_feature in enumerate(image_features):
#                 # FIXME: now assume the image is square, and split to 2x2 patches
#                 # num_patches = h * w, where h = w = sqrt(num_patches)
#                 # currently image_feature is a tensor of shape (4, num_patches, hidden_size)
#                 # we want to first unflatten it to (2, 2, h, w, hidden_size)
#                 # rank0_print("At least we are reaching here")
#                 # import pdb; pdb.set_trace()
#                 if image_idx in video_idx_in_batch:  # video operations
#                     # rank0_print("Video")
#                     if mm_newline_position == "grid":
#                         # Grid-wise
#                         image_feature = self.model.add_token_per_grid(image_feature)
#                         if getattr(self.model.config, "add_faster_video", False):
#                             faster_video_feature = self.model.add_token_per_grid(all_faster_video_features[image_idx])
#                             # Add a token for each frame
#                             concat_slow_fater_token = []
#                             # import pdb; pdb.set_trace()
#                             for _ in range(image_feature.shape[0]):
#                                 if _ % self.model.config.faster_token_stride == 0:
#                                     concat_slow_fater_token.append(torch.cat((image_feature[_], self.model.model.faster_token[None].to(image_feature.device)), dim=0))
#                                 else:
#                                     concat_slow_fater_token.append(torch.cat((faster_video_feature[_], self.model.model.faster_token[None].to(image_feature.device)), dim=0))
#                             # import pdb; pdb.set_trace()
#                             image_feature = torch.cat(concat_slow_fater_token)

#                             # print("!!!!!!!!!!!!")
                    
#                         new_image_features.append(image_feature)
#                     elif mm_newline_position == "frame":
#                         # Frame-wise
#                         image_feature = self.model.add_token_per_frame(image_feature)

#                         new_image_features.append(image_feature.flatten(0, 1))
                        
#                     elif mm_newline_position == "one_token":
#                         # one-token
#                         image_feature = image_feature.flatten(0, 1)
#                         if 'unpad' in mm_patch_merge_type:
#                             image_feature = torch.cat((
#                                 image_feature,
#                                 self.model.model.image_newline[None].to(image_feature.device)
#                             ), dim=0)
#                         new_image_features.append(image_feature)      
#                     elif mm_newline_position == "no_token":
#                         new_image_features.append(image_feature.flatten(0, 1))
#                     else:
#                         raise ValueError(f"Unexpected mm_newline_position: {mm_newline_position}")
#                 elif image_feature.shape[0] > 1:  # multi patches and multi images operations
#                     # rank0_print("Single-images")
#                     base_image_feature = image_feature[0]
#                     image_feature = image_feature[1:]
#                     height = width = self.model.get_vision_tower().num_patches_per_side
#                     assert height * width == base_image_feature.shape[0]

#                     if "anyres_max" in image_aspect_ratio:
#                         matched_anyres_max_num_patches = re.match(r"anyres_max_(\d+)", image_aspect_ratio)
#                         if matched_anyres_max_num_patches:
#                             max_num_patches = int(matched_anyres_max_num_patches.group(1))

#                     if image_aspect_ratio == "anyres" or "anyres_max" in image_aspect_ratio:
#                         if hasattr(self.model.get_vision_tower(), "image_size"):
#                             vision_tower_image_size = self.get_vision_tower().image_size
#                         else:
#                             raise ValueError("vision_tower_image_size is not found in the vision tower.")
#                         try:
#                             num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, vision_tower_image_size)
#                         except Exception as e:
#                             rank0_print(f"Error: {e}")
#                             num_patch_width, num_patch_height = 2, 2
#                         image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
#                     else:
#                         image_feature = image_feature.view(2, 2, height, width, -1)

#                     if "maxpool2x2" in mm_patch_merge_type:
#                         image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
#                         image_feature = image_feature.flatten(1, 2).flatten(2, 3)
#                         image_feature = nn.functional.max_pool2d(image_feature, 2)
#                         image_feature = image_feature.flatten(1, 2).transpose(0, 1)
#                     elif "unpad" in mm_patch_merge_type and "anyres_max" in image_aspect_ratio and matched_anyres_max_num_patches:
#                         unit = image_feature.shape[2]
#                         image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
#                         image_feature = image_feature.flatten(1, 2).flatten(2, 3)
#                         image_feature = unpad_image(image_feature, image_sizes[image_idx])
#                         c, h, w = image_feature.shape
#                         times = math.sqrt(h * w / (max_num_patches * unit**2))
#                         if times > 1.1:
#                             image_feature = image_feature[None]
#                             image_feature = nn.functional.interpolate(image_feature, [int(h // times), int(w // times)], mode="bilinear")[0]
#                         image_feature = torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
#                         image_feature = image_feature.flatten(1, 2).transpose(0, 1)
#                     elif "unpad" in mm_patch_merge_type:
#                         image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
#                         image_feature = image_feature.flatten(1, 2).flatten(2, 3)
#                         image_feature = unpad_image(image_feature, image_sizes[image_idx])
#                         image_feature = torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
#                         image_feature = image_feature.flatten(1, 2).transpose(0, 1)
#                     else:
#                         image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
#                         image_feature = image_feature.flatten(0, 3)
#                     if "nobase" in mm_patch_merge_type:
#                         pass
#                     else:
#                         image_feature = torch.cat((base_image_feature, image_feature), dim=0)
#                     new_image_features.append(image_feature)
#                 else:  # single image operations
#                     image_feature = image_feature[0]
#                     if "unpad" in mm_patch_merge_type:
#                         image_feature = torch.cat((image_feature, self.model.image_newline[None]), dim=0)

#                     new_image_features.append(image_feature)
#             image_features = new_image_features

#         return image_features[0]

#     def get_text_embedding(self, text):
#         with torch.no_grad():
#             conv = copy.deepcopy(conv_templates["qwen_1_5"])
#             conv.append_message(conv.roles[0], text)
#             conv.append_message(conv.roles[1], None)
#             prompt_question = conv.get_prompt()
#             input_ids = tokenizer_image_token(prompt_question, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)
#             text_tokens = self.model.get_model().embed_tokens(input_ids[0])
#             text_embedding = torch.mean(text_tokens, dim=0)
#             text_embedding = (text_embedding / torch.norm(text_embedding, p=2)).unsqueeze(0)
#         return text_embedding

    
#     def get_video_embedding(self,
#                             frames, return_image_embeddings=False):
#         with torch.no_grad():
#             video = self.processor.preprocess(frames, return_tensors="pt")["pixel_values"].to(torch.bfloat16).to(self.device)
#             video_tokens = self.get_video_feature_raw(video)
#             video_embedding = torch.mean(video_tokens, dim=0)
#             video_embedding = (video_embedding / torch.norm(video_embedding, p=2)).unsqueeze(0)
#         return video_embedding

# class Model():
#     def __init__(self):
#         model_name = "llava_qwen"
#         self.tokenizer, self.model, self.processor, _ = load_pretrained_model(
#             "/mnt/bn/wzr/models/LLaVA-Video-7B-Qwen2", None, model_name, 
#             torch_dtype="bfloat16", 
#             device_map="auto",
#             # overwrite_config=overwrite_config
#         )

#     def get_video_feature(self, images, modalities = ["video"]):
#         def unpad_image(tensor, original_size):
#             """
#             Unpads a PyTorch tensor of a padded and resized image.

#             Args:
#             tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
#             original_size (tuple): The original size of the image (height, width).

#             Returns:
#             torch.Tensor: The unpadded image tensor.
#             """
#             original_width, original_height = original_size
#             current_height, current_width = tensor.shape[1:]

#             # Compute aspect ratios
#             original_aspect_ratio = original_width / original_height
#             current_aspect_ratio = current_width / current_height

#             # Determine padding size and direction
#             if original_aspect_ratio > current_aspect_ratio:
#                 # Padding was added to the height
#                 scale_factor = current_width / original_width
#                 new_height = int(original_height * scale_factor)
#                 padding = (current_height - new_height) // 2
#                 unpadded_tensor = tensor[:, padding : current_height - padding, :]
#             else:
#                 # Padding was added to the width
#                 scale_factor = current_height / original_height
#                 new_width = int(original_width * scale_factor)
#                 padding = (current_width - new_width) // 2
#                 unpadded_tensor = tensor[:, :, padding : current_width - padding]

#             return unpadded_tensor

#         images = [images]

#         if type(images) is list:
#                 images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]

#         video_idx_in_batch = []
#         for _ in range(len(modalities)):
#             if modalities[_] == "video":
#                 video_idx_in_batch.append(_)

#         images_list = []
#         for image in images:
#             if image.ndim == 4:
#                 images_list.append(image)
#             else:
#                 images_list.append(image.unsqueeze(0))

#         concat_images = torch.cat([image for image in images_list], dim=0)
#         split_sizes = [image.shape[0] for image in images_list]

#         # encoded_image_features = self.model.encode_images(concat_images)

#         encoded_image_features, video_pooler_output = self.model.get_model().get_vision_tower()(concat_images)
#         encoded_image_features = self.model.get_model().mm_projector(encoded_image_features)
#         print("Pooler Output: ", video_pooler_output.shape)


#         # image_features,all_faster_video_features = self.encode_multimodals(concat_images, video_idx_in_batch, split_sizes)

#         # This is a list, each element is [num_images, patch * patch, dim]
#         # rank_print(f"Concat images : {concat_images.shape}")
#         encoded_image_features = torch.split(encoded_image_features, split_sizes)
#         image_features = []
#         for idx, image_feat in enumerate(encoded_image_features):
#             if idx in video_idx_in_batch:
#                 image_features.append(self.model.get_2dPool(image_feat))
#             else:
#                 image_features.append(image_feat)
#         # image_features = self.encode_multimodals(concat_images, video_idx_in_batch, split_sizes)
#         # rank_print(f"Encoded image feats : {[x.shape for x in image_features]}")
#         # image_features = torch.split(image_features, split_sizes, dim=0)
#         mm_patch_merge_type = getattr(self.model.config, "mm_patch_merge_type", "flat")
#         image_aspect_ratio = getattr(self.model.config, "image_aspect_ratio", "square")
#         mm_newline_position = getattr(self.model.config, "mm_newline_position", "one_token")

#         if mm_patch_merge_type == "flat":
#             image_features = [x.flatten(0, 1) for x in image_features]

#         elif mm_patch_merge_type.startswith("spatial"):
#             new_image_features = []
#             for image_idx, image_feature in enumerate(image_features):
#                 # FIXME: now assume the image is square, and split to 2x2 patches
#                 # num_patches = h * w, where h = w = sqrt(num_patches)
#                 # currently image_feature is a tensor of shape (4, num_patches, hidden_size)
#                 # we want to first unflatten it to (2, 2, h, w, hidden_size)
#                 # rank0_print("At least we are reaching here")
#                 # import pdb; pdb.set_trace()
#                 if image_idx in video_idx_in_batch:  # video operations
#                     # rank0_print("Video")
#                     if mm_newline_position == "grid":
#                         # Grid-wise
#                         image_feature = self.model.add_token_per_grid(image_feature)
#                         if getattr(self.model.config, "add_faster_video", False):
#                             faster_video_feature = self.model.add_token_per_grid(all_faster_video_features[image_idx])
#                             # Add a token for each frame
#                             concat_slow_fater_token = []
#                             # import pdb; pdb.set_trace()
#                             for _ in range(image_feature.shape[0]):
#                                 if _ % self.model.config.faster_token_stride == 0:
#                                     concat_slow_fater_token.append(torch.cat((image_feature[_], self.model.model.faster_token[None].to(image_feature.device)), dim=0))
#                                 else:
#                                     concat_slow_fater_token.append(torch.cat((faster_video_feature[_], self.model.model.faster_token[None].to(image_feature.device)), dim=0))
#                             # import pdb; pdb.set_trace()
#                             image_feature = torch.cat(concat_slow_fater_token)

#                             # print("!!!!!!!!!!!!")
                    
#                         new_image_features.append(image_feature)
#                     elif mm_newline_position == "frame":
#                         # Frame-wise
#                         image_feature = self.model.add_token_per_frame(image_feature)

#                         new_image_features.append(image_feature.flatten(0, 1))
                        
#                     elif mm_newline_position == "one_token":
#                         # one-token
#                         image_feature = image_feature.flatten(0, 1)
#                         if 'unpad' in mm_patch_merge_type:
#                             image_feature = torch.cat((
#                                 image_feature,
#                                 self.model.model.image_newline[None].to(image_feature.device)
#                             ), dim=0)
#                         new_image_features.append(image_feature)      
#                     elif mm_newline_position == "no_token":
#                         new_image_features.append(image_feature.flatten(0, 1))
#                     else:
#                         raise ValueError(f"Unexpected mm_newline_position: {mm_newline_position}")
#                 elif image_feature.shape[0] > 1:  # multi patches and multi images operations
#                     # rank0_print("Single-images")
#                     base_image_feature = image_feature[0]
#                     image_feature = image_feature[1:]
#                     height = width = self.model.get_vision_tower().num_patches_per_side
#                     assert height * width == base_image_feature.shape[0]

#                     if "anyres_max" in image_aspect_ratio:
#                         matched_anyres_max_num_patches = re.match(r"anyres_max_(\d+)", image_aspect_ratio)
#                         if matched_anyres_max_num_patches:
#                             max_num_patches = int(matched_anyres_max_num_patches.group(1))

#                     if image_aspect_ratio == "anyres" or "anyres_max" in image_aspect_ratio:
#                         if hasattr(self.model.get_vision_tower(), "image_size"):
#                             vision_tower_image_size = self.get_vision_tower().image_size
#                         else:
#                             raise ValueError("vision_tower_image_size is not found in the vision tower.")
#                         try:
#                             num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, vision_tower_image_size)
#                         except Exception as e:
#                             rank0_print(f"Error: {e}")
#                             num_patch_width, num_patch_height = 2, 2
#                         image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
#                     else:
#                         image_feature = image_feature.view(2, 2, height, width, -1)

#                     if "maxpool2x2" in mm_patch_merge_type:
#                         image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
#                         image_feature = image_feature.flatten(1, 2).flatten(2, 3)
#                         image_feature = nn.functional.max_pool2d(image_feature, 2)
#                         image_feature = image_feature.flatten(1, 2).transpose(0, 1)
#                     elif "unpad" in mm_patch_merge_type and "anyres_max" in image_aspect_ratio and matched_anyres_max_num_patches:
#                         unit = image_feature.shape[2]
#                         image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
#                         image_feature = image_feature.flatten(1, 2).flatten(2, 3)
#                         image_feature = unpad_image(image_feature, image_sizes[image_idx])
#                         c, h, w = image_feature.shape
#                         times = math.sqrt(h * w / (max_num_patches * unit**2))
#                         if times > 1.1:
#                             image_feature = image_feature[None]
#                             image_feature = nn.functional.interpolate(image_feature, [int(h // times), int(w // times)], mode="bilinear")[0]
#                         image_feature = torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
#                         image_feature = image_feature.flatten(1, 2).transpose(0, 1)
#                     elif "unpad" in mm_patch_merge_type:
#                         image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
#                         image_feature = image_feature.flatten(1, 2).flatten(2, 3)
#                         image_feature = unpad_image(image_feature, image_sizes[image_idx])
#                         image_feature = torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
#                         image_feature = image_feature.flatten(1, 2).transpose(0, 1)
#                     else:
#                         image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
#                         image_feature = image_feature.flatten(0, 3)
#                     if "nobase" in mm_patch_merge_type:
#                         pass
#                     else:
#                         image_feature = torch.cat((base_image_feature, image_feature), dim=0)
#                     new_image_features.append(image_feature)
#                 else:  # single image operations
#                     image_feature = image_feature[0]
#                     if "unpad" in mm_patch_merge_type:
#                         image_feature = torch.cat((image_feature, self.model.image_newline[None]), dim=0)

#                     new_image_features.append(image_feature)
#             image_features = new_image_features

#         return image_features[0], video_pooler_output

#     def infer(self, prompt, frames):
#         video = self.processor.preprocess(frames, return_tensors="pt")["pixel_values"].to(torch.bfloat16).to("cuda")
#         print(video.shape)
#         conv = copy.deepcopy(conv_templates["qwen_1_5"])
#         conv.append_message(conv.roles[0], prompt)
#         conv.append_message(conv.roles[1], None)
#         prompt_question = conv.get_prompt()
#         input_ids = tokenizer_image_token(prompt_question, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to("cuda")
#         print(input_ids.shape)
#         text_tokens = self.model.get_model().embed_tokens(input_ids[0])
#         print("text: ",text_tokens.shape)
#         video_tokens, video_pooler_output = self.get_video_feature(video)
#         print("video: ",video_tokens.shape)

#         video_embedding = torch.mean(video_tokens, dim=0)
#         video_embedding = (video_embedding / torch.norm(video_embedding, p=2)).unsqueeze(0)
#         text_embedding = torch.mean(text_tokens, dim=0)
#         text_embedding = (text_embedding / torch.norm(text_embedding, p=2)).unsqueeze(0)

#         print(video_embedding.shape, text_embedding.shape)
        
#         print(video_embedding @ text_embedding.T)

#         from transformers import AutoTokenizer, SiglipTextModel
#         self.origin_siglip_tokenizer = AutoTokenizer.from_pretrained("/mnt/bn/wzr/models/siglip-so400m-patch14-384")
#         self.origin_siglip_text_model = SiglipTextModel.from_pretrained("/mnt/bn/wzr/models/siglip-so400m-patch14-384").to("cuda")
#         text_inputs = self.origin_siglip_tokenizer([prompt], padding="max_length", return_tensors="pt").to("cuda")
#         with torch.no_grad():
#             text_features = self.origin_siglip_text_model(**text_inputs)
#         text_embedding2 = text_features.pooler_output

#         print(text_embedding2)
#         visual_output = torch.mean(video_pooler_output.float(), dim=0)
#         print(visual_output)
#         visual_output = visual_output / visual_output.norm(dim=-1, keepdim=True)
#         video_embedding2 = visual_output.unsqueeze(0)
#         text_embedding2 = text_embedding2 / text_embedding2.norm(dim=-1, keepdim=True)
#         print(video_embedding2.shape, text_embedding2.shape)

#         print(video_embedding2, text_embedding2)

#         print(video_embedding2 @ text_embedding2.T)

#         return "xx"
    

# def get_video_frames_for_default_llava(video_path, max_frames_num=64,fps=1,force_sample=False):
#     if max_frames_num == 0:
#         return np.zeros((1, 336, 336, 3))
#     # vr = decord.VideoReader(video_path, ctx=decord.cpu(0), num_threads=1)
#     vr = decord.VideoReader(video_path)
#     total_frame_num = len(vr)
#     video_time = total_frame_num / vr.get_avg_fps()
#     fps = round(vr.get_avg_fps()/fps)
#     frame_idx = [i for i in range(0, len(vr), fps)]
#     frame_time = [i/fps for i in frame_idx]
#     if len(frame_idx) > max_frames_num or force_sample:
#         sample_fps = max_frames_num
#         uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
#         frame_idx = uniform_sampled_frames.tolist()
#         frame_time = [i/vr.get_avg_fps() for i in frame_idx]
#     frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
#     spare_frames = vr.get_batch(frame_idx).asnumpy()
#     time_instruciton = f"The video lasts for {video_time:.2f} seconds, and {len(spare_frames)} frames are uniformly sampled from it. These frames are located at {frame_time}.Please answer the following questions related to this video."
#     return spare_frames, time_instruciton





# from safetensors import safe_open
# with safe_open("/mnt/bn/wzr/models/siglip-so400m-patch14-384/model.safetensors", framework="pt", device="cpu") as f:
#     # print("Keys:", f.keys())
#     state_dict = {}
#     for k in f.keys():
#         if str(k).startswith("vision_model.head."):
#             state_dict.update({k[len("vision_model.head."):]: f.get_tensor(k)})
#     print(state_dict.keys())

# frames, _ = get_video_frames_for_default_llava("/mnt/bn/wzr/datasets/VideoMME/data/_8lBR0E_Tx8.mp4")

# print(frames.shape)

# model = Model()
# model.model.get_model().get_vision_tower().vision_tower.vision_model.head.load_state_dict(state_dict)

# answer = model.infer("What is the action in this video?", frames)

# print(answer)
