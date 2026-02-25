import torch
from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

class VTR_Model(ABC):

    def get_text_embedding(self, text):
        r"""
            text : str
            return : torch.tensor , shape : [1, dim]
        """
        pass

    @abstractmethod
    def get_video_embedding(self,
                            frames, return_image_embeddings=False):
        r"""
            frames : np.array  , shape : [N, H, W, 3], directly from PIL.Image or decord.get_batch
            return : torch.tensor , shape : [1, dim]
        """
        pass

    def get_all_video_embeds(self, total_frames, vtr_config):
        total_frame_num = len(total_frames)
        frame_idx_s = []
        for j in range(0, total_frame_num, vtr_config['frame_num']):
            if j + vtr_config['frame_num'] > total_frame_num:
                j = total_frame_num - vtr_config['frame_num']
            frame_idx_s.append(list(range(j, j+vtr_config['frame_num'])))
            if j + vtr_config['frame_num'] == total_frame_num:
                break
        video_embeddings = []
        for frame_idx in frame_idx_s:
            video_embedding = self.get_video_embedding(total_frames[frame_idx])
            video_embeddings.append(video_embedding)
        video_embeddings = torch.cat(video_embeddings)
        return video_embeddings

    def get_similarities(self, total_frames, QAs, vtr_config):

        video_embeddings = self.get_all_video_embeds(total_frames, vtr_config)

        similarities = []

        for qa in QAs:
            similarity = []
            for vtr in qa['vtr_texts']:
                text_embedding = self.get_text_embedding(vtr)
                sim_matrix = text_embedding @ video_embeddings.transpose(0,1)
                s = sim_matrix[0].cpu().tolist()
                similarity.append(s)
            similarities.append(similarity)

        # T = video_embeddings.shape[0]
        sim_matrix = cosine_similarity(video_embeddings.detach().cpu().numpy())
        # G = nx.from_numpy_array(sim_matrix)
        # scores = nx.pagerank(G)
        # saliency = np.array([scores[i] for i in range(T)])
        # # saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min())
        # saliency = saliency.tolist()
        
        # return similarities, saliency # [N, 4, F], [F]
        return similarities, sim_matrix.tolist() # [N, 4, F], [F, F]