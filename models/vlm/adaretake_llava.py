import math
from tqdm import tqdm
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import numpy as np

from transformers.cache_utils import Cache
from transformers.utils import (
    is_flash_attn_2_available,
    logging,
)
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.processing_utils import Unpack
from transformers import Qwen2VLConfig
from transformers.models.llava_onevision.modeling_llava_onevision import (
    LlavaOnevisionCausalLMOutputWithPast,
    image_size_to_num_patches,
)
from transformers.models.qwen2.modeling_qwen2 import (
    repeat_kv,
    eager_attention_forward,
    Qwen2Attention,
    Qwen2RotaryEmbedding,
    apply_rotary_pos_emb
)

from transformers.cache_utils import DynamicCache

if is_flash_attn_2_available():
    from flash_attn import flash_attn_varlen_func
    from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
else:
    flash_attn_varlen_func = None

DEBUG_MODE = False

logger = logging.get_logger(__name__)

def memory_bank_compress_MALLM(memory_bank: torch.Tensor, compression_size: torch.Tensor, sync: bool=False) -> tuple:
    """
    Compresses the memory bank if the current memory bank length is greater than the threshold.
    Compression_size is the number of frames that are compressed into each position.
    
    Args:
        memory_bank (torch.Tensor): The input memory bank to compress. Shape: (B, T, N, C)
        compression_size (torch.Tensor): The number of frames to compress into each position. Shape: (B, T, N)
    
    Returns:
        compressed_memory_bank (torch.Tensor): The compressed memory bank. Shape: (B, T-1, N, C)
        compressed_size (torch.Tensor): The number of frames compressed into each position. Shape: (B, T-1, N)
    """
    B, T, N, C = memory_bank.shape
    # Calculate the cosine similarity between adjacent frames
    similarity_matrix = F.cosine_similarity(memory_bank[:, :-1, :], memory_bank[:, 1:, :], dim=-1)
    if sync:
        similarity_matrix = similarity_matrix.mean(-1, keepdim=True).expand(-1, -1, N)
    # Select the frame indices with the top-1 similarity 
    _, max_similarity_indices = torch.max(similarity_matrix, dim=1, keepdim=True)

    # Calculate source and dst indices for compression
    src_indices = max_similarity_indices + 1
    dst_indices = torch.arange(T - 1).to(memory_bank.device)[None, :, None].repeat(B, 1, N)
    dst_indices[dst_indices > max_similarity_indices] += 1

    # Gather source and dst memory banks and sizes
    src_memory_bank = memory_bank.gather(dim=1, index=src_indices.unsqueeze(-1).expand(-1, -1, -1, C))
    dst_memory_bank = memory_bank.gather(dim=1, index=dst_indices.unsqueeze(-1).expand(-1, -1, -1, C))
    src_size = compression_size.gather(dim=1, index=src_indices)
    dst_size = compression_size.gather(dim=1, index=dst_indices)

    # Multiply the memory banks by their corresponding sizes
    src_memory_bank *= src_size.unsqueeze(-1)
    dst_memory_bank *= dst_size.unsqueeze(-1)

    # Compress the memory bank by adding the source memory bank to the dst memory bank
    dst_memory_bank.scatter_add_(dim=1, index=max_similarity_indices.unsqueeze(-1).expand(-1, -1, -1, C), src=src_memory_bank)
    dst_size.scatter_add_(dim=1, index=max_similarity_indices, src=src_size)

    # Normalize the dst memory bank by its size
    compressed_memory_bank = dst_memory_bank / dst_size.unsqueeze(-1)
    return compressed_memory_bank, dst_size


def memory_bank_compress_MALLM_hard(memory_bank: torch.Tensor, sync: bool=False) -> tuple:
    """
    Compresses the memory bank if the current memory bank length is greater than the threshold.
    Different from MA-LLM, this method replace the tgt features with src ones
    
    Args:
        memory_bank (torch.Tensor): The input memory bank to compress. Shape: (B, T, N, C)
    
    Returns:
        compressed_memory_bank (torch.Tensor): The compressed memory bank. Shape: (B, T-1, N, C)
    """
    B, T, N, C = memory_bank.shape
    # Calculate the cosine similarity between adjacent frames
    similarity_matrix = F.cosine_similarity(memory_bank[:, :-1, :], memory_bank[:, 1:, :], dim=-1)
    if sync:
        similarity_matrix = similarity_matrix.mean(-1, keepdim=True).expand(-1, -1, N)
    # Select the frame indices with the top-1 similarity 
    _, max_similarity_indices = torch.max(similarity_matrix, dim=1, keepdim=True)

    # Calculate source and dst indices for compression
    src_indices = max_similarity_indices + 1
    dst_indices = torch.arange(T - 1).to(memory_bank.device)[None, :, None].repeat(B, 1, N)
    dst_indices[dst_indices > max_similarity_indices] += 1

    # Gather source and dst memory banks and sizes
    src_memory_bank = memory_bank.gather(dim=1, index=src_indices.unsqueeze(-1).expand(-1, -1, -1, C))
    dst_memory_bank = memory_bank.gather(dim=1, index=dst_indices.unsqueeze(-1).expand(-1, -1, -1, C))

    # Compress the memory bank by adding the source memory bank to the dst memory bank
    dst_memory_bank.scatter_(dim=1, index=max_similarity_indices.unsqueeze(-1).expand(-1, -1, -1, C), src=src_memory_bank)

    # Normalize the dst memory bank by its size
    compressed_memory_bank = dst_memory_bank
    return compressed_memory_bank


def memory_bank_compress_keyframe(memory_bank: torch.Tensor, tgt_mem_len: int, window_size: int=3, sync: bool=True) -> tuple:
    """
    Compresses the memory bank if the current memory bank length is greater than the threshold.
    Different from MA-LLM, this method replace the tgt features with src ones
    
    Args:
        memory_bank (torch.Tensor): The input memory bank to compress. Shape: (B, T, N, C)
    
    Returns:
        compressed_memory_bank (torch.Tensor): The compressed memory bank. Shape: (B, T-1, N, C)
        keypatches_mask (torch.Tensor): The compressed memory bank. Shape: (T-1 * N)
    """
    B, T, N, C = memory_bank.shape
    # Calculate the cosine similarity between adjacent frames
    similarity_matrix = F.cosine_similarity(memory_bank[:, :-1, :], memory_bank[:, 1:, :], dim=-1)
    dis_matrix = 1 - similarity_matrix[0].type(torch.float)
    # dis-similarity of the (i)th frame with the (i-1)th frame
    dis_matrix = torch.cat([
        torch.ones_like(dis_matrix[:1]),
        dis_matrix
    ], dim=0) # [T, N]

    if sync:
        # Meanpool over spatial locations
        dis_matrix = dis_matrix.mean(1) # [T]
        keypatches_mask = torch.zeros_like(dis_matrix).bool()

        # Argrelmax
        try:
            if torch.npu.is_available():
                # F.max_pool1d_with_indices is not supported and returns wrong tensor
                device = dis_matrix.device
                dis_matrix = dis_matrix.cpu()
        except:
            pass
        window_maxima = F.max_pool1d_with_indices(dis_matrix[None,None,:], window_size, 1, padding=window_size//2)[1].squeeze() # [T]
        candidates = window_maxima.unique()
        peaks = candidates[(window_maxima[candidates]==candidates).nonzero()].squeeze()
        try:
            if torch.npu.is_available():
                dis_matrix = dis_matrix.to(device)
                peaks = peaks.to(device)
        except:
            pass

        # Fill remaining frames
        keypatches_mask[peaks] = True
        dis_matrix[peaks] += 2 # select from peaks first
        peaks = torch.topk(dis_matrix, k=tgt_mem_len, sorted=False)[1] # [t]
        peaks = peaks.sort()[0]

        # Get keyframe memory
        compressed_memory_bank = memory_bank[:,peaks] # [B, t, N, C]
        keypatches_mask = keypatches_mask[peaks] # [t]
        keypatches_mask = keypatches_mask[:, None].repeat(1, N) # [t, N]
    else:
        dis_matrix = dis_matrix.transpose(0, 1) # [N, T]
        keypatches_mask = torch.zeros_like(dis_matrix).bool()
        # Argrelmax
        try:
            if torch.npu.is_available():
                # F.max_pool1d_with_indices is not supported and returns wrong tensor
                device = dis_matrix.device
                dis_matrix = dis_matrix.cpu()
                keypatches_mask = keypatches_mask.cpu()
        except:
            pass
        window_maxima = F.max_pool1d_with_indices(dis_matrix[:,None,:], window_size, 1, padding=window_size//2)[1].squeeze() # [N, T]
        for p, window_maxima_patch in enumerate(window_maxima):
            candidates_patch = window_maxima_patch.unique()
            peaks_patch = candidates_patch[(window_maxima_patch[candidates_patch]==candidates_patch).nonzero()][:,0]

            # Fill remaining frames
            keypatches_mask[p, peaks_patch] = True
            dis_matrix[p, peaks_patch] += 2
        try:
            if torch.npu.is_available():
                dis_matrix = dis_matrix.to(device)
                keypatches_mask = keypatches_mask.to(device)
        except:
            pass
        peaks = torch.topk(dis_matrix, k=tgt_mem_len, sorted=False, dim=1)[1] # [N, t]
        peaks = peaks.sort(dim=1)[0]
        peaks = peaks.transpose(0, 1) # [t, N]
        keypatches_mask = keypatches_mask.transpose(0, 1) # [t, N]

        # Get keyframe memory
        compressed_memory_bank = memory_bank.gather(dim=1, index=peaks[None,:,:,None].expand(-1, -1, -1, C))
        # [B, t, N, C]
        keypatches_mask = keypatches_mask.gather(dim=0, index=peaks) # [t, N]

    return compressed_memory_bank, keypatches_mask.flatten()

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section, unsqueeze_dim=1, reverse=False, attention_scaling=1):
    """Applies Rotary Position Embedding with Multimodal Sections to the query and key tensors (https://qwenlm.github.io/blog/qwen2-vl/).

    Explanation:
        Multimodal 3D rotary position embedding is an extension to 1D rotary position embedding. The input embedding
        sequence contains vision (images / videos) embedding and text embedding or just contains text embedding. For
        vision embedding part, we apply rotary position embedding on temporal, height and width dimension seperately.
        Here we split the channel dimension to 3 chunks for the temporal, height and width rotary position embedding.
        For text embedding part, we just apply 1D rotary position embedding. The three rotary position index (temporal,
        height and width) of text embedding is always the same, so the text embedding rotary position embedding has no
        difference with modern LLMs.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        mrope_section(`List(int)`):
            Multimodal rope section is for channel dimension of temporal, height and width in rope calculation.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    mrope_section = mrope_section * 2
    cos = torch.cat([m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
        unsqueeze_dim
    )
    sin = torch.cat([m[i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
        unsqueeze_dim
    )

    if reverse: # Rotate towards the opposite direction
        q_embed = ((q * cos) - (rotate_half(q) * sin)) / attention_scaling**2
        k_embed = ((k * cos) - (rotate_half(k) * sin)) / attention_scaling**2
    else:
        q_embed = (q * cos) + (rotate_half(q) * sin) if q is not None else None
        k_embed = (k * cos) + (rotate_half(k) * sin) if k is not None else None

    return q_embed, k_embed


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1, reverse=False, attention_scaling=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    if reverse: # Rotate towards the opposite direction
        q_embed = ((q * cos) - (rotate_half(q) * sin))  / attention_scaling**2
        k_embed = ((k * cos) - (rotate_half(k) * sin))  / attention_scaling**2
    else:
        q_embed = (q * cos) + (rotate_half(q) * sin) if q is not None else None
        k_embed = (k * cos) + (rotate_half(k) * sin) if k is not None else None

    return q_embed, k_embed


class PivotKVCache(DynamicCache):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        if hasattr(config, 'text_config'):
            # LLaVA-OneVision
            llm_config = config.text_config
        else:
            # QWen2VL
            llm_config = config
        self.hidden_size = llm_config.hidden_size
        self.num_hidden_layers = llm_config.num_hidden_layers
        self.num_heads = llm_config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = llm_config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        # Patch longvideo kwargs
        kv_compression_kwargs = config.longvideo_kwargs['kvcache_compression_kwargs']
        self.kvcache_compression = True
        self.kv_compression_kwargs = kv_compression_kwargs
        self.compression_ratio = kv_compression_kwargs['compression_ratio']
        self.compression_method = kv_compression_kwargs['compression_method']
        self.pos_embed_reforge = kv_compression_kwargs.get('pos_embed_reforge', False)
        self.position_cache: List[torch.Tensor] = []
        self.num_evicted_tokens: List[int] = []

    def before_forward(self, **kwargs):
        pass

    def after_forward(self, **kwargs):
        pass

    def update_position_ids(
        self,
        position_ids: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[torch.Tensor]:
        """
        Updates the cache with the new `position_ids` for the layer `layer_idx`.

        Parameters:
            position_ids (`torch.Tensor`):
                The new key states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.

        Return:
            A tuple containing the updated key and value states.
        """
        # Update the cache
        if len(self.position_cache) <= layer_idx:
            # There may be skipped layers, fill them with empty lists
            for _ in range(len(self.position_cache), layer_idx):
                self.position_cache.append([])
            self.position_cache.append(position_ids)
        elif len(self.position_cache[layer_idx]) == 0:  # fills previously skipped layers; checking for tensor causes errors
            self.position_cache[layer_idx] = position_ids
        else:
            self.position_cache[layer_idx] = torch.cat([self.position_cache[layer_idx], position_ids], dim=-1)

        return self.position_cache[layer_idx]

    def update_num_evicted_tokens(
        self,
        num_tokens: int,
        layer_idx: int,
    ) -> Tuple[torch.Tensor]:
        """
        Updates the `num_evicted_tokens` with an increment `num_tokens` at layer `layer_idx`.
        If `num_tokens` = 0, this function get the number of evicted tokens in layer `layer_idx`.

        Parameters:
            num_tokens (`int`):
                The number of evicted tokens.
            layer_idx (`int`):
                The index of the layer to cache the states for.

        Return:
            A tuple containing the updated key and value states.
        """
        # Update the cache
        if len(self.num_evicted_tokens) <= layer_idx:
            # There may be skipped layers without eviction, fill them with 0
            for _ in range(len(self.num_evicted_tokens), layer_idx):
                self.num_evicted_tokens.append(0)
            self.num_evicted_tokens.append(num_tokens)
        else:
            self.num_evicted_tokens[layer_idx] += num_tokens

        return self.num_evicted_tokens[layer_idx]

    def get_prev_temporal_idx(self, layer_idx: int) -> torch.LongTensor:
        if len(self.position_cache) <= layer_idx:
            return -1
        cache_layer = self.position_cache[layer_idx]
        return cache_layer[0,0,-1] if cache_layer.ndim == 3 else cache_layer[0,-1]

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
            Input
                query_states: [bsz, num_heads, q_len, d]
                key_states: [bsz, num_key_value_heads, q_len, d]
                position_ids: [3, bsz, q_len] / [bsz, q_len]
            Output
                key_states_output: for calculating self attention
                value_states_output: for calculating self attention
        """
        logger.warning_once("Enable PivotKVCache compression: length after compression %.2f" % (self.compression_ratio))

        position_ids = cache_kwargs.pop('position_ids', None)
        # 1) Hidden states for the next layer remains uncompressed in current chunked prefill iter
        key_states_output, value_states_output = super().update(key_states, value_states, layer_idx, cache_kwargs)

        if self.kvcache_compression: # when prefilling visual tokens
            query_states = cache_kwargs.pop('query_states')
            rotary_emb_fn = cache_kwargs.pop('rotary_emb')
            mrope_section = cache_kwargs.pop('mrope_section', None) # For MRope only
            bsz, num_heads, q_len, head_dim = query_states.shape
            num_key_value_heads, k_len = key_states.shape[1:3]
            assert bsz == 1
            assert q_len == k_len

            if self.pos_embed_reforge:
                cos, sin = rotary_emb_fn(value_states, position_ids)
                if mrope_section:
                    query_states, key_states = apply_multimodal_rotary_pos_emb(
                        query_states, key_states, cos, sin, mrope_section, 
                        reverse=True, attention_scaling=rotary_emb_fn.attention_scaling
                    )
                else:
                    query_states, key_states = apply_rotary_pos_emb(
                        query_states, key_states, cos, sin,
                        reverse=True, attention_scaling=rotary_emb_fn.attention_scaling
                    )
            key_states_repeated = repeat_kv(key_states, self.num_heads // self.num_key_value_heads)

            # 2) Evit KV Cache based on query_states
            keep_len = max(1, int(self.compression_ratio * q_len)) # Evict new tokens only
            attn_weights = torch.matmul(query_states, key_states_repeated.transpose(2, 3)) / math.sqrt(self.head_dim)
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
                query_states.dtype
            ).detach() # [bsz, self.num_heads, q_len, q_len(k)]
            attn_weights = attn_weights[0].sum(1) # [self.num_heads, q_len(k)]
            attn_weights = attn_weights.reshape(self.num_key_value_heads, -1, q_len).mean(1) # [num_key_value_heads, q_len(k)]
            attn_weights = attn_weights.mean(0) # [q_len(k)]

            if getattr(self, "keypatches_mask_chunk", None) is not None:
                keypatches_mask_chunk = self.keypatches_mask_chunk
                attn_weights.masked_fill_(keypatches_mask_chunk, 1.) # Select key patches first

            _, keep_indices = attn_weights.topk(keep_len)
            keep_indices = keep_indices.sort().values # [keep_len]
            keep_indices_kv = keep_indices[None,None,:,None].repeat(bsz, self.num_key_value_heads, 1, self.head_dim) # [bsz, num_key_value_heads, keep_len, head_dim]
            compressed_key_states = torch.gather(input=key_states, dim=2, index=keep_indices_kv)
            compressed_value_states = torch.gather(input=value_states, dim=2, index=keep_indices_kv) # [bsz, num_k_heads, keep_len, head_dim]

            # Calculate new postional ids
            if mrope_section:
                keep_indices_ids = keep_indices[None,None,:].repeat(3, bsz, 1) # [3, bsz, keep_len]
                compressed_position_ids = torch.gather(input=position_ids, dim=2, index=keep_indices_ids) # [3, bsz, keep_len]
            else:
                keep_indices_ids = keep_indices[None,:].repeat(bsz, 1) # [bsz, keep_len]
                compressed_position_ids = torch.gather(input=position_ids, dim=1, index=keep_indices_ids) # [bsz, keep_len]

            if self.pos_embed_reforge:
                assert bsz == 1
                min_temp_id = compressed_position_ids[0].min()
                comp_ratio = keep_len / k_len # NOTE: avoid truncating issues when calculating keep_len
                compressed_position_ids[0] = min_temp_id + ((compressed_position_ids[0] - min_temp_id) * comp_ratio).long()

                # Add new rotary embedding
                cos, sin = rotary_emb_fn(compressed_value_states, compressed_position_ids)
                if mrope_section:
                    _, compressed_key_states = apply_multimodal_rotary_pos_emb(
                        None, compressed_key_states, cos, sin, mrope_section
                    )
                else:
                    _, compressed_key_states = apply_rotary_pos_emb(
                        None, compressed_key_states, cos, sin,
                    )

            if self.pos_embed_reforge:
                _ = self.update_position_ids(compressed_position_ids, layer_idx)
            _ = self.update_num_evicted_tokens(k_len - keep_len, layer_idx)

            # 3) Update KVCache
            self.key_cache[layer_idx] = torch.cat([
                key_states_output[...,:-q_len,:], compressed_key_states
            ], dim=2)
            self.value_cache[layer_idx] = torch.cat([
                value_states_output[...,:-q_len,:], compressed_value_states
            ], dim=2)
        else: # when prefilling textual tokens / decoding / kvcache compression disabled
            if self.pos_embed_reforge:
                _ = self.update_position_ids(position_ids, layer_idx)

        return key_states_output, value_states_output


class VidLangKVCache(PivotKVCache):
    def __init__(self, config) -> None:
        super().__init__(config)
        # For KV cache compression
        self.prompt_guided_compression = self.kv_compression_kwargs.get('prompt_guided_compression', False)
        self.prompt_compression = self.kv_compression_kwargs.get('prompt_compression', False)
        assert self.prompt_guided_compression
        # For KV cache budget allocaation
        self.budget_allocation_method = self.kv_compression_kwargs.get('budget_allocation_method', 'even')

    def before_forward(self, prompt_length, **kwargs):
        self.prompt_length = prompt_length

    def after_forward(self, **kwargs):
        self.prompt_length = None # Turned off by default

    def compress_prompt(self, query_states, key_states_repeated, q_len, num_key_value_heads, head_dim):
        # Same with text raters in SparseVLMs
        attn_weights = torch.matmul(query_states, key_states_repeated.transpose(2, 3)) / math.sqrt(head_dim)
        attn_weights = nn.functional.softmax(attn_weights, dim=2, dtype=torch.float32).to(
            query_states.dtype
        ).detach() # [bsz, self.num_heads, q_len, k_len]
        attn_weights = attn_weights[0].sum(2) # [self.num_heads, q_len]
        attn_weights = attn_weights.reshape(num_key_value_heads, -1, q_len).mean(1) # [num_key_value_heads, q_len]
        attn_weights = attn_weights.mean(0) # [q_len]
        t_token_idx = torch.where(attn_weights > attn_weights.mean())[0]  # [q_len']
        query_states = query_states[:,:,t_token_idx]
        # [bsz, self.num_heads, q_len', head_dim]
        return query_states

    def budget_allocation(self, layer_idx):
        # No compression
        if not self.kvcache_compression or self.compression_ratio == 1.0:
            return 1.0

        if self.budget_allocation_method.lower() == 'even':
            compression_ratio = self.compression_ratio
        elif self.budget_allocation_method.lower() == 'pyramid':
            pyramid_beta = self.kv_compression_kwargs.get('pyramid_beta')
            min_comp_ratio = self.compression_ratio / pyramid_beta
            max_comp_ratio = 2 * self.compression_ratio
            comp_ratio = (max_comp_ratio - min_comp_ratio) - (max_comp_ratio - 2 * min_comp_ratio) / (self.num_hidden_layers - 1) * layer_idx
            compression_ratio = min(1.0, comp_ratio)
        elif self.budget_allocation_method.lower() == 'emprical':
            if layer_idx < 2 * self.num_hidden_layers / 3:
                compression_ratio = - ((0.3 - 0.1) * 3 /self.num_hidden_layers / 2) * layer_idx + 0.3
            else:
                compression_ratio = (0.2 - 0.1) / (self.num_hidden_layers / 3 - 1) * layer_idx + 0.2 - (self.num_hidden_layers - 1) * (0.2 - 0.1) / (self.num_hidden_layers / 3 - 1)
            # scaling
            scale_ratio = self.compression_ratio / (0.55/3)
            compression_ratio = scale_ratio * compression_ratio
            compression_ratio = min(1, compression_ratio)
            compression_ratio = max(0.01, compression_ratio)
        else:
            raise NotImplementedError

        return compression_ratio

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logger.warning_once("Enable VidLangKVCache compression: length after compression %.2f" % (self.compression_ratio))

        position_ids = cache_kwargs.pop('position_ids', None)
        compression_ratio = self.budget_allocation(layer_idx)
        # print('compression_ratio of layer %d: %.4f' % (layer_idx, compression_ratio))

        if self.kvcache_compression and self.compression_ratio < 1.0 and compression_ratio == 1.0:
            # Truncate the prompts directly when no compression
            key_states = key_states[:,:,:-self.prompt_length]
            value_states = value_states[:,:,:-self.prompt_length]
            position_ids = position_ids[...,:-self.prompt_length]

        # 1) Hidden states for the next layer remains uncompressed in current chunked prefill iter
        key_states_output, value_states_output = super(PivotKVCache, self).update(key_states, value_states, layer_idx, cache_kwargs)

        if self.kvcache_compression and compression_ratio < 1.0: # when compression is enabled
            query_states = cache_kwargs.pop('query_states')
            rotary_emb_fn = cache_kwargs.pop('rotary_emb')
            mrope_section = cache_kwargs.pop('mrope_section', None)
            if self.pos_embed_reforge:
                cos, sin = rotary_emb_fn(value_states, position_ids)
                if mrope_section:
                    query_states, key_states = apply_multimodal_rotary_pos_emb(
                        query_states, key_states, cos, sin, mrope_section, 
                        reverse=True, attention_scaling=rotary_emb_fn.attention_scaling
                    )
                else:
                    query_states, key_states = apply_rotary_pos_emb(
                        query_states, key_states, cos, sin,
                        reverse=True, attention_scaling=rotary_emb_fn.attention_scaling
                    )
            query_states = query_states[:,:,-self.prompt_length:]
            key_states = key_states[:,:,:-self.prompt_length]
            value_states = value_states[:,:,:-self.prompt_length]
            position_ids_key = position_ids[...,:-self.prompt_length]

            bsz, num_heads, q_len, head_dim = query_states.shape
            num_key_value_heads, k_len = key_states.shape[1:3]
            ori_cache_len = q_len + k_len
            assert bsz == 1

            key_states_repeated = repeat_kv(key_states, num_heads // num_key_value_heads)

            # 2) Evit KV Cache based on query_states
            if self.prompt_compression:
                query_states = self.compress_prompt(query_states, key_states_repeated, q_len, num_key_value_heads, head_dim)
                q_len = query_states.shape[2]
                # [bsz, self.num_heads, q_len', head_dim]

            keep_len = max(1, int(compression_ratio * k_len)) # Evict new tokens only
            attn_weights = torch.matmul(query_states, key_states_repeated.transpose(2, 3)) / math.sqrt(head_dim)
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
                query_states.dtype
            ).detach() # [bsz, self.num_heads, q_len, k_len]
            attn_weights = attn_weights[0].sum(1) # [self.num_heads, k_len]
            attn_weights = attn_weights.reshape(num_key_value_heads, -1, k_len).mean(1) # [num_key_value_heads, k_len]
            attn_weights = attn_weights.mean(0) # [k_len]
            # attn_weights = attn_weights.max(0).values # [k_len]

            _, keep_indices = attn_weights.topk(keep_len)
            keep_indices = keep_indices.sort().values # [keep_len]
            keep_indices_kv = keep_indices[None,None,:,None].repeat(bsz, num_key_value_heads, 1, head_dim) # [bsz, num_key_value_heads, keep_len, head_dim]
            compressed_key_states = torch.gather(input=key_states, dim=2, index=keep_indices_kv)
            compressed_value_states = torch.gather(input=value_states, dim=2, index=keep_indices_kv) # [bsz, num_k_heads, keep_len, head_dim]

            # Calculate new postional ids
            if mrope_section:
                keep_indices_ids = keep_indices[None,None,:].repeat(3, bsz, 1) # [3, bsz, keep_len]
                compressed_position_ids = torch.gather(input=position_ids, dim=2, index=keep_indices_ids) # [3, bsz, keep_len]
            else:
                keep_indices_ids = keep_indices[None,:].repeat(bsz, 1) # [bsz, keep_len]
                compressed_position_ids = torch.gather(input=position_ids, dim=1, index=keep_indices_ids) # [bsz, keep_len]

            if self.pos_embed_reforge:
                assert bsz == 1

                # # NOTE: type 1
                # # Get the unique elements and their corresponding re-indexed values
                # new_temporal_index = torch.unique(compressed_position_ids[0], return_inverse=True)[1]
                # compressed_position_ids[0] = self.get_prev_temporal_idx(layer_idx) + 1 + new_temporal_index

                # NOTE: type 2
                min_temp_id = compressed_position_ids[0].min()
                comp_ratio = keep_len / k_len # NOTE: avoid truncating issues when calculating keep_len
                compressed_position_ids[0] = min_temp_id + ((compressed_position_ids[0] - min_temp_id) * comp_ratio).long()

                # Add new rotary embedding
                cos, sin = rotary_emb_fn(compressed_value_states, compressed_position_ids)
                if mrope_section:
                    _, compressed_key_states = apply_multimodal_rotary_pos_emb(
                        None, compressed_key_states, cos, sin, mrope_section
                    )
                else:
                    _, compressed_key_states = apply_rotary_pos_emb(
                        None, compressed_key_states, cos, sin,
                    )

            _ = self.update_position_ids(compressed_position_ids, layer_idx)
            _ = self.update_num_evicted_tokens(k_len - keep_len, layer_idx)

            # 3) Update KVCache
            self.key_cache[layer_idx] = torch.cat([
                key_states_output[...,:-ori_cache_len,:], compressed_key_states
            ], dim=2)
            self.value_cache[layer_idx] = torch.cat([
                value_states_output[...,:-ori_cache_len,:], compressed_value_states
            ], dim=2)
        else: # when prefilling textual tokens or decoding / kvcache compression disabled
            _ = self.update_position_ids(position_ids, layer_idx)

        return key_states_output, value_states_output


class StandardVidLangKVCache(VidLangKVCache):
    """Standard Implementation of VidLangKVCache.
    It perform KV cache compression after the prefill phase of each layer is finished.
    """
    def __init__(self, config) -> None:
        super().__init__(config)
        self.attn_cumscores_cache: List[torch.Tensor] = []
        # self.rope_kwargs_list: List[Tuple] = []
        self.enable_temporal_adaptation = self.kv_compression_kwargs.get('enable_temporal_adaptation', False)
        if self.enable_temporal_adaptation:
            self.temporal_adaptation_ratio = self.kv_compression_kwargs.get('temporal_adaptation_ratio', 10.0)

    def update_attn_cumscores(
        self,
        attn_cumscores: torch.Tensor,
        layer_idx: int,
    ):
        """
        Updates the cache with the new `attn_cumscores` for the layer `layer_idx`.

        Parameters:
            attn_cumscores (`torch.Tensor`):
                The new attention weights to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
        """
        self.attn_cumscores_cache.append(attn_cumscores)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logger.warning_once("Enable StandardVidLangKVCache compression: length after compression %.2f" % (self.compression_ratio))

        position_ids = cache_kwargs.pop('position_ids', None)
        # 1) Hidden states for the next layer remains uncompressed in current chunked prefill iter
        key_states_output, value_states_output = super(PivotKVCache, self).update(key_states, value_states, layer_idx, cache_kwargs)

        if self.kvcache_compression and self.compression_ratio < 1.0: # when kvcache compression is enabled
            query_states = cache_kwargs.pop('query_states')
            rotary_emb_fn = cache_kwargs.pop('rotary_emb')
            mrope_section = cache_kwargs.pop('mrope_section', None)
            if self.pos_embed_reforge:
                cos, sin = rotary_emb_fn(value_states, position_ids)
                if mrope_section:
                    query_states, key_states = apply_multimodal_rotary_pos_emb(
                        query_states, key_states, cos, sin, mrope_section, 
                        reverse=True, attention_scaling=rotary_emb_fn.attention_scaling
                    )
                else:
                    query_states, key_states = apply_rotary_pos_emb(
                        query_states, key_states, cos, sin,
                        reverse=True, attention_scaling=rotary_emb_fn.attention_scaling
                    )
                # NOTE: clone to avoid in-place ops, since latter layers will use it
                position_ids = position_ids.clone()
                min_temp_id = position_ids[0].min()
                position_ids[0] = min_temp_id + ((position_ids[0] - min_temp_id) * self.compression_ratio).long()
            query_states = query_states[:,:,-self.prompt_length:]
            key_states = key_states[:,:,:-self.prompt_length]
            value_states = value_states[:,:,:-self.prompt_length]
            position_ids_key = position_ids[...,:-self.prompt_length]

            bsz, num_heads, q_len, head_dim = query_states.shape
            num_key_value_heads, k_len = key_states.shape[1:3]
            ori_cache_len = q_len + k_len
            assert bsz == 1

            key_states_repeated = repeat_kv(key_states, num_heads // num_key_value_heads)

            # 2) Evit KV Cache based on query_states
            attn_weights = torch.matmul(query_states, key_states_repeated.transpose(2, 3)) / math.sqrt(head_dim)
            attn_scores = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
                query_states.dtype
            ).detach() # [bsz, self.num_heads, q_len, k_len]
            attn_cumscores = attn_scores[0].sum(1) # [self.num_heads, k_len]
            attn_cumscores = attn_cumscores.reshape(num_key_value_heads, -1, k_len).mean(1) # [num_key_value_heads, k_len]
            attn_cumscores = attn_cumscores.mean(0) # [k_len]
            # attn_cumscores = attn_cumscores.max(0).values # [k_len]

            # 3) Send to list
            if self.pos_embed_reforge:
                # Add new rotary embedding
                cos, sin = rotary_emb_fn(value_states, position_ids_key)
                if mrope_section:
                    _, key_states = apply_multimodal_rotary_pos_emb(
                        None, key_states, cos, sin, mrope_section
                    )
                else:
                    _, key_states = apply_rotary_pos_emb(
                        None, key_states, cos, sin,
                    )

            self.update_attn_cumscores(attn_cumscores, layer_idx)
            # self.rope_kwargs_list.append((rotary_emb_fn, mrope_section))

            self.key_cache[layer_idx] = torch.cat([
                key_states_output[...,:-ori_cache_len,:], key_states
            ], dim=2)
            self.value_cache[layer_idx] = torch.cat([
                value_states_output[...,:-ori_cache_len,:], value_states
            ], dim=2)
            self.position_cache[layer_idx] = torch.cat([
                self.position_cache[layer_idx], position_ids_key
            ], dim=-1)
        else: # when prefilling textual tokens or decoding / kvcache compression disabled
            assert getattr(self, 'prompt_length', None) is None
            # Make sure text chunks are kept
            attn_cumscores = 1000. * torch.ones_like(position_ids)
            attn_cumscores = attn_cumscores[0, 0, :] if attn_cumscores.ndim == 3 else attn_cumscores[0, :]
            self.update_attn_cumscores(attn_cumscores, layer_idx)
            _ = self.update_position_ids(position_ids, layer_idx)

        return key_states_output, value_states_output

    def budget_allocation(self):
        if self.budget_allocation_method.lower() == 'even':
            compression_ratio_layers = self.compression_ratio * torch.ones(self.num_hidden_layers)
        elif self.budget_allocation_method.lower() == 'adakv':
            k_len = self.attn_cumscores_cache[0].shape[0]
            if self.attn_cumscores_cache[0].device != self.attn_cumscores_cache[-1].device:
                attn_cumscores_cache = [
                    attn_cache.cpu() for attn_cache in self.attn_cumscores_cache
                ]
            else:
                attn_cumscores_cache = self.attn_cumscores_cache
            attn_cumscores_layers = torch.cat(attn_cumscores_cache) # [num_layers * k_len]
            cache_bugdet = int(max(1, self.compression_ratio * k_len) * self.num_hidden_layers)
            _, keep_indices = attn_cumscores_layers.topk(cache_bugdet)

            compression_ratio_layers = torch.ones(self.num_hidden_layers)
            for keep_index in keep_indices.tolist():
                layer_idx = keep_index // k_len
                compression_ratio_layers[layer_idx] += 1
            compression_ratio_layers = compression_ratio_layers / compression_ratio_layers.sum()
            compression_ratio_layers = (self.num_hidden_layers * self.compression_ratio) * compression_ratio_layers

        return compression_ratio_layers.tolist()

    def after_forward(self, **kwargs):
        if self.kvcache_compression and self.compression_ratio < 1.0: # when compression is enabled
            compression_ratio_layers = self.budget_allocation()
            # print("AdaVidLangKVCache.after_forward(): compression_ratio_layers", compression_ratio_layers)

            bsz = 1
            for layer_idx, compression_ratio in enumerate(compression_ratio_layers):
                attn_cumscores = self.attn_cumscores_cache[layer_idx]
                k_len = attn_cumscores.shape[0]
                key_states = self.key_cache[layer_idx][:,:,-k_len:]
                value_states = self.value_cache[layer_idx][:,:,-k_len:]
                position_ids_key = self.position_cache[layer_idx][...,-k_len:]
                # rotary_emb_fn, mrope_section = self.rope_kwargs_list[layer_idx]

                # comp_ratio_ori = compression_ratio
                if self.enable_temporal_adaptation:
                    ratio = torch.where(attn_cumscores > 0.01 * attn_cumscores.max())[0].shape[0] / attn_cumscores.shape[0]
                    ratio = math.sqrt(self.temporal_adaptation_ratio * ratio)
                    ratio = min(2, max(1/2, ratio))
                    compression_ratio = min(1, ratio * compression_ratio)
                # print('global comp ratio: %.4f, layer comp_ratio: %.4f, chunk comp_ratio %.4f' % (self.compression_ratio, comp_ratio_ori, compression_ratio))
                keep_len = max(1, int(max(0.01, compression_ratio) * k_len)) # Evict new tokens only
                # if keep_len == 1:
                #     # print("AdaVidLangKVCache.after_forward(): Got ill compression_ratio! compression_ratio_layers", compression_ratio_layers)
                #     print("AdaVidLangKVCache.after_forward(): Got ill compression_ratio!")
                _, keep_indices = attn_cumscores.topk(keep_len)
                keep_indices = keep_indices.sort().values # [keep_len]
                keep_indices_kv = keep_indices[None,None,:,None].repeat(bsz, self.num_key_value_heads, 1, self.head_dim) # [bsz, num_key_value_heads, keep_len, head_dim]
                compressed_key_states = torch.gather(input=key_states, dim=2, index=keep_indices_kv)
                compressed_value_states = torch.gather(input=value_states, dim=2, index=keep_indices_kv) # [bsz, num_k_heads, keep_len, head_dim]

                # Calculate new postional ids
                if position_ids_key.ndim == 3:
                    keep_indices_ids = keep_indices[None,None,:].repeat(3, bsz, 1) # [3, bsz, keep_len]
                    compressed_position_ids = torch.gather(input=position_ids_key, dim=2, index=keep_indices_ids) # [3, bsz, keep_len]
                else:
                    keep_indices_ids = keep_indices[None,:].repeat(bsz, 1) # [bsz, keep_len]
                    compressed_position_ids = torch.gather(input=position_ids_key, dim=1, index=keep_indices_ids) # [bsz, keep_len]

                _ = self.update_num_evicted_tokens(k_len - keep_len, layer_idx)

                # 4) Update KVCache
                self.key_cache[layer_idx] = torch.cat([
                    self.key_cache[layer_idx][:,:,:-k_len], compressed_key_states
                ], dim=2)
                self.value_cache[layer_idx] = torch.cat([
                    self.value_cache[layer_idx][:,:,:-k_len], compressed_value_states
                ], dim=2)
                self.position_cache[layer_idx] = torch.cat([
                    self.position_cache[layer_idx][...,:-k_len], compressed_position_ids
                ], dim=-1)

        self.prompt_length = None # Turned off by default
        self.attn_cumscores_cache.clear()
        # self.rope_kwargs_list.clear()


def build_kvcache(config):
    if getattr(config, "longvideo_kwargs", None) is None or not config.longvideo_kwargs.get('kvcache_compression', False):
        return DynamicCache()
    else:
        compression_method = config.longvideo_kwargs['kvcache_compression_kwargs']['compression_method']
        if compression_method.lower() == 'pivotkv':
            return PivotKVCache(config)
        elif compression_method.lower() == 'vidlkv':
            return VidLangKVCache(config)
        elif compression_method.lower().replace('_', '') == 'stdvidlkv':
            return StandardVidLangKVCache(config)
        else:
            raise NotImplementedError

###############

Qwen2Attention_original_init = Qwen2Attention.__init__
def retake_Qwen2Attention_init(
    self, 
    config: Qwen2VLConfig, 
    layer_idx: Optional[int] = None
):
    Qwen2Attention_original_init(self, config, layer_idx)
    self.rotary_emb = Qwen2RotaryEmbedding(config=self.config)

def retake_Qwen2Attention_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_value: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    # Update position_ids if positional embeddings are reforged
    position_ids = None
    if past_key_value is not None and getattr(past_key_value, "pos_embed_reforge", False):
        # This code reforge the `position_ids` of current chunk, 
        # the `position_ids` of previous chunks are reforged in KVCache.update()
        position_ids = kwargs.get('position_ids')
        prev_tempo_idx = past_key_value.get_prev_temporal_idx(self.layer_idx)
        cur_tempo_idx = position_ids[0,0]
        if prev_tempo_idx + 1 != cur_tempo_idx:
            # print("Warning! Discontinuous positional ids %d (prev) + 1 != %d (cur) at layer %d. Fixed!" % (prev_tempo_idx,  cur_tempo_idx, self.layer_idx))
            # NOTE: clone `position_ids` to avoid influence of in-place ops in different layers
            position_ids = position_ids.clone()
            position_ids[0,:] += prev_tempo_idx + 1 - cur_tempo_idx
        position_embeddings = None # `position_embeddings` need to be re-calculated

    if position_embeddings is None:
        logger.warning_once(
            "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
            "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
            "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
            "removed and `position_embeddings` will be mandatory."
        )
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        # Specific to KVCache compression methods
        cache_kwargs.update({"query_states": query_states, "position_ids": position_ids, "rotary_emb": self.rotary_emb})
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    sliding_window = None
    if (
        self.config.use_sliding_window
        and getattr(self.config, "sliding_window", None) is not None
        and self.layer_idx >= self.config.max_window_layers
    ):
        sliding_window = self.config.sliding_window

    attention_interface: Callable = eager_attention_forward
    if self.config._attn_implementation != "eager":
        if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
            logger.warning_once(
                "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
        else:
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        sliding_window=sliding_window,  # main diff with Llama
        **kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights

def retake_LlavaOnevisionForConditionalGeneration_get_chunk_size(
    self, 
    config, 
    pixel_values_videos
) -> int:
    # Calculate the number of tokens in each prefill chunk
    chunk_frames = (
        config.longvideo_kwargs.get('chunked_prefill_frames', None) if getattr(config, 'longvideo_kwargs', None) 
        else None
    )
    if chunk_frames is None:
        chunk_prefill_size = None
    else:
        T, _, H, W = pixel_values_videos[0].shape
        H = math.ceil(H // self.config.vision_config.patch_size / self.pool_stride)
        W = math.ceil(W // self.config.vision_config.patch_size / self.pool_stride)
        chunk_prefill_size = min(chunk_frames, T) * H * W
    return chunk_prefill_size

def retake_LlavaOnevisionForConditionalGeneration_segment_input_ids(
    self,
    input_ids
):
    """Split video and text segments in the input_ids
    return: list[(s, e, type)], indices of [s, e) are of `type`.
    """
    videomask = (input_ids[0] == self.config.video_token_index)
    # Find the difference between consecutive elements
    diff = torch.diff(videomask.long())
    diff_pos_indices = (torch.where(diff == 1)[0] + 1).cpu().numpy()
    diff_neg_indices = (torch.where(diff == -1)[0] + 1).cpu().numpy()

    # True mask
    start_indices_true = diff_pos_indices
    end_indices_true = diff_neg_indices
    if videomask[0] == True: # segment starts at the beginning
        start_indices_true = np.insert(start_indices_true, 0, 0)
    if videomask[-1] == True: # segment ends at the beginning
        end_indices_true = np.append(end_indices_true, len(videomask))

    # False mask
    start_indices_flase = diff_neg_indices
    end_indices_flase = diff_pos_indices
    if videomask[0] == False:
        start_indices_flase = np.insert(start_indices_flase, 0, 0)
    if videomask[-1] == False:
        end_indices_flase = np.append(end_indices_flase, len(videomask))

    segments = (
        list(zip(start_indices_true, end_indices_true, ['video']*len(end_indices_true))) + 
        list(zip(start_indices_flase, end_indices_flase, ['text']*len(end_indices_flase)))
    )
    segments = sorted(segments, key=lambda x: x[0])
    return segments

def retake_LlavaOnevisionForConditionalGeneration_compress_video_tokens(
    self, 
    input_ids: torch.LongTensor = None,
    attention_mask: torch.Tensor = None,
    selected_video_feature: torch.Tensor = None,
    position_ids: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    labels: Optional[torch.LongTensor] = None,
) -> Tuple[torch.LongTensor, torch.Tensor, torch.Tensor, torch.LongTensor]:
    # Parse long video compression configs
    visual_compression = False
    if getattr(self.config, "longvideo_kwargs", None) is not None:
        visual_compression = self.config.longvideo_kwargs.get("visual_compression", False)
        if visual_compression:
            compression_kwargs = self.config.longvideo_kwargs['visual_compression_kwargs']
            compression_ratio = compression_kwargs.get("compression_ratio")
            compression_method = compression_kwargs.get("compression_method")
            patch_sync = compression_kwargs.get("patch_sync")
            return_keyframe_mask = compression_kwargs.get("return_keyframe_mask")

    grid_t, grid_hw = selected_video_feature.shape[:2]
    tgt_grid_t = grid_t
    if visual_compression:
        assert labels is None
        assert input_ids.shape[0] == 1, "Currently, only inference are supported"
        video_token_indices = torch.where(input_ids[0] == self.config.video_token_index)[0]
        s_index, e_index = video_token_indices[0], video_token_indices[-1]
        height = width = self.config.vision_config.image_size // self.config.vision_config.patch_size
        grid_hw_after_pool = math.ceil(height / self.pool_stride) * math.ceil(width / self.pool_stride)
        ori_seq_len = input_ids.shape[1]
        tgt_grid_t = max(1, round(compression_ratio * grid_t))

        # Compress
        compressed_memory_bank = selected_video_feature.reshape(1, grid_t, grid_hw, -1)
        if compression_method == "MA-LLM":
            compression_size = torch.ones_like(compressed_memory_bank[:,:,:,0])
            while compressed_memory_bank.shape[1] > tgt_grid_t:
                compressed_memory_bank, compression_size = memory_bank_compress_MALLM(compressed_memory_bank, compression_size, sync=patch_sync)
            keypatches_mask = None
        elif compression_method == "MA-LLM-hard":
            while compressed_memory_bank.shape[1] > tgt_grid_t:
                compressed_memory_bank = memory_bank_compress_MALLM_hard(compressed_memory_bank, sync=patch_sync)
            keypatches_mask = None
        elif compression_method == "Keyframe":
            compressed_memory_bank, keypatches_mask = memory_bank_compress_keyframe(compressed_memory_bank, tgt_grid_t, 3, sync=patch_sync)
            keypatches_mask = keypatches_mask if return_keyframe_mask else None
        else:
            raise NotImplementedError
        selected_video_feature = compressed_memory_bank[0]
        mem_len_after = tgt_grid_t * grid_hw_after_pool

        # Reforge the input
        input_ids = torch.cat([
            input_ids[:, :s_index],
            input_ids[:, s_index:e_index+1][:,:mem_len_after],
            input_ids[:, e_index+1:]
            ],
            dim=1)
        num_token_diff = ori_seq_len - input_ids.shape[1]
        if num_token_diff and attention_mask is not None:
            attention_mask = attention_mask[:, num_token_diff:]
        if num_token_diff and position_ids is not None:
            position_ids = position_ids[:,:-num_token_diff]
        if num_token_diff and cache_position is not None:
            cache_position = cache_position[:-num_token_diff]
    else:
        keypatches_mask = None

    return input_ids, attention_mask, selected_video_feature, position_ids, cache_position, tgt_grid_t, keypatches_mask

def retake_LlavaOnevisionForConditionalGeneration_forge_input_chunks(
    self, 
    ss, 
    ee, 
    modality_segments, 
    position_ids, 
    cache_position, 
    attention_mask, 
    past_key_values, 
    inputs_embeds
):
    position_ids_chunk = position_ids[:,ss:ee]
    cache_position_chunk = cache_position[:ee]
    attention_mask_chunk = attention_mask[:,:ee] # NOTE: specially from 0 to ee
    inputs_embeds_chunk = inputs_embeds[:,ss:ee]
    prompt_length = None

    if getattr(self.config, 'longvideo_kwargs', None) and self.config.longvideo_kwargs.get('kvcache_compression', False):
        compression_kwargs = self.config.longvideo_kwargs['kvcache_compression_kwargs']
        if compression_kwargs.get('prompt_guided_compression', False) and compression_kwargs.get('compression_ratio', 1) < 1.0:
            # Prompt guided KV cache compression
            s_p, e_p, t_p = modality_segments[-1]
            max_guide_length = min(e_p - s_p, compression_kwargs.get('max_guide_length', 999999999999999999))
            s_p = s_p + (e_p - s_p - max_guide_length)
            assert t_p == 'text'
            pos_offset = position_ids[0,s_p] - position_ids_chunk[0,-1] - 1 # (bs, seq_len)
            # print('forge_input_chunks() temporal position_ids of prompts', position_ids[0,s_p:e_p])
            position_ids_chunk = torch.cat([position_ids_chunk, position_ids[:,s_p:e_p] - pos_offset], dim=1)
            cache_position_chunk = torch.cat([cache_position_chunk, cache_position[s_p:e_p] - pos_offset], dim=0)
            attention_mask_chunk = torch.cat([attention_mask_chunk, attention_mask[:,s_p:e_p]], dim=1)
            inputs_embeds_chunk = torch.cat([inputs_embeds_chunk, inputs_embeds[:,s_p:e_p]], dim=1)
            prompt_length = e_p - s_p

    return position_ids_chunk, cache_position_chunk, attention_mask_chunk, inputs_embeds_chunk, prompt_length

def retake_LlavaOnevisionForConditionalGeneration_forward(
    self,
    input_ids: torch.LongTensor = None,
    pixel_values: torch.FloatTensor = None,
    image_sizes: Optional[torch.LongTensor] = None,
    pixel_values_videos: torch.FloatTensor = None,
    image_sizes_videos: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    vision_feature_layer: Optional[int] = None,
    vision_feature_select_strategy: Optional[str] = None,
    vision_aspect_ratio: Optional[str] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    logits_to_keep: Union[int, torch.Tensor] = 0,
) -> Union[Tuple, LlavaOnevisionCausalLMOutputWithPast]:
    assert input_ids.shape[0] == 1, "Batch inference of long video is not supported yet!"

    self.pool_stride = 2
    if input_ids.shape[1] > 1: # Prefill
        is_prefill = True
        # Calculate chunk size based on inputs
        chunk_size = self.get_chunk_size(self.config, pixel_values_videos)
        # Configuring KV Cache compression kwargs
        if getattr(self.config, 'longvideo_kwargs', None) and self.config.longvideo_kwargs.get('kvcache_compression', False):
            compression_kwargs = self.config.longvideo_kwargs['kvcache_compression_kwargs']
            if compression_kwargs.get('dynamic_compression_ratio', False):
                # Dynamic compression ratio
                input_length = input_ids.shape[1]
                max_input_length = compression_kwargs['max_input_length']
                if input_length <= max_input_length:
                    compression_kwargs['compression_ratio'] = 1
                else:
                    compression_kwargs['compression_ratio'] = max_input_length / input_length
        if chunk_size is not None:
            modality_segments = self.segment_input_ids(input_ids)
            # print("modality_segments", modality_segments)
            past_key_values = build_kvcache(self.config)
            use_cache = True
    else:
        is_prefill = False
        chunk_size = None

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    vision_feature_layer = (
        vision_feature_layer if vision_feature_layer is not None else self.config.vision_feature_layer
    )
    vision_feature_select_strategy = (
        vision_feature_select_strategy
        if vision_feature_select_strategy is not None
        else self.config.vision_feature_select_strategy
    )
    vision_aspect_ratio = (
        vision_aspect_ratio if vision_aspect_ratio is not None else self.config.vision_aspect_ratio
    )

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError(
            "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
        )

    if (pixel_values is not None or pixel_values_videos is not None) and inputs_embeds is not None:
        raise ValueError(
            "You cannot specify both pixel_values/pixel_values_videos and inputs_embeds at the same time, and must specify either one"
        )

    # Images are processed with Anyres
    if pixel_values is not None:
        image_num_patches = [
            image_size_to_num_patches(
                image_size=imsize,
                grid_pinpoints=self.config.image_grid_pinpoints,
                patch_size=self.config.vision_config.image_size,
            )
            for imsize in image_sizes
        ]

        # unpad extra patches and concatenate them
        if pixel_values.dim() == 5:
            _pixel_values_list = [
                pix_val[:num_patch] for pix_val, num_patch in zip(pixel_values, image_num_patches)
            ]
            # [batch_size*frames*num_patches, num_channels, height, width] where frames=1 for images
            pixel_values = torch.cat(_pixel_values_list, dim=0)
        elif pixel_values.dim() != 4:
            raise ValueError(f"pixel_values of shape {pixel_values.shape}, expect to be of 4 or 5 dimensions")

        image_features = self.vision_tower(pixel_values, output_hidden_states=True)
        selected_image_feature = image_features.hidden_states[vision_feature_layer]

        if vision_feature_select_strategy == "default":
            selected_image_feature = selected_image_feature[:, 1:]
        elif vision_feature_select_strategy == "full":
            selected_image_feature = selected_image_feature
        image_features = self.multi_modal_projector(selected_image_feature)

        image_features = torch.split(image_features, image_num_patches, dim=0)
        image_features, feature_lens = self.pack_image_features(
            image_features,
            image_sizes,
            image_newline=self.image_newline,
            vision_aspect_ratio=vision_aspect_ratio,
        )

    # Video are simply embedded and further pooled to decrease seq len
    if pixel_values_videos is not None:
        batch_size, frames, channels, height, width = pixel_values_videos.shape
        pixel_values_videos = pixel_values_videos.view(batch_size * frames, channels, height, width)
        # NOTE: Split video into chunks to avoid OOM due to large activations during visual forward
        # chunk_size can be up to 128 or higher if you have flash attention
        frame_chunk_size = getattr(self.config, 'longvideo_kwargs', {}).get('frame_chunk_size', 1000000000)
        if batch_size * frames < frame_chunk_size:
            video_features = self.vision_tower(pixel_values_videos, output_hidden_states=True)
            selected_video_feature = video_features.hidden_states[vision_feature_layer]
        else:
            selected_video_feature = []
            for i in range(0, batch_size*frames, frame_chunk_size):
                pixel_values_videos_chunk = pixel_values_videos[i:i+frame_chunk_size]
                selected_video_feature.append(
                    self.vision_tower(pixel_values_videos_chunk, 
                                      output_hidden_states=True
                    ).hidden_states[vision_feature_layer]
                )
            selected_video_feature = torch.cat(selected_video_feature)

        # Compression video tokens
        input_ids, attention_mask, selected_video_feature, position_ids, cache_position, frames, keypatches_mask = self.compress_video_tokens(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            selected_video_feature=selected_video_feature, 
            position_ids=position_ids,
            cache_position=cache_position,
            labels=labels,
        )

        if vision_feature_select_strategy == "default":
            selected_video_feature = selected_video_feature[:, 1:]
        elif vision_feature_select_strategy == "full":
            selected_video_feature = selected_video_feature
        video_features = self.multi_modal_projector(selected_video_feature)

        video_features = self.apply_pooling(video_features)
        video_features = video_features.reshape(batch_size, frames * video_features.shape[1], -1)
        image_newline = self.image_newline[None, None, :].repeat(batch_size, 1, 1).to(video_features.device)
        video_features = torch.cat((video_features, image_newline), dim=1)
        video_features = video_features.flatten(0, 1)

    if inputs_embeds is None:
        inputs_embeds = self.get_input_embeddings()(input_ids)

    # Concatenate vision and language features
    if pixel_values is not None:
        special_image_mask = (
            (input_ids == self.config.image_token_index)
            .unsqueeze(-1)
            .expand_as(inputs_embeds)
            .to(inputs_embeds.device)
        )
        image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)
    if pixel_values_videos is not None:
        special_video_mask = (
            (input_ids == self.config.video_token_index)
            .unsqueeze(-1)
            .expand_as(inputs_embeds)
            .to(inputs_embeds.device)
        )
        video_features = video_features.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(special_video_mask, video_features)
        if keypatches_mask is not None:
            keypatches_mask = torch.zeros_like(input_ids).bool().masked_scatter(special_video_mask[:,:,0], keypatches_mask)

    if is_prefill and chunk_size is not None: # Chunked prefill stage
        assert past_key_values is not None
        kvcache_compression = getattr(past_key_values, 'kvcache_compression', False)
        for seg_id, (s, e, dtype) in enumerate(modality_segments):
            if dtype == 'text': # Prefill text without kvcache_compression
                past_key_values.kvcache_compression = False
                outputs = self.language_model(
                    attention_mask=attention_mask[:,:e],
                    position_ids=position_ids[:,s:e],
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds[:,s:e],
                    use_cache=True,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    cache_position=cache_position[:e],
                    logits_to_keep=logits_to_keep,
                )
                past_key_values = outputs['past_key_values']
            elif dtype == 'video': # Prefill video may with kvcache_compression
                num_chunks = math.ceil((e - s) / chunk_size)
                past_key_values.kvcache_compression = kvcache_compression
                for idx in tqdm(range(num_chunks), total=num_chunks, desc='Prefilling chunk', disable=not DEBUG_MODE):
                    ss = s + idx * chunk_size
                    ee = min(s + (idx + 1) * chunk_size, e)
                    if keypatches_mask is not None:
                        past_key_values.keypatches_mask_chunk = keypatches_mask[0, ss:ee]
                    position_ids_chunk, cache_position_chunk, attention_mask_chunk, inputs_embeds_chunk, prompt_length = self.forge_input_chunks(
                        ss, ee, modality_segments, position_ids, cache_position, attention_mask, past_key_values, inputs_embeds
                    )
                    if hasattr(past_key_values, 'before_forward'):
                        past_key_values.before_forward(prompt_length=prompt_length)
                    outputs = self.language_model(
                        attention_mask=attention_mask_chunk,
                        position_ids=position_ids_chunk,
                        past_key_values=past_key_values,
                        inputs_embeds=inputs_embeds_chunk,
                        use_cache=True,
                        output_attentions=output_attentions,
                        output_hidden_states=output_hidden_states,
                        return_dict=return_dict,
                        cache_position=cache_position_chunk,
                        logits_to_keep=logits_to_keep,
                    )
                    past_key_values = outputs['past_key_values']
                    if hasattr(past_key_values, 'after_forward'):
                        past_key_values.after_forward()
                past_key_values.keypatches_mask_chunk = None
                past_key_values.kvcache_compression = False # Turned off for decoding
            else:
                raise ValueError
    else: # Decode / Standard prefill stage
        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
        )

    logits = outputs[0]

    loss = None
    if labels is not None:
        # Shift so that tokens < n predict n
        if attention_mask is not None:
            shift_attention_mask = attention_mask[..., 1:]
            shift_logits = logits[..., :-1, :][shift_attention_mask.to(logits.device) != 0].contiguous()
            shift_labels = labels[..., 1:][shift_attention_mask.to(labels.device) != 0].contiguous()
        else:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1).to(shift_logits.device)
        )

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return LlavaOnevisionCausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        image_hidden_states=image_features if pixel_values is not None else None,
        video_hidden_states=video_features if pixel_values_videos is not None else None,
    )

import transformers
from .model import VLM_Model
from transformers import AutoProcessor
from transformers import LlavaOnevisionConfig, LlavaOnevisionForConditionalGeneration

class AdaReTake_LLaVA_VLM_Model(VLM_Model):
    def __init__(self, model_path, extra_model_config):
        self.vlm_model_path = model_path
        transformers.models.qwen2.modeling_qwen2.Qwen2Attention.__init__ = retake_Qwen2Attention_init
        transformers.models.qwen2.modeling_qwen2.Qwen2Attention.forward = retake_Qwen2Attention_forward
        transformers.models.llava_onevision.modeling_llava_onevision.LlavaOnevisionForConditionalGeneration.get_chunk_size = retake_LlavaOnevisionForConditionalGeneration_get_chunk_size
        transformers.models.llava_onevision.modeling_llava_onevision.LlavaOnevisionForConditionalGeneration.segment_input_ids = retake_LlavaOnevisionForConditionalGeneration_segment_input_ids
        transformers.models.llava_onevision.modeling_llava_onevision.LlavaOnevisionForConditionalGeneration.compress_video_tokens = retake_LlavaOnevisionForConditionalGeneration_compress_video_tokens
        transformers.models.llava_onevision.modeling_llava_onevision.LlavaOnevisionForConditionalGeneration.forge_input_chunks = retake_LlavaOnevisionForConditionalGeneration_forge_input_chunks
        transformers.models.llava_onevision.modeling_llava_onevision.LlavaOnevisionForConditionalGeneration.forward = retake_LlavaOnevisionForConditionalGeneration_forward
        llava_onevision_config = LlavaOnevisionConfig.from_pretrained(model_path)
        llava_onevision_config.text_config.rope_scaling = {
            'rope_type': 'yarn',
            'factor': 4,
            'beta_fast': 32.0,
            'beta_slow': 1.0,   
        }
        llava_onevision_config.longvideo_kwargs = {
            'frame_chunk_size': 32,
            'chunked_prefill_frames': 32,
            # Keyframe compression
            'visual_compression': True,
            'visual_compression_kwargs': {
                'compression_ratio': 1.0,
                'compression_method': 'Keyframe',
                'patch_sync': False,
                'return_keyframe_mask': True
            },
            # KVCache compression
            'kvcache_compression': True,
            'kvcache_compression_kwargs': {
                'dynamic_compression_ratio': True,
                'compression_method': 'pivotkv',
                'pos_embed_reforge': True,
                'max_input_length': 40000
            },
        }
        self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            model_path,
            config=llava_onevision_config,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto"
        ).eval()
        self.processor = AutoProcessor.from_pretrained(model_path)
    
    def infer(self, prompt, frames_info):
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
        
        generated_ids = self.model.generate(**inputs, max_new_tokens=128, do_sample=False, temperature=0)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs['input_ids'], generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        answer = output_text[0]
        return answer