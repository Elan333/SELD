import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

from .lcm import min_max_normalize


def convert_hf_state_dict(state, depth=24):
    converted = {}
    direct = {
        "embeddings.cls_token": "cls_token",
        "embeddings.patch_embeddings.weight": "patch_embed.proj.weight",
        "embeddings.patch_embeddings.bias": "patch_embed.proj.bias",
        "embeddings.register_tokens": "storage_tokens",
    }
    for source, target in direct.items():
        if source in state:
            converted[target] = state[source]
    if "embeddings.mask_token" in state:
        value = state["embeddings.mask_token"]
        converted["mask_token"] = value.squeeze(1) if value.dim() == 3 else value

    for index in range(depth):
        hf = f"layer.{index}"
        local = f"blocks.{index}"
        for norm in ("norm1", "norm2"):
            for key in ("weight", "bias"):
                name = f"{hf}.{norm}.{key}"
                if name in state:
                    converted[f"{local}.{norm}.{key}"] = state[name]
        q = state.get(f"{hf}.attention.q_proj.weight")
        if q is not None:
            k = state[f"{hf}.attention.k_proj.weight"]
            v = state[f"{hf}.attention.v_proj.weight"]
            converted[f"{local}.attn.qkv.weight"] = torch.cat([q, k, v], dim=0)
            q_bias = state.get(f"{hf}.attention.q_proj.bias")
            if q_bias is not None:
                k_bias = state.get(f"{hf}.attention.k_proj.bias")
                v_bias = state[f"{hf}.attention.v_proj.bias"]
                if k_bias is None:
                    k_bias = torch.zeros_like(q_bias)
                converted[f"{local}.attn.qkv.bias"] = torch.cat([q_bias, k_bias, v_bias], dim=0)
        for key in ("weight", "bias"):
            name = f"{hf}.attention.o_proj.{key}"
            if name in state:
                converted[f"{local}.attn.proj.{key}"] = state[name]
            name = f"{hf}.mlp.up_proj.{key}"
            if name in state:
                converted[f"{local}.mlp.fc1.{key}"] = state[name]
            name = f"{hf}.mlp.down_proj.{key}"
            if name in state:
                converted[f"{local}.mlp.fc2.{key}"] = state[name]
    for key in ("weight", "bias"):
        name = f"norm.{key}"
        if name in state:
            converted[name] = state[name]
    return converted


def load_dinov3(weight_dir, repo_dir=None):
    if repo_dir:
        sys.path.insert(0, repo_dir)
    from dinov3.models.vision_transformer import vit_large
    from safetensors.torch import load_file

    model = vit_large(patch_size=16, n_storage_tokens=4)
    state = load_file(os.path.join(weight_dir, "model.safetensors"))
    model.load_state_dict(convert_hf_state_dict(state), strict=False)
    model.eval()
    return model


class MultiLayerHook:
    def __init__(self, blocks, layers, head_weights, heads, storage_tokens):
        self.layers = sorted(layers)
        self.head_weights = head_weights
        self.heads = heads
        self.storage_tokens = storage_tokens
        self.outputs = {layer: [] for layer in self.layers}
        self.handles = [
            blocks[layer].attn.qkv.register_forward_hook(self._capture(layer))
            for layer in self.layers
        ]

    def _capture(self, layer):
        def hook(module, inputs, output):
            self.outputs[layer].append(output.detach().cpu())

        return hook

    def attention_maps(self):
        maps = {}
        total_weight = sum(self.head_weights.values())
        for layer in self.layers:
            qkv = self.outputs[layer][-1]
            batch, tokens, channels = qkv.shape
            head_dim = channels // 3 // self.heads
            grid = int(math.isqrt(tokens - 1 - self.storage_tokens))
            extra = 1 + self.storage_tokens
            q, k, _ = qkv.reshape(batch, tokens, 3, self.heads, head_dim).unbind(2)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            weights = F.softmax(torch.matmul(q, k.transpose(-2, -1)) * head_dim ** -0.5, dim=-1)
            attention = weights.numpy()
            scores = np.zeros((batch, grid * grid), dtype=np.float64)
            for item in range(batch):
                cls = sum(weight * attention[item, head, 0, extra:]
                          for head, weight in self.head_weights.items())
                patch = sum(weight * attention[item, head, extra:, 0]
                            for head, weight in self.head_weights.items())
                scores[item] = (0.5 * cls + 0.5 * patch) / total_weight
            maps[layer] = scores.reshape(batch, grid, grid)
        return maps

    def clear(self):
        for layer in self.layers:
            self.outputs[layer] = []

    def remove(self):
        for handle in self.handles:
            handle.remove()


def fuse_attention(maps, mid_layer, deep_layer):
    mid = maps[mid_layer]
    deep = maps[deep_layer]
    fused = np.zeros_like(deep)
    for index in range(mid.shape[0]):
        mid_map = min_max_normalize(mid[index].astype(np.float64))
        deep_map = min_max_normalize(deep[index].astype(np.float64))
        fused[index] = min_max_normalize(mid_map * (1.0 - deep_map))
    return fused
