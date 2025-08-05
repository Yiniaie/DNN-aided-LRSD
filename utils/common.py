import torch
import torch.nn as nn
import numpy as np
import random


def get_collate_fn_pad(batch_size, frame_input):
    target_batch_size = batch_size * frame_input
    def collate_fn(batch):
        valid_size = len(batch)
        if valid_size == target_batch_size:
            return default_collate(batch), {'valid_size': valid_size}
        padded_batch = list(batch)
        while len(padded_batch) < target_batch_size:
            padded_batch.append(random.choice(batch))
        return default_collate(padded_batch), {'valid_size': valid_size}
    return collate_fn


def default_collate(batch):
    imgs = torch.stack([item[0] for item in batch], dim=0)
    masks = torch.stack([item[1] for item in batch], dim=0)
    coords = [item[2] for item in batch]  # 保持 list 形式，不堆叠成 tensor
    return imgs, masks, coords


def post_processing_mean(img, normalize_per_channel=True, ratio=0.2):
    """
    - normalize_per_channel: 是否对每个通道分别归一化
    - 输出为 uint8，(C, H, W)
    """
    img = img.astype(np.float32)
    C, H, W = img.shape
    out = np.zeros((C, H, W), dtype=np.uint8)

    # 每通道单独归一化（可选）
    if normalize_per_channel:
        for i in range(C):
            ch = img[i]
            ch_min = ch.min()
            ch_max = ch.max()
            img[i] = (ch - ch_min) / (ch_max - ch_min + 1e-8) if ch_max != ch_min else 0
    else:
        img_min = img.min()
        img_max = img.max()
        img = (img - img_min) / (img_max - img_min + 1e-8)

    for i in range(C):
        layer = img[i]
        max_val = layer.max()
        threshold = ratio * max_val
        mask = layer > threshold

        if np.any(mask):
            mean_val = layer[mask].mean()
            processed = np.maximum(layer - mean_val, 0)
        else:
            processed = np.zeros_like(layer)

        # 每层独立归一化再转 uint8
        p_min = processed.min()
        p_max = processed.max()
        if p_max != p_min:
            processed = (processed - p_min) / (p_max - p_min + 1e-8)
        else:
            processed[:] = 0

        out[i] = (processed * 255).astype(np.uint8)

    return out

def np_normalized(input: np.ndarray) -> np.ndarray:
    """
    将 float32 numpy 数组按每张图像独立归一化到 [0, 255] 并转为 uint8。

    Args:
        input: np.ndarray of shape (N, H, W), float32

    Returns:
        np.ndarray of shape (N, H, W), uint8
    """
    input = input.astype(np.float32)
    output = np.zeros_like(input)

    for i in range(input.shape[0]):
        image = input[i]
        min_val = image.min()
        max_val = image.max()

        if max_val != min_val:
            norm = (image - min_val) / (max_val - min_val)
            output[i] = norm * 255.
        else:
            output[i] = 0  # 或 255，具体取决于你需求

    return output.astype(np.uint8)

# 核范数计算函数
def compute_nuc_norms_CPU(lambda1, A, B, C, iter_value):
    return (
        lambda1 * torch.norm(A[iter_value % A.shape[0]], 'nuc') +
        lambda1 * torch.norm(B[:, iter_value % B.shape[1]], 'nuc') +
        lambda1 * torch.norm(C[:, :, iter_value % C.shape[2]], 'nuc')
    )