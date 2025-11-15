import torch
import torch.nn as nn
import numpy as np
import random
from scipy.ndimage import grey_dilation, grey_erosion, gaussian_filter, convolve,  sobel


def collate_fn_filter_none(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None  # 或者 raise StopIteration("Empty batch")
    return default_collate(batch)

def convert_to_serializable(d):
    """将 dict 中不可序列化的 Path 等对象转换为字符串"""
    new_d = {}
    for k, v in d.items():
        if isinstance(v, dict):
            new_d[k] = convert_to_serializable(v)
        elif isinstance(v, Path):
            new_d[k] = str(v)
        else:
            new_d[k] = v
    return new_d


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

def post_processing(input: np.ndarray, R_o=11, R_i=3) -> np.ndarray:
    """
    Morphological small target enhancement + per-frame normalization to [0,255].

    Args:
        input: np.ndarray of shape (N, H, W), float32 or uint8

    Returns:
        np.ndarray of shape (N, H, W), uint8
    """
    img = input.astype(np.float32)
    n1, n2, n3 = img.shape

    # Construct ring-shaped structuring element
    d = 2 * R_o + 1
    SE = np.ones((d, d), dtype=np.float32)
    start = R_o + 1 - R_i
    end = R_o + 1 + R_i
    SE[start:end, start:end] = 0
    B_b = np.ones((R_i, R_i), dtype=np.float32)

    # Top-hat enhancement + normalization in a single loop
    for i in range(n1):
        # white top-hat: orig - open(orig)
        img_d = grey_dilation(img[i], structure=SE)
        img_e = grey_erosion(img_d, structure=B_b)
        out = img[i] - img_e
        out[out < 0] = 0

        # per-frame normalization to [0,255]
        min_val = out.min()
        max_val = out.max()
        if max_val != min_val:
            out = (out - min_val) / (max_val - min_val) * 255.
        else:
            out[:] = 0  # all values are identical

        img[i] = out

    return img.astype(np.uint8)


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