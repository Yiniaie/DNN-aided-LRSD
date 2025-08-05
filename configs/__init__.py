from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import os
import os.path as osp

@dataclass
class Config:
    method_name: str = "DNN-aided-LRSD"  # 计算方法
    dataset_path: str = r"datas"
    dataset_input_type: str = "base"
    base_dir: str = r".\result"
    save_images: bool = True  # 是否存储图片，bool 类型更合适
    Dataset_name: str = "DAUB"
    post_processing: str = 'post_processing_tophat'  # 后处理 post_processing_tophat  post_processing_mean



    # model parameters
    seed: int = 21
    img_resize: bool = False  # 使用bool更语义清晰
    img_resize_size: int = 256
    iteration: int = 2000
    batch_size: int = 1
    frame_input: int = 60
    frame_all: int = 500
    rank: int = 40
    patch: bool = False
    patch_size: int = 64
    slide_step: int = 64
    lr: float = 0.0012

    sine_layers: int = 1
    lambda1: float = 0.001
    lambda2: float = 8
    lambda3: float = 1
    lambda4: float = 1
    lambda5: float = 1
    mu1: float = 0.8
    down: int = 2
    omega_0: int = 30
    convergence: float = 0.03

    def __post_init__(self):
        # Derived attributes
        self.time_name = datetime.now().strftime('%Y%m%dT%H-%M-%S')
        self.folder_name = f"{self.time_name}_{self.method_name}"
        self.base_dir = os.path.join(self.base_dir, self.Dataset_name)
        self.save_folder =  os.path.join(self.base_dir, self.folder_name)

@dataclass
class Config_3DSTPM(Config):
    method_name: str = 'net_3DSTPM'  # 模式建议
    post_processing: str = 'post_processing_mean'
    base_dir: str = r".\result"