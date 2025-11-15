import torch
import numpy as np
import cv2
import os
import os.path as osp
import random
from torch.utils.data import Dataset
from scipy.io import loadmat
import scipy.io as sio
import h5py


class MyDataset(Dataset):

    # mode取值为: full train test
    def __init__(self, dataset_path, sequences):
        super().__init__()
        self.names = []
        self.dataset_path = dataset_path
        self.sequences = sequences
        self.base_size = 256
        for filename in os.listdir(osp.join(self.dataset_path, self.sequences)):
            if filename.endswith('png'):
                self.names.append(filename)
        if 1:
            self.names.sort(key=lambda x: int(x.split('.')[0].split(')')[0].split('(')[1]))
        else:
            self.names.sort(key=lambda x: int(x.split('.')[0]))
        self.size = len(self.names[0:50])
        # print("break point")

    def __getitem__(self, i):
        name = self.names[i]
        img = cv2.imread(osp.join(self.dataset_path, self.sequences, name))
        mask = cv2.imread(osp.join(self.dataset_path.replace("images", "masks"), self.sequences, name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, [self.base_size, self.base_size], interpolation=cv2.INTER_LINEAR) / 255.
        mask = cv2.resize(mask, [self.base_size, self.base_size], interpolation=cv2.INTER_NEAREST)
        img = torch.from_numpy(img).type(torch.cuda.FloatTensor)
        mask = torch.from_numpy(mask).type(torch.cuda.FloatTensor)
        return img, mask

    def __len__(self):

        return self.size

class MyDataset_DAUB_IRDST(Dataset):

    # mode取值为: full train test
    def __init__(self, seqences_path, foldername, args):
        super().__init__()
        self.iamge_name = []
        self.seqences_path = seqences_path
        self.sequences_name = args.sequence_name
        self.dataset_path = args.dataset_path


        self.frame_num = args.frame_num
        self.img_resize = args.img_resize
        self.base_size = args.img_resize_size
        self.SE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        self.Batch_size = args.Batch_size
        for filename in os.listdir(osp.join(self.seqences_path, self.sequences_name)):
            if filename.endswith('bmp'):
                self.iamge_name.append(filename)
        if 0:
            self.iamge_name.sort(key=lambda x: int(x.split('.')[0].split(')')[0].split('(')[1]))
        else:
            self.iamge_name.sort(key=lambda x: int(x.split('.')[0]))

        mat = sio.loadmat(osp.join(self.seqences_path.replace('images', 'labels'), self.sequences_name))
        self.masks = mat["var"]
        # mask = self.masks[0,0]['object_coords']
        if len(self.iamge_name)>self.frame_num:
            self.size = len(self.iamge_name[0:self.frame_num])
        else:
            self.size = len(self.iamge_name)
        # print("break point")
        img = cv2.imread(osp.join(self.seqences_path, self.sequences_name, self.iamge_name[1]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        [n11, n22] = img.shape
        self.input_size = [self.Batch_size, n11, n22]

    def __getitem__(self, i):
        iamge_name = self.iamge_name[i]
        img_path = osp.join(self.seqences_path, self.sequences_name, iamge_name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"[Warning] Failed to read image: {img_path}")
            return None  # 或者 raise Exception

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        n1, n2 = img.shape

        if self.img_resize:
            img = cv2.resize(img, [self.base_size, self.base_size], interpolation=cv2.INTER_LINEAR) / 255.
        else:
            img = img / 255.
        n11, n22 = img.shape

        try:
            img = torch.from_numpy(img).type(torch.cuda.FloatTensor)
        except RuntimeError as e:
            print(f"CUDA error: {e}")
            import traceback
            traceback.print_exc()
            return None

        # ----------------------mask-------------------------#
        if 'data' in self.sequences_name:
            mask0 = torch.zeros(img.shape, dtype=torch.float32)
            object_coords_np = self.masks[0, int(iamge_name.split('.')[0]) - 1]['object_coords']

            if object_coords_np is None or object_coords_np.size == 0:
                return None  # 跳过空目标帧

            object_coords = torch.tensor(object_coords_np.astype(np.int64), dtype=torch.int64)

            if object_coords.numel() == 0 or object_coords.ndim != 2 or object_coords.size(1) != 2:
                return None  # 无效坐标，跳过

            x_idx = (object_coords[:, 0] * n11 / n1).long()
            y_idx = (object_coords[:, 1] * n22 / n2).long()

            valid_mask = (x_idx >= 0) & (x_idx < n11) & (y_idx >= 0) & (y_idx < n22)
            x_idx = x_idx[valid_mask]
            y_idx = y_idx[valid_mask]

            if x_idx.numel() == 0:
                return None  # 所有坐标都非法，跳过

            mask0[x_idx, y_idx] = 255
            mask0 = mask0.numpy()
            mask0 = cv2.dilate(mask0, self.SE)
            mask = torch.tensor(mask0, device=img.device, dtype=torch.uint8)

        else:
            self.mask_path = osp.join(self.seqences_path.replace('images', 'masks'), self.sequences_name)
            object_coords_np = self.masks[0, int(iamge_name.split('.')[0]) - 1]['object_coords']

            if object_coords_np is None or object_coords_np.size == 0:
                return None

            object_coords = torch.tensor(object_coords_np.astype(np.int64), dtype=torch.int64)
            object_coords = object_coords.squeeze()[0:2] if object_coords.numel() >= 2 else torch.zeros(2,
                                                                                                        dtype=torch.int64)

            mask_path = osp.join(self.mask_path, iamge_name)
            mask0 = cv2.imread(mask_path)
            if mask0 is None:
                print(f"[Warning] Failed to read mask: {mask_path}")
                return None

            mask0 = cv2.cvtColor(mask0, cv2.COLOR_BGR2GRAY)
            mask0[mask0 > 0] = 255
            mask = torch.tensor(mask0, device=img.device, dtype=torch.uint8)

        return img, mask, object_coords
    def __len__(self):

        return self.size
