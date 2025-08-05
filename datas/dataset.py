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
class MyDataset_Ours_v2_mat(Dataset):

    # mode取值为: full train test
    def __init__(self, dataset_path, sequences, args):
        super().__init__()
        self.names = []
        self.dataset_path = dataset_path
        self.sequences = sequences
        self.base_size = 256
        self.frame_num = args.frame_num
        self.img_resize = args.img_resize
        self.SE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        for filename in os.listdir(osp.join(self.dataset_path, self.sequences)):
            if filename.endswith('bmp'):
                self.names.append(filename)
        if 0:
            self.names.sort(key=lambda x: int(x.split('.')[0].split(')')[0].split('(')[1]))
        else:
            self.names.sort(key=lambda x: int(x.split('.')[0]))

        mat = sio.loadmat(osp.join(args.dataset_path,'Zlabel', self.sequences))
        self.masks = mat["var"]
        # mask = self.masks[0,0]['object_coords']
        if len(self.names)>self.frame_num:
            self.size = len(self.names[0:self.frame_num])
        else:
            self.size = len(self.names)
        # print("break point")

    def __getitem__(self, i):
        name = self.names[i]
        img = cv2.imread(osp.join(self.dataset_path, self.sequences, name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        [n1, n2] = img.shape
        if self.img_resize:
            img = cv2.resize(img, [self.base_size, self.base_size], interpolation=cv2.INTER_LINEAR) / 255.
        else:
            img = img / 255.
        [n11, n22] = img.shape
        # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        try:
            img = torch.from_numpy(img).type(torch.cuda.FloatTensor)

        except RuntimeError as e:
            print(f"CUDA error: {e}")
            # 打印完整的堆栈跟踪
            import traceback
            traceback.print_exc()

        mask0 = torch.zeros(img.shape)
        object_coords = torch.tensor(self.masks[0, int(name.split('.')[0])-1]['object_coords'].astype(np.int64), dtype=torch.int64)
        mask0[(object_coords[:, 0]*n11/n1).long(), (object_coords[:, 1]*n22/n2).long()] = 255
        mask0 = mask0.numpy()
        mask0 = cv2.dilate(mask0, self.SE)
        mask = torch.tensor(mask0, device=img.device, dtype=torch.uint8)
        return img, mask

    def __len__(self):

        return self.size
class Ours_v2_mat(Dataset):

    # mode取值为: full train test
    def __init__(self, seqences_path, foldername, args):
        super().__init__()
        self.iamge_name = []
        self.seqences_path = seqences_path
        self.sequences_name = args.sequence_name
        self.dataset_path = args.dataset_path


        self.frame_all = args.frame_all
        self.img_resize = args.img_resize
        self.base_size = args.img_resize_size
        self.SE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        self.frame_input = args.frame_input
        for filename in os.listdir(osp.join(self.seqences_path, self.sequences_name)):
            if filename.endswith('bmp'):
                self.iamge_name.append(filename)
        if 0:
            self.iamge_name.sort(key=lambda x: int(x.split('.')[0].split(')')[0].split('(')[1]))
        else:
            self.iamge_name.sort(key=lambda x: int(x.split('.')[0]))

        mat = sio.loadmat(osp.join(args.dataset_path,args.Dataset_name+'_label', self.sequences_name))
        self.masks = mat["var"]
        # mask = self.masks[0,0]['object_coords']
        if len(self.iamge_name)>self.frame_all:
            self.size = len(self.iamge_name[0:self.frame_all])
        else:
            self.size = len(self.iamge_name)
        # print("break point")
        img = cv2.imread(osp.join(self.seqences_path, self.sequences_name, self.iamge_name[1]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        [n11, n22] = img.shape
        self.input_size = [self.frame_input, n11, n22]

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
            mask0 = torch.zeros((n11, n22), dtype=torch.uint8, device=img.device)
            object_coords_np = self.masks[0, int(iamge_name.split('.')[0]) - 1]['object_coords']

            if object_coords_np is None or object_coords_np.size == 0:
                object_coords = torch.zeros((0, 2), dtype=torch.int64)
                return img, mask0, object_coords

            object_coords = torch.tensor(object_coords_np.astype(np.int64), dtype=torch.int64)

            if object_coords.numel() == 0 or object_coords.ndim != 2 or object_coords.size(1) != 2:
                object_coords = torch.zeros((0, 2), dtype=torch.int64)
                return img, mask0, object_coords

            x_idx = (object_coords[:, 0] * n11 / n1).long()
            y_idx = (object_coords[:, 1] * n22 / n2).long()

            valid_mask = (x_idx >= 0) & (x_idx < n11) & (y_idx >= 0) & (y_idx < n22)
            x_idx = x_idx[valid_mask]
            y_idx = y_idx[valid_mask]

            if x_idx.numel() == 0:
                object_coords = torch.zeros((0, 2), dtype=torch.int64)
                return img, mask0, object_coords

            mask_np = mask0.cpu().numpy()
            mask_np[x_idx.cpu(), y_idx.cpu()] = 255
            mask_np = cv2.dilate(mask_np, self.SE)
            if self.img_resize:
                mask_np = cv2.resize(mask_np, (self.base_size, self.base_size),
                                   interpolation=cv2.INTER_NEAREST)  # mask用nearest防止插值
            mask = torch.tensor(mask_np, device=img.device, dtype=torch.uint8)

        else:
            self.mask_path = osp.join(self.dataset_path, 'masks', self.sequences_name)
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
            if self.img_resize:
                mask0 = cv2.resize(mask0, (self.base_size, self.base_size),
                                   interpolation=cv2.INTER_NEAREST)  # mask用nearest防止插值
            mask = torch.tensor(mask0, device=img.device, dtype=torch.uint8)

        return img, mask, object_coords

    def __len__(self):
        return self.size
class Ours_v2_mat_MultiInput_SingleOutput_all_frames(Dataset):
    def __init__(self, seqences_path, foldername, args):
        super().__init__()
        self.seqences_path = seqences_path
        self.sequences_name = args.sequence_name
        self.dataset_path = args.dataset_path

        self.img_resize = args.img_resize
        self.base_size = args.img_resize_size
        self.SE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        self.frame_input = args.frame_input
        self.frame_all = args.frame_all

        # 排序图像名
        self.image_names = sorted([
            f for f in os.listdir(osp.join(self.seqences_path, self.sequences_name))
            if f.endswith('bmp')
        ], key=lambda x: int(x.split('.')[0]))

        self.total_frames = len(self.image_names)


        # 读取 mat 中的 ground truth
        mat = sio.loadmat(osp.join(args.dataset_path, 'Zlabel', self.sequences_name))
        self.masks = mat["var"]
        if len(self.image_names)>self.frame_all:
            self.size = len(self.image_names[0:self.frame_all])
        else:
            self.size = len(self.image_names)

    def __len__(self):
        return self.size  # 所有帧都参与推理

    def __getitem__(self, idx):
        imgs = []

        # 构造5帧：前4帧辅助帧，第5帧目标帧（即当前帧 idx）
        for offset in range(-4, 1):
            frame_idx = idx + offset
            if frame_idx < 0:
                frame_idx = 0  # 头部不足时重复第一帧
            elif frame_idx >= self.total_frames:
                frame_idx = self.total_frames - 1  # 尾部保护
            image_name = self.image_names[frame_idx]
            img_path = osp.join(self.seqences_path, self.sequences_name, image_name)

            img = cv2.imread(img_path)
            if img is None:
                raise RuntimeError(f"Cannot load image {img_path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, [self.base_size, self.base_size], interpolation=cv2.INTER_LINEAR) / 255. \
                if self.img_resize else img / 255.
            img = torch.from_numpy(img).float()
            imgs.append(img)

        imgs = torch.stack(imgs, dim=0)  # [5, H, W]

        # 加载第 idx 帧的标签（mask + coords）
        image_name = self.image_names[idx]
        n11, n22 = imgs[-1].shape

        if 'data' in self.sequences_name:
            mask = torch.zeros((n11, n22), dtype=torch.uint8)
            object_coords_np = self.masks[0, int(image_name.split('.')[0]) - 1]['object_coords']

            if object_coords_np is None or object_coords_np.size == 0:
                object_coords = torch.zeros((0, 2), dtype=torch.int64)
                return imgs, mask, object_coords

            object_coords = torch.tensor(object_coords_np.astype(np.int64), dtype=torch.int64)

            x_idx = (object_coords[:, 0] * n11 / imgs[-1].shape[0]).long()
            y_idx = (object_coords[:, 1] * n22 / imgs[-1].shape[1]).long()
            valid = (x_idx >= 0) & (x_idx < n11) & (y_idx >= 0) & (y_idx < n22)
            x_idx = x_idx[valid]
            y_idx = y_idx[valid]

            if x_idx.numel() == 0:
                object_coords = torch.zeros((0, 2), dtype=torch.int64)
                return imgs, mask, object_coords

            mask_np = mask.numpy()
            mask_np[x_idx, y_idx] = 255
            mask_np = cv2.dilate(mask_np, self.SE)
            if self.img_resize:
                mask_np = cv2.resize(mask_np, (self.base_size, self.base_size),
                                   interpolation=cv2.INTER_NEAREST)  # mask用nearest防止插值
            mask = torch.tensor(mask_np, dtype=torch.uint8)

        else:
            self.mask_path = osp.join(self.dataset_path, 'masks', self.sequences_name)
            object_coords_np = self.masks[0, int(image_name.split('.')[0]) - 1]['object_coords']

            if object_coords_np is None or object_coords_np.size == 0:
                object_coords = torch.zeros((0, 2), dtype=torch.int64)
            else:
                object_coords = torch.tensor(object_coords_np.astype(np.int64), dtype=torch.int64)
                object_coords = object_coords.squeeze()[0:2] if object_coords.numel() >= 2 else torch.zeros(2, dtype=torch.int64)

            mask_path = osp.join(self.mask_path, image_name)
            mask0 = cv2.imread(mask_path)
            if mask0 is None:
                print(f"[Warning] Failed to read mask: {mask_path}")
                return None

            mask0 = cv2.cvtColor(mask0, cv2.COLOR_BGR2GRAY)
            if self.img_resize:
                mask0 = cv2.resize(mask0, (self.base_size, self.base_size),
                                   interpolation=cv2.INTER_NEAREST)  # mask用nearest防止插值
            mask0[mask0 > 0] = 255
            mask = torch.tensor(mask0, device=img.device, dtype=torch.uint8)

        return imgs, mask, object_coords

class Ours_v2_mat_drop_last(Dataset):
    def __init__(self, seqences_path, foldername, args):
        super().__init__()
        self.seqences_path = seqences_path
        self.sequences_name = args.sequence_name
        self.dataset_path = args.dataset_path

        self.img_resize = args.img_resize
        self.base_size = args.img_resize_size
        self.SE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        self.frame_input = args.frame_input

        # 排序图像名
        self.image_names = sorted([
            f for f in os.listdir(osp.join(self.seqences_path, self.sequences_name))
            if f.endswith('bmp')
        ], key=lambda x: int(x.split('.')[0]))

        self.total_frames = len(self.image_names)
        self.seq_len = 5  # 5帧滑窗：前4帧辅助，第5帧目标

        # 读取 mat 中的 ground truth
        mat = sio.loadmat(osp.join(args.dataset_path, 'Zlabel', self.sequences_name))
        self.masks = mat["var"]

    def __len__(self):
        return self.total_frames  # 所有帧都参与推理

    def __getitem__(self, idx):
        imgs = []

        # 构造5帧：前4帧辅助帧，第5帧目标帧（即当前帧 idx）
        for offset in range(-4, 1):
            frame_idx = idx + offset
            if frame_idx < 0:
                frame_idx = 0  # 头部不足时重复第一帧
            elif frame_idx >= self.total_frames:
                frame_idx = self.total_frames - 1  # 尾部保护
            image_name = self.image_names[frame_idx]
            img_path = osp.join(self.seqences_path, self.sequences_name, image_name)

            img = cv2.imread(img_path)
            if img is None:
                raise RuntimeError(f"Cannot load image {img_path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, [self.base_size, self.base_size], interpolation=cv2.INTER_LINEAR) / 255. \
                if self.img_resize else img / 255.
            img = torch.from_numpy(img).float()
            imgs.append(img)

        imgs = torch.stack(imgs, dim=0)  # [5, H, W]

        # 加载第 idx 帧的标签（mask + coords）
        image_name = self.image_names[idx]
        n11, n22 = imgs[-1].shape

        if 'data' in self.sequences_name:
            mask = torch.zeros((n11, n22), dtype=torch.uint8)
            object_coords_np = self.masks[0, int(image_name.split('.')[0]) - 1]['object_coords']

            if object_coords_np is None or object_coords_np.size == 0:
                object_coords = torch.zeros((0, 2), dtype=torch.int64)
                return imgs, mask, object_coords

            object_coords = torch.tensor(object_coords_np.astype(np.int64), dtype=torch.int64)

            x_idx = (object_coords[:, 0] * n11 / imgs[-1].shape[0]).long()
            y_idx = (object_coords[:, 1] * n22 / imgs[-1].shape[1]).long()
            valid = (x_idx >= 0) & (x_idx < n11) & (y_idx >= 0) & (y_idx < n22)
            x_idx = x_idx[valid]
            y_idx = y_idx[valid]

            if x_idx.numel() == 0:
                object_coords = torch.zeros((0, 2), dtype=torch.int64)
                return imgs, mask, object_coords

            mask_np = mask.numpy()
            mask_np[x_idx, y_idx] = 255
            mask_np = cv2.dilate(mask_np, self.SE)
            mask = torch.tensor(mask_np, dtype=torch.uint8)

        else:
            raise NotImplementedError("仅支持 'data' 模式")

        return imgs, mask, object_coords