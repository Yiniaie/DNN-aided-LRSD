import numpy as np
import torch.nn as nn
import torch
import os
import glob
from PIL import Image
from skimage import measure
import matplotlib.pyplot as plt
import math
import natsort
from sklearn.metrics import auc
# import matlab.engine
import scipy.io as sio
import os.path as osp
import cv2
import argparse
from mpl_toolkits.mplot3d import Axes3D

class ROCMetric():
    """Computes pixAcc and mIoU metric scores /
        mask
    """
    def __init__(self):  #bin的意义实际上是确定ROC曲线上的threshold取多少个离散值
        super(ROCMetric, self).__init__()
        self.nclass = 1
        self.bins = 10


    def update(self, preds, labels, args):

        self.frame_num = preds.shape[0]
        self.tp_arr = np.zeros([self.frame_num, self.bins + 1])
        self.pos_arr = np.zeros([self.frame_num, self.bins + 1])
        self.fp_arr = np.zeros([self.frame_num, self.bins + 1])
        self.neg_arr = np.zeros([self.frame_num, self.bins + 1])
        self.class_pos = np.zeros([self.frame_num, self.bins + 1])

        self.args = args
        if  preds.dtype != np.uint8 and labels.dtype != np.uint8:
            raise TypeError("Preds and labels must be of uint8 dtype.")

        for i in range(self.frame_num):
            for iBin in range(self.bins+1):
                score_thresh = iBin * (255/self.bins)
                # print(iBin, "-th, score_thresh: ", score_thresh)
                i_tp, i_pos, i_fp, i_neg, i_class_pos = cal_tp_pos_fp_neg(preds[i, :, :], labels[i, :, :], self.nclass, score_thresh)
                self.tp_arr[i, iBin]    = i_tp
                self.pos_arr[i, iBin]   = i_pos
                self.fp_arr[i, iBin]    = i_fp
                self.neg_arr[i, iBin]   = i_neg
                self.class_pos[i, iBin] = i_class_pos
        self.tp_rates    = np.sum(self.tp_arr, axis= 0) / (np.sum(self.pos_arr, axis= 0) + 0.00001)
        self.fp_rates    = np.sum(self.fp_arr, axis= 0) / (np.sum(self.neg_arr, axis= 0) + 0.00001)
        self.recall      = np.sum(self.tp_arr, axis= 0) / (np.sum(self.pos_arr, axis= 0) + 0.00001)
        self.precision   = np.sum(self.tp_arr, axis= 0) / (np.sum(self.class_pos, axis= 0) + 0.00001)


        if not hasattr(self, 'All_tp_arr'):
            self.All_tp_arr = self.tp_arr
            self.All_pos_arr = self.pos_arr
            self.All_fp_arr = self.fp_arr
            self.All_neg_arr = self.neg_arr
            self.All_class_pos = self.class_pos
        else:
            self.All_tp_arr = np.concatenate((self.All_tp_arr, self.tp_arr),axis=0)
            self.All_pos_arr = np.concatenate((self.All_pos_arr, self.pos_arr),axis=0)
            self.All_fp_arr = np.concatenate((self.All_fp_arr, self.fp_arr),axis=0)
            self.All_neg_arr = np.concatenate((self.All_neg_arr, self.neg_arr),axis=0)
            self.All_class_pos = np.concatenate((self.All_class_pos, self.class_pos),axis=0)
    def get(self):
        self.plot_FPTP_PR()
        return self.tp_rates, self.fp_rates, self.recall, self.precision

    def get_all(self):
        self.Finall_all_tp_rates = np.sum(self.All_tp_arr, axis=0) / (np.sum(self.All_pos_arr, axis=0) + 0.00001)
        self.Finall_all_fp_rates = np.sum(self.All_fp_arr, axis=0) / (np.sum(self.All_neg_arr, axis=0) + 0.00001)
        self.Finall_all_recall = np.sum(self.All_tp_arr, axis=0) / (np.sum(self.All_pos_arr, axis=0) + 0.00001)
        self.Finall_all_precision = np.sum(self.All_tp_arr, axis=0) / (np.sum(self.All_class_pos, axis=0) + 0.00001)
        self.plot_all_FPTP_PR()
        return self.Finall_all_tp_rates, self.Finall_all_fp_rates, self.Finall_all_recall, self.Finall_all_precision

    def plot_FPTP_PR(self):
        tp_rates = np.array([0.99997963, 0.99038264, 0.96111441, 0.87400231, 0.63969207,
                             0.33121196, 0.10480968, 0.02337239, 0.00566463, 0.00126344,
                             0.0])
        fp_rates = np.array([0.99997963, 0.99038264, 0.96111441, 0.87400231, 0.63969207,
                             0.33121196, 0.10480968, 0.02337239, 0.00566463, 0.00126344,
                             0.0])
        ## FPTP Curve
        fig, ax = plt.subplots(num="FPTP Curve", figsize=(5, 5))
        ax.plot(self.fp_rates, self.tp_rates)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve')
        fig.savefig(os.path.join(self.args.save_path_sequence, 'FPTP_pixel_level.png'))

        ## PR Curve
        fig1, ax1 = plt.subplots(num="PR Curve", figsize=(5, 5))
        ax1.plot(self.recall, self.precision)
        ax1.set_xlabel('Recall')
        ax1.set_ylabel('Percision')
        ax1.set_title('PR Curve')
        fig1.savefig(os.path.join(self.args.save_path_sequence, 'PR_curve_pixel_level.png'))
        plt.close('all')
        # ax.set_xlim(0, 1)  # 设置横坐标范围为 0 到 1
        # ax.set_ylim(0, 1)  # 设置纵坐标范围为 0 到 1
        # return fig
    def plot_all_FPTP_PR(self):
        ## FPTP Curve
        fig, ax = plt.subplots(num="FPTP Curve", figsize=(5, 5))
        ax.plot(self.Finall_all_fp_rates, self.Finall_all_tp_rates)
        ax.set_xlabel('All images False Positive Rate')
        ax.set_ylabel('All images True Positive Rate')
        ax.set_title('All images ROC Curve')
        fig.savefig(os.path.join(self.args.save_path_method, 'All images FPTP_pixel_level.png'))

        ## PR Curve
        fig1, ax1 = plt.subplots(num="All images PR Curve", figsize=(5, 5))
        ax1.plot(self.Finall_all_recall, self.Finall_all_precision)
        ax1.set_xlabel('All images Recall')
        ax1.set_ylabel('All images Percision')
        ax1.set_title('All images PR Curve')
        fig1.savefig(os.path.join(self.args.save_path_method, 'All images PR_curve_pixel_level.png'))
        plt.close('all')
class Pd_Fa():
    '''
    Probability of detection (Pd) with target-level:预测目标像素块中心位置与真实目标像素块中位置的差 < distance
    False alarm rate (Fa) with pixel-level: 像素块中心位置欧式距离小于 distance 的所有像素
    '''
    def __init__(self): # #bin的意义实际上是确定ROC曲线上的threshold取多少个离散值
        super(Pd_Fa, self).__init__()
        self.nclass = 1
        self.bins = 10
        self.opts = dict()
    def reset(self):
        self.opts = dict()
    def resumes(self, mat_data, args):
        self.opts = dict()
        self.args = args
        self.save_path_sequence = args.save_folder
        self.method_name = args.method
        self.PD = mat_data['Pd']
        self.FA = mat_data['Fa']
        self.target = mat_data['target_num']
        self.H_size = mat_data['H_size']
        self.W_size = mat_data['W_size']

        if not hasattr(self, 'All_FA'):
            self.All_FA = self.FA
            self.All_PD = self.PD
            self.All_target = self.target

        else:
            self.All_FA = np.concatenate((self.All_FA, self.FA),axis=0)
            self.All_PD = np.concatenate((self.All_PD, self.PD),axis=0)
            self.All_target = np.concatenate((self.All_target, self.target),axis=0)

        if 'data' in args.sequence_name:
            if not hasattr(self, 'data_FA'):
                self.data_FA = self.FA
                self.data_PD = self.PD
                self.data_target = self.target
            else:
                self.data_FA = np.concatenate((self.data_FA, self.FA), axis=0)
                self.data_PD = np.concatenate((self.data_PD, self.PD), axis=0)
                self.data_target = np.concatenate((self.data_target, self.target), axis=0)
        if 'IRDST' in args.sequence_name:
            if not hasattr(self, 'IRDST_FA'):
                self.IRDST_FA = self.FA
                self.IRDST_PD = self.PD
                self.IRDST_target = self.target
            else:
                self.IRDST_FA = np.concatenate((self.IRDST_FA, self.FA), axis=0)
                self.IRDST_PD = np.concatenate((self.IRDST_PD, self.PD), axis=0)
                self.IRDST_target = np.concatenate((self.IRDST_target, self.target), axis=0)
        self.get('sequence')
    def update(self, preds, labels, args):
        if preds.shape == labels.shape and len(preds.shape) == 3:
            self.W_size = preds.shape[2]
            self.H_size = preds.shape[1]
            self.C_size = preds.shape[0]
        else:
            raise ValueError("Shapes of predictions and labels do not match.")
        if preds.dtype != np.uint8 and labels.dtype != np.uint8:
            raise TypeError("Preds and labels must be of uint8 dtype.")
        self.opts = dict()
        self.args = args
        self.image_area_total = []
        self.image_area_match = []
        self.FA = np.zeros([self.C_size, self.bins + 1])
        self.PD = np.zeros([self.C_size, self.bins + 1])
        self.target = np.zeros([self.C_size, self.bins + 1])

        for iBin in range(self.bins+1):
            score_thresh = iBin * (255/self.bins)
            predits  = np.array(preds > score_thresh).astype('int64')
            # predits = np.reshape(predits, (self.size, self.size))

            labelss = np.array(labels).astype('int64')
            # labelss = np.reshape(labelss, (self.size, self.size))
            for f in range(self.C_size):
                image = measure.label(predits[f], connectivity=2)
                coord_image = measure.regionprops(image)
                label = measure.label(labelss[f], connectivity=2)
                coord_label = measure.regionprops(label)

                self.target[f, iBin]    += len(coord_label)
                self.image_area_total = []
                self.image_area_match = []
                self.distance_match   = []
                self.dismatch         = []

                for K in range(len(coord_image)):
                    area_image = np.array(coord_image[K].area)
                    self.image_area_total.append(area_image)

                for i in range(len(coord_label)):
                    centroid_label = np.array(list(coord_label[i].centroid))
                    for m in range(len(coord_image)):
                        centroid_image = np.array(list(coord_image[m].centroid))
                        distance = np.linalg.norm(centroid_image - centroid_label)
                        area_image = np.array(coord_image[m].area)
                        if distance < 3:
                            self.distance_match.append(distance)
                            self.image_area_match.append(area_image)

                            del coord_image[m]
                            break

                self.dismatch = [x for x in self.image_area_total if x not in self.image_area_match]
                self.FA[f, iBin] = np.sum(self.dismatch)  # FA = FN      FP =
                self.PD[f, iBin] = len(self.distance_match)  # PD = TP   TN


        if not hasattr(self, 'All_FA'):
            self.All_FA = self.FA
            self.All_PD = self.PD
            self.All_target = self.target

        else:
            self.All_FA = np.concatenate((self.All_FA, self.FA),axis=0)
            self.All_PD = np.concatenate((self.All_PD, self.PD),axis=0)
            self.All_target = np.concatenate((self.All_target, self.target),axis=0)

        if 'data' in args.sequence_name:
            if not hasattr(self, 'data_FA'):
                self.data_FA = self.FA
                self.data_PD = self.PD
                self.data_target = self.target
            else:
                self.data_FA = np.concatenate((self.data_FA, self.FA), axis=0)
                self.data_PD = np.concatenate((self.data_PD, self.PD), axis=0)
                self.data_target = np.concatenate((self.data_target, self.target), axis=0)
        if 'IRDST' in args.sequence_name:
            if not hasattr(self, 'IRDST_FA'):
                self.IRDST_FA = self.FA
                self.IRDST_PD = self.PD
                self.IRDST_target = self.target
            else:
                self.IRDST_FA = np.concatenate((self.IRDST_FA, self.FA), axis=0)
                self.IRDST_PD = np.concatenate((self.IRDST_PD, self.PD), axis=0)
                self.IRDST_target = np.concatenate((self.IRDST_target, self.target), axis=0)



    def get(self,dataset_name):
        """
        Probability of detection (Pd)
        False-alarm rate (Fa)
        """
        if 'sequence' == dataset_name:
            target_num = np.sum(self.target, axis=0)
            Sequence_FA = np.sum(self.FA, axis=0) / (self.H_size * self.W_size * self.FA.shape[0])
            Sequence_PD = np.sum(self.PD, axis=0) / target_num
            self.opts['Pd'] = self.FA
            self.opts['Fa'] = self.PD
            self.opts['target_num'] = self.target
            self.opts['H_size'] = self.H_size
            self.opts['W_size'] = self.W_size

            save_path_name = os.path.join(self.args.save_path_sequence,'FAPD_curve_target_level.png')
            self.plot_FA_PD(Sequence_FA, Sequence_PD, save_path_name)
            return self.opts

        elif 'data' == dataset_name and hasattr(self, 'data_FA'):

            data_FA = np.sum(self.data_FA, axis=0) / (self.H_size * self.W_size * self.data_FA.shape[0])
            data_PD = np.sum(self.data_PD, axis=0) / np.sum(self.data_target, axis=0)
            self.opts['data_FA'] = data_FA
            self.opts['data_PD'] = data_PD

            save_path_name = os.path.join(self.args.save_path_method, 'dataset_FAPD_curve_target_level.png')
            self.plot_FA_PD(data_FA, data_PD, save_path_name)
            return self.opts

        elif 'IRDST' == dataset_name and hasattr(self, 'IRDST_FA'):
            IRDST_FA = np.sum(self.IRDST_FA, axis=0) / (self.H_size * self.W_size * self.IRDST_FA.shape[0])
            IRDST_PD = np.sum(self.IRDST_PD, axis=0) / np.sum(self.IRDST_target, axis=0)
            self.opts['IRDST_FA'] = IRDST_FA
            self.opts['IRDST_PD'] = IRDST_PD

            save_path_name = os.path.join(self.args.save_path_method, 'IRDSTset_FAPD_curve_target_level.png')
            self.plot_FA_PD(IRDST_FA, IRDST_PD, save_path_name)
            return self.opts

        elif 'SIRSTD' == dataset_name and hasattr(self, 'SIRSTD_FA'):
            SIRSTD_FA = np.sum(self.SIRSTD_FA, axis=0) / (self.H_size * self.W_size * self.SIRSTD_FA.shape[0])
            SIRSTD_PD = np.sum(self.SIRSTD_PD, axis=0) / np.sum(self.SIRSTD_target, axis=0)
            self.opts['SIRSTD_FA'] = SIRSTD_FA
            self.opts['SIRSTD_PD'] = SIRSTD_PD

            save_path_name = os.path.join(self.args.save_path_method, 'SIRSTDset_FAPD_curve_target_level.png')
            self.plot_FA_PD(SIRSTD_FA, SIRSTD_PD, save_path_name)
            return self.opts

        elif 'All' == dataset_name and hasattr(self, 'All_FA'):
            all_FA = np.sum(self.All_FA, axis=0) / (self.H_size * self.W_size * self.All_FA.shape[0])
            all_PD = np.sum(self.All_PD, axis=0) / np.sum(self.All_target, axis=0)
            self.opts['all_FA'] = all_FA
            self.opts['all_PD'] = all_PD
            save_path_name = os.path.join(self.args.save_path_method, 'All FAPD_curve_target_level.png')
            self.plot_FA_PD(all_FA, all_PD, save_path_name)
            return self.opts
        else:
            print("No dataset ")
            return 0, 0
    def plot_FA_PD(self,FA, PD, save_path_name):
        # plt.ion()
        # self.get()
        # 初始化一个空列表来存储分数阈值
        score_thresh = []
        for iBin in range(self.bins + 1):
            score_thresh.append(int(iBin * (255 / self.bins)))

        fig, ax = plt.subplots(num="FA PD Curve", figsize=(8, 6))
        ax.set_ylim(0, 1)
        bars = ax.bar(score_thresh, PD, width=(255 / self.bins), edgecolor="white", linewidth=0.7)

        # 在每个柱状图底部内部显示竖直文本，包括PD百分比和FA值（以科学计数法显示）
        for x, (bar, pd_value, fa_value) in zip(score_thresh, zip(bars, PD, FA)):
            plt.text(x, 0.01, f'PD: {pd_value * 100:.2f}%\nFA: {fa_value:.2e}',
                     ha='center', va='bottom', fontsize=11, rotation='vertical', color='red')

        # 设置 x 轴的刻度和标签
        plt.xticks(score_thresh, rotation=45)  # 设置 x 轴刻度为 score_thresh，并旋转标签
        ax.set_xlabel('Threshold', fontsize=11)
        ax.set_ylabel('PD and FA', fontsize=11)
        fig.savefig(save_path_name)
        plt.close('all')
        # ax.set_title('FA PD values')
        # return fig
    def plot_all_FA_PD(self):
        # plt.ion()
        # self.get()
        # 初始化一个空列表来存储分数阈值
        score_thresh = []
        for iBin in range(self.bins + 1):
            score_thresh.append(int(iBin * (255 / self.bins)))

        fig, ax = plt.subplots(num="All FA PD Curve", figsize=(8, 6))
        bars = ax.bar(score_thresh, self.Final_all_PD, width=(255 / self.bins), edgecolor="white", linewidth=0.7)

        # 在每个柱状图底部内部显示竖直文本，包括PD百分比和FA值（以科学计数法显示）
        for x, (bar, pd_value, fa_value) in zip(score_thresh, zip(bars, self.Final_all_PD, self.Final_all_FA)):
            plt.text(x, 0.01, f'All PD: {pd_value * 100:.2f}%\nAll FA: {fa_value:.2e}',
                     ha='center', va='bottom', fontsize=11, rotation='vertical', color='red')

        # 设置 x 轴的刻度和标签
        plt.xticks(score_thresh, rotation=45)  # 设置 x 轴刻度为 score_thresh，并旋转标签
        ax.set_xlabel('Threshold', fontsize=11)
        ax.set_ylabel('All PD and FA', fontsize=11)
        fig.savefig(os.path.join(self.args.save_path_method, 'All FAPD_curve_target_level.png'))
        plt.close('all')
        # ax.set_title('FA PD values')
        # return fig
    def _cal_auc(self, x, y):
        lent_t = x.size
        auc = 0
        auc += (y[0] + y[1]) * x[0] / 2
        for i in range(1,lent_t):
            auc += (y[i] + y[i-1]) * (x[i]-x[i-1]) / 2
        # auc += (y[-1] + y[-2]) * (x[-1]-x[-2]) / 2
        # auc = 0.5 * np.sum((y[1:] + y[:-1]) * np.diff(x))
        return round(auc, 4)

class Metric3DROC_pixel:
    def __init__(self):
        """total 是预测目标、背景、原图
            opts，是其它相关信息
            TPR (target-point-level)：只要目标区域检测到一个像素，便检测到目标
            FPR (pixel-level)：检测到的像素占总像素值比例

            """
        super(Metric3DROC_pixel, self).__init__()
    def reset(self):
        self.opts = dict()
    def resumes(self,mat_data, args):
        self.opts = dict()
        self.args = args
        self.save_path_sequence = args.save_folder
        self.method_name = args.method
        self.PD = mat_data['PD']
        self.PF = mat_data['PF']
        self.tau = mat_data['Tau']

        if not hasattr(self, 'all_PD'):
            self.all_PD = self.PD
            self.all_PF = self.PF

        else:
            self.all_PD = np.concatenate((self.all_PD, self.PD), axis=0)
            self.all_PF = np.concatenate((self.all_PF, self.PF), axis=0)

        if 'data' in args.sequence_name:
            if not hasattr(self, 'data_PF'):
                self.data_PF = self.PF
                self.data_PD = self.PD
            else:
                self.data_PF = np.concatenate((self.data_PF, self.PF), axis=0)
                self.data_PD = np.concatenate((self.data_PD, self.PD), axis=0)

        if 'IRDST' in args.sequence_name:
            if not hasattr(self, 'IRDST_PF'):
                self.IRDST_PF = self.PF
                self.IRDST_PD = self.PD
            else:
                self.IRDST_PF = np.concatenate((self.IRDST_PF, self.PF), axis=0)
                self.IRDST_PD = np.concatenate((self.IRDST_PD, self.PD), axis=0)

        # self.get('sequence')
    def update(self, preds, GT, args):
        "输入uint8 ---- float32"
        if preds.dtype != np.uint8 and GT.dtype != np.uint8:
            raise TypeError("Preds and labels must be of uint8 dtype.")
        self.args = args
        self.preds = preds.astype(np.float32) / 255.
        self.GT = GT.astype(np.float32) / 255.
        self.save_folder = args.save_folder
        self.sequence_name = args.sequence_name
        self.method_name = args.method_name
        self.opts = dict()
        self.tau0 = np.arange(0, 1.004, 0.004)
        self.tau = np.sort(self.tau0)[::-1]
        self.num_map = self.preds.shape[0]
        self.PD = np.zeros((self.num_map, len(self.tau)))
        self.PF = np.zeros((self.num_map, len(self.tau)))

        for k in range(self.num_map):
            for j, t in enumerate(self.tau):
                map = self.preds[k, :, :].copy()
                map[self.preds[k, :, :] >= t] = 1
                map[self.preds[k, :, :] < t] = 0
                label = self.GT[k, :, :].copy()
                label[self.GT[k, :, :] >= t] = 1
                label[self.GT[k, :, :] < t] = 0
                PD0 = np.sign(((map * label) > 0).sum())
                PF0 = map.sum() / label.size
                self.PD[k, j] = PD0
                self.PF[k, j] = PF0

        ## all images
        if not hasattr(self, 'all_PD'):
            self.all_PD = self.PD
            self.all_PF = self.PF

        else:
            self.all_PD = np.concatenate((self.all_PD, self.PD), axis=0)
            self.all_PF = np.concatenate((self.all_PF, self.PF), axis=0)

        if 'data' in args.sequence_name:
            if not hasattr(self, 'data_PF'):
                self.data_PF = self.PF
                self.data_PD = self.PD
            else:
                self.data_PF = np.concatenate((self.data_PF, self.PF), axis=0)
                self.data_PD = np.concatenate((self.data_PD, self.PD), axis=0)

        if 'IRDST' in args.sequence_name:
            if not hasattr(self, 'IRDST_PF'):
                self.IRDST_PF = self.PF
                self.IRDST_PD = self.PD
            else:
                self.IRDST_PF = np.concatenate((self.IRDST_PF, self.PF), axis=0)
                self.IRDST_PD = np.concatenate((self.IRDST_PD, self.PD), axis=0)


    def get(self,dataset_name):
        if 'sequence' == dataset_name and hasattr(self, 'PD'):
            sequence_PD = np.mean(self.PD, axis=0)
            sequence_PF = np.mean(self.PF, axis=0)
            self.opts['PD'] = self.PD
            self.opts['PF'] = self.PF
            self.opts['Tau'] = self.tau
            self.opts['PD_mean_1'] = sequence_PD
            self.opts['PF_mean_1'] = sequence_PF
            self.opts['auc_PF_PD'] = auc(sequence_PF, sequence_PD)
            self.opts['auc_tau_PD'] = auc(self.tau, sequence_PD)
            self.opts['auc_tau_PF'] = auc(self.tau, sequence_PF)
            self.opts['auc_TD'] = self.opts['auc_PF_PD'] + self.opts['auc_tau_PD']
            self.opts['auc_BS'] = self.opts['auc_PF_PD'] - self.opts['auc_tau_PF']
            self.opts['auc_SNPR'] = self.opts['auc_tau_PD'] / self.opts['auc_tau_PF']
            self.opts['auc_TDBS'] = self.opts['auc_tau_PD'] - self.opts['auc_tau_PF']
            self.opts['auc_OD'] = self.opts['auc_PF_PD'] + self.opts['auc_tau_PD'] - self.opts['auc_tau_PF']
            self.opts['auc_ODP'] = self.opts['auc_tau_PD'] + (1 - self.opts['auc_tau_PF'])
            save_path_dataset = os.path.join(self.args.save_folder, self.sequence_name)
            self.plot_3DROC( sequence_PD, sequence_PF, self.tau, self.opts['auc_PF_PD'], self.opts['auc_tau_PD'],
                             self.opts['auc_tau_PF'], self.method_name, save_path_dataset)
            return self.opts

        elif 'data' == dataset_name and hasattr(self, 'data_PD'):
            data_PD = np.mean(self.data_PD, axis=0)
            data_PF = np.mean(self.data_PF, axis=0)
            self.opts['data_PD_mean_1'] = data_PD
            self.opts['data_PF_mean_1'] = data_PF
            self.opts['Tau'] = self.tau
            self.opts['data_auc_PF_PD'] = auc(data_PF, data_PD)
            self.opts['data_auc_tau_PD'] = auc(self.tau, data_PD)
            self.opts['data_auc_tau_PF'] = auc(self.tau, data_PF)
            self.opts['data_auc_TD'] = self.opts['data_auc_PF_PD'] + self.opts['data_auc_tau_PD']
            self.opts['data_auc_BS'] = self.opts['data_auc_PF_PD'] - self.opts['data_auc_tau_PF']
            self.opts['data_auc_SNPR'] = self.opts['data_auc_tau_PD'] / self.opts['data_auc_tau_PF']
            self.opts['data_auc_TDBS'] = self.opts['data_auc_tau_PD'] - self.opts['data_auc_tau_PF']
            self.opts['data_auc_OD'] = self.opts['data_auc_PF_PD'] + self.opts['data_auc_tau_PD'] - self.opts['data_auc_tau_PF']
            self.opts['data_auc_ODP'] = self.opts['data_auc_tau_PD'] + (1 - self.opts['data_auc_tau_PF'])
            save_path_dataset = os.path.join(self.args.save_folder, dataset_name)
            self.plot_3DROC(data_PD, data_PF, self.tau, self.opts['data_auc_PF_PD'], self.opts['data_auc_tau_PD'],
                            self.opts['data_auc_tau_PF'], self.method_name, save_path_dataset)
            return self.opts

        elif 'IRDST' == dataset_name and hasattr(self, 'IRDST_PD'):
            IRDST_PD = np.mean(self.IRDST_PD, axis=0)
            IRDST_PF = np.mean(self.IRDST_PF, axis=0)
            self.opts['IRDST_PD_mean_1'] = IRDST_PD
            self.opts['IRDST_PF_mean_1'] = IRDST_PF
            self.opts['Tau'] = self.tau
            self.opts['IRDST_auc_PF_PD'] = auc(IRDST_PF, IRDST_PD)
            self.opts['IRDST_auc_tau_PD'] = auc(self.tau, IRDST_PD)
            self.opts['IRDST_auc_tau_PF'] = auc(self.tau, IRDST_PF)
            self.opts['IRDST_auc_TD'] = self.opts['IRDST_auc_PF_PD'] + self.opts['IRDST_auc_tau_PD']
            self.opts['IRDST_auc_BS'] = self.opts['IRDST_auc_PF_PD'] - self.opts['IRDST_auc_tau_PF']
            self.opts['IRDST_auc_SNPR'] = self.opts['IRDST_auc_tau_PD'] / self.opts['IRDST_auc_tau_PF']
            self.opts['IRDST_auc_TDBS'] = self.opts['IRDST_auc_tau_PD'] - self.opts['IRDST_auc_tau_PF']
            self.opts['IRDST_auc_OD'] = self.opts['IRDST_auc_PF_PD'] + self.opts['IRDST_auc_tau_PD'] - self.opts[
                'IRDST_auc_tau_PF']
            self.opts['IRDST_auc_ODP'] = self.opts['IRDST_auc_tau_PD'] + (1 - self.opts['IRDST_auc_tau_PF'])
            save_path_dataset = os.path.join(self.args.save_folder, dataset_name)
            self.plot_3DROC(IRDST_PD, IRDST_PF, self.tau, self.opts['IRDST_auc_PF_PD'], self.opts['IRDST_auc_tau_PD'],
                            self.opts['IRDST_auc_tau_PF'], self.method_name, save_path_dataset)
            return self.opts


        elif 'All' == dataset_name and hasattr(self, 'all_PD'):
            all_PD = np.mean(self.all_PD, axis=0)
            all_PF = np.mean(self.all_PF, axis=0)
            self.opts['All_PD'] = all_PD
            self.opts['All_PF'] = all_PF
            self.opts['All_Tau'] = self.tau
            self.opts['All_auc_PF_PD'] = auc(all_PF, all_PD)
            self.opts['All_auc_tau_PD'] = auc(self.tau, all_PD)
            self.opts['All_auc_tau_PF'] = auc(self.tau, all_PF)
            self.opts['All_auc_TD'] = self.opts['All_auc_PF_PD'] + self.opts['All_auc_tau_PD']
            self.opts['All_auc_BS'] = self.opts['All_auc_PF_PD'] - self.opts['All_auc_tau_PF']
            self.opts['All_auc_SNPR'] = self.opts['All_auc_tau_PD'] / self.opts['All_auc_tau_PF']
            self.opts['All_auc_TDBS'] = self.opts['All_auc_tau_PD'] - self.opts['All_auc_tau_PF']
            self.opts['All_auc_OD'] = self.opts['All_auc_PF_PD'] + self.opts['All_auc_tau_PD'] - self.opts['All_auc_tau_PF']
            self.opts['All_auc_ODP'] = self.opts['All_auc_tau_PD'] + (1 - self.opts['All_auc_tau_PF'])
            save_path_dataset = os.path.join(self.args.save_folder, dataset_name)
            self.plot_3DROC(all_PD, all_PF, self.tau, self.opts['All_auc_PF_PD'],
                            self.opts['All_auc_tau_PD'],
                            self.opts['All_auc_tau_PF'], self.method_name, save_path_dataset)
            return self.opts
        else:
            self.opts['None_this_dataset'] = 1
            return self.opts
    def _generate_gt(self, image_size, gts):
        n3 = len(gts)
        GT_map = np.zeros(image_size, dtype=np.uint8)
        SE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        for i in range(n3):
            gt = np.array(gts[i]).reshape(-1, 2)
            for x, y in gt:
                GT_map[i, x, y] = 1
            GT_map[i, :, :] = cv2.dilate(GT_map[i, :, :], SE)
        return GT_map


    def _cal_auc(self, x, y):
        lent_t = x.size
        auc = 0
        auc += (y[0] + y[1]) * x[0] / 2
        for i in range(1,lent_t):
            auc += (y[i] + y[i-1]) * (x[i]-x[i-1]) / 2
        return round(auc, 4)

    def plot_3DROC(self, PD, PF, tau, auc_PF_PD, auc_tau_PD, auc_tau_PF, method_name, save_path_sequence):
        # Plot PF vs PD
        plt.ioff()
        plt.figure(figsize=(5, 5))
        plt.plot(PF, PD, linewidth=2)
        plt.ylim([-0.05, 1.05])
        plt.xlim([-0.05, 1.05])
        plt.xlabel('$P_F$', fontsize=18)
        plt.ylabel('$P_D$', fontsize=18)
        plt.grid(True)
        plt.box(True)
        plt.title(f"{method_name}  auc = {auc_PF_PD:.4f}")
        save_path = f"{save_path_sequence}_PF_PD.jpg"
        plt.savefig(save_path)
        # plt.close()

        # Plot Tau vs PD
        plt.figure(figsize=(5, 5))
        plt.plot(tau, PD, linewidth=2)
        plt.ylim([-0.05, 1.05])
        plt.xlim([-0.05, 1.05])
        plt.xlabel(r'$\tau$', fontsize=18)
        plt.ylabel('$P_D$', fontsize=18)
        plt.grid(True)
        plt.box(True)
        plt.title(f"{method_name}  auc = {auc_tau_PD:.4f}")
        save_path = f"{save_path_sequence}_PD_tau.jpg"
        plt.savefig(save_path)
        # plt.close()

        # Plot Tau vs PF
        plt.figure(figsize=(5, 5))
        plt.plot(tau, PF, linewidth=2)
        plt.ylim([-0.05, 1.05])
        plt.xlim([-0.05, 1.05])
        plt.xlabel(r'$\tau$', fontsize=18)
        plt.ylabel('$P_F$', fontsize=18)
        plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.9)
        plt.box(True)
        plt.title(f"{method_name}  auc = {auc_tau_PF:.4f}")
        save_path = f"{save_path_sequence}_PF_tau.jpg"
        plt.savefig(save_path)

        # plt.close()

        # 3D ROC
        fig = plt.figure(frameon=True)
        ax = fig.add_axes(Axes3D(fig))

        # 设置三维图形背景颜色（r,g,b,a）
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

        ax.plot(PF, tau, PD, linewidth=2)
        ax.view_init(elev=12., azim=-128)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_zlim([0, 1])
        ax.set_xlabel(r'$P_F$', fontsize=18)
        ax.set_ylabel(r'$\tau$', fontsize=18)
        ax.set_zlabel(r'$P_D$', fontsize=18)
        ax.xaxis._axinfo["grid"].update({"linewidth": 1, "color": "green"})
        ax.grid(color='red', linestyle='--', linewidth=0.05, alpha=0.07)
        plt.title(f"{method_name}")
        save_path = f"{save_path_sequence}_3D_ROC.jpg"
        plt.savefig(save_path)
        # plt.show()
        plt.close('all')
    def plot_all_3DROC(self):
        # Plot PF vs PD
        plt.ioff()
        plt.figure(figsize=(5, 5))
        plt.plot(self.Final_all_PF, self.Final_all_PD, linewidth=2)
        plt.ylim([-0.05, 1.05])
        plt.xlim([-0.05, 1.05])
        plt.xlabel('All $P_F$', fontsize=18)
        plt.ylabel('All $P_D$', fontsize=18)
        plt.grid(True)
        plt.box(True)
        plt.title(f"{self.method_name}  auc = {self.opts['All_auc_PF_PD']:.4f}")
        plt.savefig(os.path.join(self.args.save_path_method,"All PF_PD.jpg"))
        # plt.close()

        # Plot Tau vs PD
        plt.figure(figsize=(5, 5))
        plt.plot(self.tau, self.Final_all_PD, linewidth=2)
        plt.ylim([-0.05, 1.05])
        plt.xlim([-0.05, 1.05])
        plt.xlabel(r'$\tau$', fontsize=18)
        plt.ylabel('All $P_D$', fontsize=18)
        plt.grid(True)
        plt.box(True)
        plt.title(f"{self.method_name}  auc = {self.opts['All_auc_tau_PD']:.4f}")
        plt.savefig(os.path.join(self.args.save_path_method,"All PD_tau.jpg"))
        # plt.close()

        # Plot Tau vs PF
        plt.figure(figsize=(5, 5))
        plt.plot(self.tau, self.Final_all_PF, linewidth=2)
        plt.ylim([-0.05, 1.05])
        plt.xlim([-0.05, 1.05])
        plt.xlabel(r'$\tau$', fontsize=18)
        plt.ylabel('All $P_F$', fontsize=18)
        plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.9)
        plt.box(True)
        plt.title(f"{self.method_name}  auc = {self.opts['All_auc_tau_PF']:.4f}")
        plt.savefig(os.path.join(self.args.save_path_method,"All PF_tau.jpg"))

        # plt.close()

        # 3D ROC
        fig = plt.figure(frameon=True)
        ax = fig.add_axes(Axes3D(fig))

        # 设置三维图形背景颜色（r,g,b,a）
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

        ax.plot(self.Final_all_PF, self.tau, self.Final_all_PD, linewidth=2)
        ax.view_init(elev=12., azim=-128)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_zlim([0, 1])
        ax.set_xlabel(r'All $P_F$', fontsize=18)
        ax.set_ylabel(r'$\tau$', fontsize=18)
        ax.set_zlabel(r'All $P_D$', fontsize=18)
        ax.xaxis._axinfo["grid"].update({"linewidth": 1, "color": "green"})
        ax.grid(color='red', linestyle='--', linewidth=0.05, alpha=0.07)
        plt.title(f"{self.method_name}")
        plt.savefig(os.path.join(self.args.save_path_method,"All 3D_ROC.jpg"))
        # plt.show()
        plt.close('all')


class mIoU():
    " pixel-level"
    def __init__(self):
        super(mIoU, self).__init__()
        self.nclass = 1
        self.reset()

    def update(self, preds, labels):
        # print('come_ininin')

        correct, labeled = batch_pix_accuracy(preds, labels)
        inter, union = batch_intersection_union(preds, labels, self.nclass)
        self.total_correct += correct
        self.total_label += labeled
        self.total_inter += inter
        self.total_union += union


    def get(self):

        pixAcc = 1.0 * self.total_correct / (np.spacing(1) + self.total_label)
        IoU = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
        mIoU = IoU.mean()
        return pixAcc, mIoU

    def reset(self):

        self.total_inter = 0
        self.total_union = 0
        self.total_correct = 0
        self.total_label = 0


def cal_tp_pos_fp_neg(output, label, nclass, score_thresh):
    predict = (output >= score_thresh)
    target = (label > 0)


    intersection = predict * (predict == target)

    tp = intersection.sum()
    fp = (predict * ((predict != target))).sum()
    tn = ((1 - predict) * ((predict == target))).sum()
    fn = (((predict != target)) * (1 - predict)).sum()
    pos = tp + fn
    neg = fp + tn
    class_pos = tp + fp

    return tp, pos, fp, neg, class_pos

def batch_pix_accuracy(output, target):
    threshold = 0.55 * 255
    assert output.shape == target.shape, "Predict and Label Shape Don't Match"
    predict_bool = (output > threshold)
    label_bool = (target > threshold)
    pixel_labeled = label_bool.sum()
    pixel_correct = ((predict_bool == label_bool)*(target > threshold)).sum()
    assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"
    return pixel_correct, pixel_labeled


def batch_intersection_union(output, target, nclass):

    threshold = 0.55 * 255
    predict_bool = (output > threshold)
    target_bool = (target > threshold)
    intersection = target_bool * ((predict_bool == target_bool))
    area_inter = (intersection).sum()
    area_union = ((target_bool + predict_bool) > 0).sum()
    assert (area_inter <= area_union).all(), \
        "Error: Intersection area should be smaller than Union area"
    return area_inter, area_union
def load_images_from_folder(folder):
    # 获取所有以 b.bmp 结尾的文件路径
    file_pattern = os.path.join(folder, '*.bmp')
    files0 = glob.glob(file_pattern)
    files = sorted(files0, key=os.path.getmtime)
    # 获取图像数量
    num_images = len(files)
    if num_images == 0:
        raise ValueError("No images found in the specified folder.")

    # 创建一个空的 NumPy 数组来存储所有图像，假设图像是灰度图像
    preds = np.zeros((num_images, 256, 256), dtype=np.uint8)

    for i, file in enumerate(files):
        # 打开图像
        image = Image.open(file)
        # 将图像转换为NumPy数组
        image_np = np.array(image)
        # 确保图像是灰度图像
        if image_np.ndim == 3 and image_np.shape[2] == 3:
            image_np = np.mean(image_np, axis=2).astype(np.uint8)
        # 将图像数组存储在预定义的 NumPy 数组中
        preds[i, :, :] = image_np
        print(f"Loaded image and converted to array: {file}")


    return preds

def read_images_from_folder(folder_path):
    # 获取所有以 b.bmp 结尾的文件路径
    file_pattern = os.path.join(folder_path, '*.png')
    files0 = glob.glob(file_pattern)
    files = natsort.natsorted(files0)
    # 获取图像数量
    num_images = len(files)
    if num_images == 0:
        raise ValueError("No images found in the specified folder.")

    # 创建一个空的 NumPy 数组来存储所有图像，假设图像是灰度图像
    preds = np.zeros((num_images, 480, 720), dtype=np.uint8)

    for i, file in enumerate(files):
        image = Image.open(file)
        # 将图像转换为NumPy数组
        image_np = np.array(image)
        # 确保图像是灰度图像
        if image_np.ndim == 3 and image_np.shape[2] == 3:
            image_np = np.mean(image_np, axis=2).astype(np.uint8)
        # 将图像数组存储在预定义的 NumPy 数组中
        preds[i, :, :] = image_np
        print(f"Loaded image and converted to array: {file}")

    return preds
def display_images(images):
    num_images = len(images)
    cols = math.ceil(math.sqrt(num_images))  # 列数
    rows = math.ceil(num_images / cols)      # 行数
    # plt.ion()
    fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
    axes = axes.flatten()  # 将二维数组展平成一维数组以便迭代

    for ax, img in zip(axes, images):
        ax.imshow(img)
        ax.axis('off')
        for spine in ax.spines.values():
            spine.set_edgecolor('black')  # 设置边框颜色
            spine.set_linewidth(1)  # 设置边框宽度

    # 隐藏多余的子图
    for ax in axes[len(images):]:
        ax.axis('off')

    plt.subplots_adjust(wspace=0.001, hspace=0.01)  # 调整子图间的空间
    # plt.show()

