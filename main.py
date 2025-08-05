import re
import os
import time
import scipy.io as scio
from numpy import random
from tqdm import tqdm
from torch.utils.data import DataLoader
from pprint import pformat
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


from utils.common import *
from datas.dataset import Ours_v2_mat
from configs import Config, Config_3DSTPM
from metrics import metrics_mshnet
from utils.logger import setup_logger
from models.htr_sparseatt import HTR_SparseAtt

# 设置环境变量
os.environ["MPLBACKEND"] = "Agg"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# 打印设置，防止自动换行
np.set_printoptions(linewidth=np.inf)   # 控制每行的最大字符数，确保不会换行
torch.set_printoptions(linewidth=1000)  # 控制每行的最大字符数，确保不会换行

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def sort_key(file_name):
    # 提取字母部分和数字部分
    letter_part = re.findall(r'[a-zA-Z]+', file_name)[0]
    number_part = int(re.search(r'\d+', file_name).group())
    return (letter_part.lower(), number_part)

def Print_loss(name, RMES, MAE, MAPE, RSE, m):
    print("%s_dataset sum loss: RMSE:%f, MAE:%f, MAPE:%f, RSE:%f" % (name, RMES, MAE / m, MAPE / m, RSE / m))


def main():

    # 设备和随机种子
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    #----------------------------------------------configuration------------------------------------------------#
    args = Config()
    seed_torch(args.seed)
    torch.backends.cudnn.deterministic = True

    # 日志设置
    logger = setup_logger(args.method_name, args.save_folder, 0, filename='log.txt')
    logger.info("Arguments:\n{}".format(pformat(vars(args))))

    # 数据集路径和文件夹排序
    seqences_dir = os.path.join(args.dataset_path, args.Dataset_name)
    foldernames = sorted(os.listdir(seqences_dir), key=sort_key)[0:4]

    # -----------------------------评价指标 初始化----------------------------------------#
    Metric3DROC = metrics_mshnet.Metric3DROC_pixel()
    num0 = 0
    # --------------------------------开始序列图像---------------------------------------------#
    for foldername in foldernames:
        num0 += 1
        args.sequence_name = foldername
        logger.info(f"\n {'-' * 100}\n Sequence: {foldername}({num0}/{len(foldernames)})")
        # --------------------------------载入数据---------------------------------------------#
        seed_torch(args.seed)
        collate_fn = get_collate_fn_pad(args.batch_size, args.frame_input)
        Sequences_data = Ours_v2_mat(seqences_dir, foldername, args)
        IR_loader_train = DataLoader(dataset=Sequences_data, batch_size=args.batch_size * args.frame_input, shuffle=False, drop_last=False, collate_fn=collate_fn)
        kk = 0
        # 进度条
        tbar = tqdm(IR_loader_train, position=0)
        for ii, ((images, labels, _), meta) in enumerate(tbar):
            valid_size = meta['valid_size']
            tensor = images.to(device).float()
            model = HTR_SparseAtt(tensor, args).to(device)

            outs_B00 = tensor
            start_time = time.time()
            #------------------------------------iteration----------------------------------#
            for i in range(0, args.iteration):
                model(tensor)
                outs_B0, outs_T0 = model.get()

                # ---------------------------------误差评估----------------------------------------#
                MRE = torch.norm(tensor - outs_B0,2) / torch.norm(tensor,2)
                MRE0 = torch.norm(tensor - outs_B0-outs_T0, 2) / torch.norm(tensor, 2)
                ie = torch.norm(outs_B00 - outs_B0, 2)
                outs_B00 = outs_B0

                tbar.set_description("Sequence:%s(%1d/%1d),  images:%1d-%1d, iteration:%3d,  MRE:%.6f,  MSE0:%.6f  , IE:%.6f" % (
                    foldername, num0, foldernames.__len__(),ii * args.batch_size*args.frame_input, ii * args.batch_size*args.frame_input + args.batch_size*args.frame_input, i, MRE, MRE0, ie))
                if MRE < args.convergence:
                    break
            end_time = time.time()
            logger.info("Runntime=%3f F/s" % ((end_time-start_time)/(args.batch_size*args.frame_input)))

            outs_B1 = outs_B0[:valid_size].clone().cpu().detach().numpy()
            outs_T1 = outs_T0[:valid_size].clone().cpu().detach().numpy()

            #---------------------------------------past process ------------------------------------#
            outs_B = np_normalized(outs_B1) #   display_images(outs_B_)
            outs_T = post_processing_mean(outs_T1)

            labels = labels[:valid_size].cpu().detach().numpy()
            D = (images[:valid_size].cpu().detach().numpy() * 255).astype(np.uint8)

            if args.save_images:
                # ------------------------------------- save path------------------------------#
                result_dir = os.path.join(args.save_folder, foldername)
                # --------------------------------存储结果 - 图片 -------------------------------#
                if not os.path.exists(result_dir):
                    os.makedirs(result_dir)
                for k in range(outs_T.shape[0]):
                    kk += 1
                    plt.ioff()
                    # plt.ion()
                    plt.figure()
                    plt.subplot(221)
                    plt.imshow(D[k,:,:], cmap='gray')
                    plt.title('image')

                    plt.subplot(222)
                    plt.imshow(labels[k,:,:], cmap='gray')
                    plt.title('label')
                    # plt.show()

                    plt.subplot(223)
                    plt.imshow(outs_B[k,:,:], cmap='gray')
                    plt.title('Recovered:B')

                    plt.subplot(224)
                    plt.imshow(outs_T[k, :, :], cmap='gray')
                    plt.title(f'Target :T')
                    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.3)

                    plt.savefig(os.path.join(result_dir, str(kk) + ".jpg"), dpi=300)
                    plt.close()

            if ii == 0:
                image_all = D
                labels_all = labels
                outs_T_all = outs_T
                outs_B_all = outs_B

            else:
                image_all = np.concatenate((image_all, D), 0)
                labels_all = np.concatenate((labels_all, labels), 0)
                outs_T_all = np.concatenate((outs_T_all, outs_T), 0)
                outs_B_all = np.concatenate((outs_B_all, outs_B), 0)

        #---------------------------------  存储 opts- mat  ----------------------------------#
        Metric3DROC.update(outs_T_all, labels_all, args)
        logger.info(f"\n{'-' * 100}")
        tbar.close()

    # -------------------------------------- 所有结果  ----------------------------------#
    #  3DROC
    Metric3DROC.reset()
    data_opts = Metric3DROC.get('data')
    IRDST_opts = Metric3DROC.get('IRDST')
    All_opts = Metric3DROC.get('All')

    # save mat
    save_path_mat = os.path.join(args.save_folder, 'All_opts.mat')
    scio.savemat(save_path_mat, All_opts)
    formatted_opts = '\n'.join([f"{k}: {v}" for k, v in All_opts.items()])
    logger.info(f"Options:\n{formatted_opts}")

if __name__ == "__main__":
    total_start = time.time()  # 开始计时
    main()
    total_end = time.time()  # 结束计时
    print("程序总运行时间：%.2f 秒 (约 %.2f 分钟)" % (
        total_end - total_start, (total_end - total_start) / 60.0))