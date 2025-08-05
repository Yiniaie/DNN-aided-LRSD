import torch
import numpy as np
import torch.nn as nn
import concurrent.futures
import math
import torch.nn.functional as F
from utils.common import *

class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=False,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            # stdv = 1. / math.sqrt( self.linear.weight.size(1))
            # self.linear.weight.uniform_(-stdv, stdv)
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                            1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                            np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

class permute_change(nn.Module):
    def __init__(self, *dims):
        super(permute_change, self).__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(self.dims)

def siren_init(tensor: torch.Tensor, is_first: bool = False, omega_0: float = 30.0):
    """
    针对 SIREN 网络的初始化，适用于三维 TR 因子。
    tensor: 需要初始化的三维张量，形状通常是 [r, r, C]
    is_first: 是否为网络的第一层，第一层初始化范围更小
    omega_0: 正弦激活频率因子，默认30
    """
    with torch.no_grad():
        dim = tensor.size(-1)  # 通道数或最后一维大小
        if is_first:
            bound = 1.0 / dim
        else:
            bound = np.sqrt(6.0 / dim) / omega_0
        tensor.uniform_(-bound, bound)
def make_sine_layers(in_dim, out_dim, num_layers, hidden_dim, omega_0):
    layers = []


    if num_layers == 1:
        layers.append(SineLayer(in_dim, out_dim, is_first=True, omega_0=omega_0))
    else:
        for i in range(num_layers):
            if i == 0:
                layers.append(SineLayer(in_dim, hidden_dim, is_first=True, omega_0=omega_0))
            elif i == num_layers - 1:
                layers.append(SineLayer(hidden_dim, out_dim, is_first=False, omega_0=omega_0))
            else:
                layers.append(SineLayer(hidden_dim, hidden_dim, is_first=False, omega_0=omega_0))
    return layers

def TN_composition(G):
    N = len(G)
    m = 2
    n = 1
    Out = G[0]
    M = N
    for i in range(0, N - 1):
        Out = tensor_contraction(Out, G[i + 1], M, N, m, n)
        M = M + N - 2 * (i + 1)
        n = np.append(n, 1 + i + 1)
        tempm = 2 + (i + 1) * (N - i - 1)
        if i + 1 > 1:
            m[1:] = m[1:] - range(1, i + 1)
        m = np.append(m, tempm)

    return Out
def tensor_contraction(X, Y, Sx, Sy, n, m):
    Lx = X.shape
    Lx = np.array(Lx)
    Ly = Y.shape
    Ly = np.array(Ly)
    Nx = len(X.shape)
    Ny = len(Y.shape)
    if Nx < Sx:
        tempLx = torch.ones(1, Sx - Nx)
        Lx = np.append(Lx, tempLx)
    if Ny < Sy:
        tempLy = torch.ones(1, Sy - Ny)
        Ly = np.append(Ly, tempLy)

    indexx = [range(0, Sx)]
    indexy = [range(0, Sy)]
    # indexx[n] = []
    # indexy[m] = []
    indexx = np.delete(indexx, n - 1)
    indexy = np.delete(indexy, m - 1)

    tempX = torch.permute(X, tuple(np.append(indexx, n - 1)))
    tempXX = torch.reshape(tempX, (np.prod(Lx[indexx]), np.prod(Lx[n - 1])))

    tempY = torch.permute(Y, tuple(np.append(m - 1, indexy)))
    tempYY = torch.reshape(tempY, (np.prod(Ly[m - 1]), np.prod(Ly[indexy])))
    tempOut = torch.mm(tempXX, tempYY)

    Out = torch.reshape(tempOut, tuple(np.append(Lx[indexx], Ly[indexy])))
    return Out


class Nonlinear_TR(nn.Module):
    def __init__(self, images, args):
        super(Nonlinear_TR, self).__init__()
        r = args.rank
        self.rank = r
        expend = 1
        down = args.down
        omega_0 = args.omega_0
        C, H, W = images.size()
        C0, H0, W0 = int(C * down), int(H * down), int(W * down)
        C1, H1, W1 = int(C * expend), int(H * expend), int(W * expend)
        self.C, self.H, self.W = C, H, W
        num_layers = args.sine_layers

        rank = [[r, r, C0],
                [r, r, H0],
                [r, r, W0]
                ]
        # self.rank = rank
        self.relu = nn.LeakyReLU()
        # self.NB = nn.InstanceNorm2d(C)  # nn.InstanceNorm2d    nn.BatchNorm2d
        # self.L = nn.Linear(W, W)
        self.A_hat = nn.Parameter(torch.empty(rank[0]))
        # nn.init.kaiming_uniform_(self.A_hat)
        siren_init(self.A_hat, is_first=True, omega_0=30)

        self.B_hat = nn.Parameter(torch.empty(rank[1]))
        # nn.init.kaiming_uniform_(self.B_hat)
        siren_init(self.B_hat, is_first=True, omega_0=30)

        self.C_hat = nn.Parameter(torch.empty(rank[2]))
        # nn.init.kaiming_uniform_(self.C_hat)
        siren_init(self.C_hat, is_first=True, omega_0=30)

        self.A_net = nn.Sequential(
            *make_sine_layers(C0, C, num_layers, C1, omega_0),
            permute_change(2, 0, 1),
        )

        self.B_net = nn.Sequential(
            *make_sine_layers(H0, H, num_layers, H1, omega_0),
            permute_change(1, 2, 0),
        )

        self.C_net = nn.Sequential(
            *make_sine_layers(W0, W, num_layers, W1, omega_0),
            # 如果需要变换，也可加 permute_change(...)
        )

    def forward(self, x):
        # end_time2 = time.time()
        A = self.A_net(self.A_hat)
        B = self.B_net(self.B_hat)
        C = self.C_net(self.C_hat)
        outs_B = TN_composition([A, B, C])
        return self.relu(outs_B)

class soft(nn.Module):
    def __init__(self):
        super(soft, self).__init__()

    def forward(self, x, lam):
        x_abs = x.abs() - lam
        zeros = x_abs - x_abs
        n_sub = torch.max(x_abs, zeros)
        x_out = torch.mul(torch.sign(x), n_sub)
        return x_out


class LambdaLayer(nn.Module):
    def __init__(self, func):
        super(LambdaLayer, self).__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)

class Top_Hat_attention(nn.Module):
    '''
    感觉理解好注意力机制，还是得从mlp -> rnn -> gru -> seq2seq -> encoder-decoder -> attention ->mha ->transformer的顺序，以及序列的角度去理解
    # 初始化, in_channel代表输入特征图的通道数, ratio代表第一个全连接下降通道的倍数
    '''
    def __init__(self, in_channel, ratio=4):
        # 继承父类初始化方法
        super(Top_Hat_attention, self).__init__()
        kernel_sizes = [5, 9]  # 7
        # kernel_sizes = [7]
        stride = 1
        # 属性分配
        # 最大池化，输出的特征图的HW宽高
        self.scales_WTHAM = []
        self.scales_WTHAM = nn.ModuleList([self._make_scale_WTHAM(in_channel, size) for size in kernel_sizes])

        self.scales_BTHAM = []
        self.scales_BTHAM = nn.ModuleList([self._make_scale_BTHAM(in_channel, size) for size in kernel_sizes])
        # 第一个全连接层将特征图的通道数下降4倍
        # self.fc1 = nn.Linear(in_features=in_channel, out_features=in_channel // ratio, bias=False)
        # relu激活
        self.ReLU = nn.ReLU()
        self.InstanceNorm2d = nn.InstanceNorm2d(in_channel)
        # sigmoid激活函数，将权值归一化到0-1
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        # self.TopHatTransform = TopHatTransform(10,5)
        self.weight_net = nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1, groups=in_channel)
        self.Temporal_cross = nn.Sequential(nn.Conv3d(1, 3, kernel_size=3, stride=1, padding=1),
                                            nn.InstanceNorm2d(in_channel),
                                            nn.ReLU(),
                                            nn.Conv3d(3, 1, kernel_size=3, stride=1, padding=1),
                                            )
        # 初始化
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():  # 继承nn.Module的方法
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            if isinstance(m, nn.Linear):
                # 计算stdv
                stdv = 1. / math.sqrt(m.weight.size(1))
                m.weight.data.uniform_(-stdv, stdv)

    def _make_scale_WTHAM(self, in_channel, kernel_size):
        WTHAM = nn.Sequential(nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2),
                              LambdaLayer(lambda x: -x),
                              nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2),
                              LambdaLayer(lambda x: -x),
                              # nn.AvgPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2),
                              nn.Conv2d(in_channel, in_channel, kernel_size=kernel_size, stride=1,
                                        padding=kernel_size // 2, groups=in_channel),
                              nn.InstanceNorm2d(in_channel),
                              nn.ReLU(),
                              )
        return WTHAM

    def _make_weight(self, x1, x2):
        weight_feature = self.InstanceNorm2d(x1 - x2)
        # 假设 weight_feature 的形状是 (1, 10, 256, 256)
        # mean_values = weight_feature.mean(dim=(2, 3), keepdim=True)
        # weight_feature = torch.where(weight_feature < mean_values, torch.tensor(0.0).to(weight_feature.device), weight_feature)
        weight_feature = self.weight_net(weight_feature)  # 先通过 weight_net

        return weight_feature

    def _make_scale_BTHAM(self, in_channel, kernel_size):
        BTHAM = nn.Sequential(  # nn.AvgPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size//2),
            LambdaLayer(lambda x: -x),
            nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2),
            LambdaLayer(lambda x: -x),
            nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2),
            nn.Conv2d(in_channel, in_channel, kernel_size=kernel_size, stride=1, padding=kernel_size // 2,
                      groups=in_channel),
            nn.InstanceNorm2d(in_channel),
            nn.ReLU(),
        )
        return BTHAM
        # 前向传播

    def forward(self, x0):  # inputs 代表输入特征图
        if x0.dim() == 2:
            x = x0.unsqueeze(0).unsqueeze(0)
        elif x0.dim() == 3:
            x = x0.unsqueeze(0)
        elif x0.dim() == 4:
            x = x0
        else:
            print("input 维度不匹配")
        # elif x.dim() == 4:
        # 获取输入特征图的shape
        # x[x<=0]=0
        # x = F.conv2d(x, self.kernel, padding=2, groups=x.size(1))
        b, c, h, w = x.shape
        # White top-hat transformation
        multi_scales_WTHAM = [F.interpolate(input=stage(x), size=(h, w), mode='bilinear', align_corners=False) for stage
                              in self.scales_WTHAM]
        weights_WTHAM = [self._make_weight(x, scale_feature) for scale_feature in multi_scales_WTHAM]

        # Black top-hat transformation
        multi_scales_BTHAM = [F.interpolate(input=stage(x), size=(h, w), mode='bilinear', align_corners=False) for stage
                              in self.scales_BTHAM]
        weights_BTHAM = [self._make_weight(scale_feature, x) for scale_feature in multi_scales_BTHAM]

        combined_weights_WTHAM = torch.sum(torch.stack(weights_WTHAM), dim=0)
        combined_weights_BTHAM = torch.sum(torch.stack(weights_BTHAM), dim=0)

        # temporal-cross
        conv_output = combined_weights_BTHAM + combined_weights_WTHAM
        conv_output = self.Temporal_cross(conv_output)

        y0 = self.sigmoid(conv_output)
        y = self.ReLU(x * y0)
        if x0.dim() == 2:
            y = y.squeeze(0).squeeze(0)

        elif x0.dim() == 3:
            y = y.squeeze(0)
        elif x0.dim() == 4:
            y = y
        return y


class HTR_SparseAtt(nn.Module):
    """
    ###########################
    影响性能的网络参数：
    1. 初始化U和V的维度大小
    2. 学习率
    3. 最大迭代次数
    4. TR
    5.
    ###########################
    存在问题：
    1. T 随着迭代越来越稀疏，直至消失
    """
    def __init__(self, images, args):
        super(HTR_SparseAtt,self).__init__()
        self.images = images.clone().detach()
        if images.dim() == 3:
            self.C, self.H, self.W = images.shape
        elif images.dim() == 4:
            self.C, self.P, self.H, self.W = images.shape
        self.F_norm = torch.nn.MSELoss()
        self.lambda1 = args.lambda1
        self.lambda2 = args.lambda2
        self.lambda3 = args.lambda3
        self.lambda4 = args.lambda4
        self.lambda5 = args.lambda5
        self.mu1 = args.mu1
        self.lr = args.lr

        # self.priorWeight = torch.tensor(postprocess_tophat0(self.images.cpu().numpy()), dtype=torch.float).cuda()
        self.Q = torch.zeros(images.shape).cuda()
        self.E = torch.zeros(images.shape).cuda()

        self.iter = 0
        self.soft_thres = soft()
        self.Background_module = Nonlinear_TR(images, args)
        self.Target_module =  Top_Hat_attention(self.C)

        # 获取两个模块的参数
        self.background_params = list(self.Background_module.parameters())
        self.target_params = list(self.Target_module.parameters())

        # 合并参数
        self.params = self.background_params + self.target_params
        self.opt = torch.optim.Adam(self.params, lr=self.lr)
        # self.opt = torch.optim.RMSprop(self.params, lr=0.0002)


    def forward(self,images):
        self.opt.zero_grad()
        total_loss = 0
        self.iter += 1

        self.outs_B = self.Background_module(self.images)

        ##---------------------------  updating T  Sparse target Attention Module (CBAM)
        self.outs_T0 = self.images - self.outs_B
        self.outs_T = self.Target_module(self.outs_T0)

        ##---------------------------Adam  Loss ------------------------------##
        total_loss += (self.lambda3 / 2) * torch.norm(self.images- self.outs_B - self.outs_T, 2)
        total_loss += (self.mu1 / 2) * torch.norm(self.images - self.outs_B - self.E + self.Q / self.mu1,2)


        ##----------------------------ADMM  optimation ----------------------------------##
        self.outs_T_ = self.outs_T.clone().detach()
        self.outs_B_ = self.outs_B.clone().detach()

        self.E = self.soft_thres(self.images - self.outs_B_ + self.Q / self.mu1, self.lambda2 / self.mu1).clone().detach()
        self.Q = (self.Q + self.mu1 * (self.images - self.outs_B_ - self.E)).clone().detach()

        # --------------------    2 并行计算 TR 核范数  -------------------------------
        try:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(compute_nuc_norms_CPU, self.lambda1, self.Background_module.A_hat, self.Background_module.B_hat, self.Background_module.C_hat, self.iter)
                ]
                total_loss += sum(future.result() for future in concurrent.futures.as_completed(futures))

        except Exception:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(compute_nuc_norms_CPU, self.lambda1, self.Background_module.A_hat, self.Background_module.B_hat, self.Background_module.C_hat, self.iter + 1)
                ]
                total_loss += sum(future.result() for future in concurrent.futures.as_completed(futures))

        #------------------------ 梯度反向传播 Backpropagation----------------------
        total_loss.backward()
        # 更新参数
        self.opt.step()
    def get(self):

        return self.outs_B, self.outs_T
    def __str__(self):
        info = "TRTNNet Model Attributes:\n"
        for key, value in self.__dict__.items():
            if key == "background_params" or key == "target_params" or key == "params":
                continue  # 如果需要也可以排除 Target_module
            elif key == "Target_module" or key == "Background_module" or key == 'images':
                continue  # 如果需要也可以排除 Target_module
            else:
                info += f"{key}: {value}\n"
        return info