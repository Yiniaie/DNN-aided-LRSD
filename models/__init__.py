import torch
from utils.images_patch_tensor import gen_patch_tensor
from models.htr_sparseatt import HTR_SparseAtt



def model_chose(images, args, device):
    """
        根据 patch 设置处理图像，并根据 args.Mode 构造对应模型。

        参数:
            images (Tensor): 输入图像，形状为 [B, T, 1, H, W]
            args: 配置对象，包含 patch、Mode 等参数
            device: 当前运行的 device（如 'cuda'）

        返回:
            tensor (Tensor): 处理后的输入 tensor
            model (nn.Module): 初始化的模型
        """
    # 图像 patch 处理
    if args.patch:
        patch_tensor = gen_patch_tensor(images.cpu().detach().numpy(), args.patch_size, args.slide_step)
        tensor = torch.from_numpy(patch_tensor).to(device).float()  # [N, T, 1, H, W]
        tensor = tensor.permute(1, 0, 2, 3)  # 转为 [T, N, H, W]
    else:
        tensor = images.to(device).float()

    # 模型构造

    model = HTR_SparseAtt(tensor, args).to(device)



    return tensor, model