import numpy as np
import scipy.io as sio
import cv2
import os
import os.path as osp
import matplotlib.pyplot as plt
import torch
def gen_patch_tensor(images, patch_size=50, slide_step=50):
    """
    Generate patches from a 3D array of images and concatenate them into a 4D array.

    Parameters:
        images (numpy.ndarray): 3D array of images with shape (num_images, height, width).
        patch_size (int): Size of the patches to extract.
        slide_step (int): Step size for sliding window.

    Returns:
        patch_ten (numpy.ndarray): 4D array of patches with shape (num_images, num_patches, patch_size, patch_size).
    """

    num_images, img_hei, img_wid = images.shape

    row_patch_num = int(np.ceil((img_hei - patch_size) / slide_step)) + 1
    col_patch_num = int(np.ceil((img_wid - patch_size) / slide_step)) + 1

    num_patches = row_patch_num * col_patch_num
    patch_ten = np.zeros((num_images, num_patches, patch_size, patch_size))
    patch_image = np.zeros((num_patches, patch_size, patch_size))
    row_pos_arr = list(range(0, (row_patch_num - 1) * slide_step, slide_step)) + [img_hei - patch_size]
    col_pos_arr = list(range(0, (col_patch_num - 1) * slide_step, slide_step)) + [img_wid - patch_size]
    for i in range(0, num_images):
        j =0
        for col in col_pos_arr:
            for row in row_pos_arr:
                tmp_patch = images[i, row:row + patch_size, col:col + patch_size]
                patch_image[j,:,:] = tmp_patch
                j += 1
        patch_ten[i,:] = patch_image
    return patch_ten


def reconstruct_images(patch_ten, original_shape, patch_size=50, slide_step=50):
    """
    Reconstruct the original images from the 4D array of patches.

    Parameters:
        patch_ten (numpy.ndarray): 4D array of patches with shape (num_images*num_patches, patch_size, patch_size).
        original_shape (tuple): The original shape of the images (num_images, height, width).
        patch_size (int): Size of the patches to extract.
        slide_step (int): Step size for sliding window.

    Returns:
        images (numpy.ndarray): Reconstructed 3D array of images with shape (num_images, height, width).
    """

    num_images, img_hei, img_wid = original_shape
    row_patch_num = int(np.ceil((img_hei - patch_size) / slide_step)) + 1
    col_patch_num = int(np.ceil((img_wid - patch_size) / slide_step)) + 1

    images = np.zeros(original_shape)
    image = np.zeros((img_hei, img_wid))
    patch_count = np.zeros((num_images, img_hei, img_wid))
    num_patches = row_patch_num * col_patch_num
    row_pos_arr = list(range(0, (row_patch_num - 1) * slide_step, slide_step)) + [img_hei - patch_size]
    col_pos_arr = list(range(0, (col_patch_num - 1) * slide_step, slide_step)) + [img_wid - patch_size]

    for i in range(0,num_images):
        j =0
        for col in col_pos_arr:
            for row in row_pos_arr:
                tmp_patch = patch_ten[i,j, :, :]
                image[ row:row + patch_size, col:col + patch_size] = tmp_patch
                j +=1
                # patch_count[:, row:row + patch_size, col:col + patch_size] += 1
        images[i,:] = image
    # Avoid division by zero
    # patch_count[patch_count == 0] = 1

    # images /= patch_count

    return images


def gen_patch_tensor_GPU(images, patch_size=50, slide_step=50):
    """
    Generate patches from a 3D tensor of images and concatenate them into a 4D tensor.

    Parameters:
        images (torch.Tensor): 3D tensor of images with shape (num_images, height, width).
        patch_size (int): Size of the patches to extract.
        slide_step (int): Step size for sliding window.

    Returns:
        patch_ten (torch.Tensor): 4D tensor of patches with shape (num_images, num_patches, patch_size, patch_size).
    """

    device = images.device  # Get the device of the input tensor
    num_images, img_hei, img_wid = images.shape

    row_patch_num = int(np.ceil((img_hei - patch_size) / slide_step)) + 1
    col_patch_num = int(np.ceil((img_wid - patch_size) / slide_step)) + 1

    num_patches = row_patch_num * col_patch_num
    patch_ten = torch.zeros((num_images, num_patches, patch_size, patch_size), device=device)
    patch_image = torch.zeros((num_patches, patch_size, patch_size), device=device)

    row_pos_arr = list(range(0, (row_patch_num - 1) * slide_step, slide_step)) + [img_hei - patch_size]
    col_pos_arr = list(range(0, (col_patch_num - 1) * slide_step, slide_step)) + [img_wid - patch_size]

    for i in range(num_images):
        j = 0
        for col in col_pos_arr:
            for row in row_pos_arr:
                tmp_patch = images[i, row:row + patch_size, col:col + patch_size]
                patch_image[j, :, :] = tmp_patch
                j += 1
        patch_ten[i, :, :, :] = patch_image

    return patch_ten
def reconstruct_images_GPU(patch_ten, original_shape, patch_size=50, slide_step=50):
    """
    Reconstruct the original images from the 4D tensor of patches.

    Parameters:
        patch_ten (torch.Tensor): 4D tensor of patches with shape (num_images*num_patches, patch_size, patch_size).
        original_shape (tuple): The original shape of the images (num_images, height, width).
        patch_size (int): Size of the patches to extract.
        slide_step (int): Step size for sliding window.

    Returns:
        images (torch.Tensor): Reconstructed 3D tensor of images with shape (num_images, height, width).
    """

    device = patch_ten.device  # Get the device of the input tensor
    num_images, img_hei, img_wid = original_shape
    row_patch_num = int(np.ceil((img_hei - patch_size) / slide_step)) + 1
    col_patch_num = int(np.ceil((img_wid - patch_size) / slide_step)) + 1

    images = torch.zeros(original_shape, device=device)
    image = torch.zeros((img_hei, img_wid), device=device)

    num_patches = row_patch_num * col_patch_num
    row_pos_arr = list(range(0, (row_patch_num - 1) * slide_step, slide_step)) + [img_hei - patch_size]
    col_pos_arr = list(range(0, (col_patch_num - 1) * slide_step, slide_step)) + [img_wid - patch_size]

    for i in range(0, num_images):
        j = 0
        for col in col_pos_arr:
            for row in row_pos_arr:
                tmp_patch = patch_ten[i, j, :, :]
                image[row:row + patch_size, col:col + patch_size] = tmp_patch
                j += 1
                # patch_count[:, row:row + patch_size, col:col + patch_size] += 1
        images[i, :] = image
    # Avoid division by zero
    # patch_count[patch_count == 0] = 1

    # images /= patch_count

    return images
# 示例用法
if __name__ == "__main__":
    plt.ioff()
    img1 = cv2.imread(osp.join(r'H:\Dataset\Ours_v2_mat\Sequences\data9\1.bmp'))
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img10 = cv2.imread(osp.join(r'H:\Dataset\Ours_v2_mat\Sequences\data9\10.bmp'))
    img10 = cv2.cvtColor(img10, cv2.COLOR_BGR2GRAY)
    img12 = cv2.imread(osp.join(r'H:\Dataset\Ours_v2_mat\Sequences\data8\50.bmp'))
    img12 = cv2.cvtColor(img12, cv2.COLOR_BGR2GRAY)
    img3 = cv2.imread(osp.join(r'H:\Dataset\Ours_v2_mat\Sequences\data18\10.bmp'))
    img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
    img4 = cv2.imread(osp.join(r'H:\Dataset\Ours_v2_mat\Sequences\data18\50.bmp'))
    img4 = cv2.cvtColor(img4, cv2.COLOR_BGR2GRAY)
    img5 = cv2.imread(osp.join(r'H:\Dataset\Ours_v2_mat\Sequences\data20\10.bmp'))
    img5 = cv2.cvtColor(img5, cv2.COLOR_BGR2GRAY)
    plt.figure()
    plt.imshow(img12, cmap='gray')
    # plt.show()
    img = np.stack((img1,img10,img12,img3,img4,img5),axis=0)
    Torch_img = torch.from_numpy(img)
    patch_ten = gen_patch_tensor_GPU(Torch_img, patch_size=50, slide_step=50)
    Torch_img010 = reconstruct_images_GPU(patch_ten,img.shape, patch_size=50, slide_step=50)
    img010 = Torch_img010
    plt.figure()
    plt.imshow(img010[0,:,:], cmap='gray')
    plt.figure()
    plt.imshow(img010[1, :, :], cmap='gray')
    plt.figure()
    plt.imshow(img010[2, :, :], cmap='gray')
    plt.figure()
    plt.imshow(img010[3, :, :], cmap='gray')
    plt.figure()
    plt.imshow(img010[4, :, :], cmap='gray')
    # plt.figure()
    # plt.imshow(img010[5, :, :], cmap='gray')
    plt.show()
    print(patch_ten)
