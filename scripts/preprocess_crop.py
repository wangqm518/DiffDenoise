import os

import torch
import torchvision.transforms.functional as F
from PIL import Image
import numpy as np
from torchvision.transforms import ToPILImage
from torchvision.transforms import ToTensor
import json


def sliding_window_crop(image_path, patch_size=256, stride_1080=206, stride_1920=208):
    """
    对输入图像进行滑动窗口裁剪
    :param image_path: 输入图像的路径
    :param patch_size: 裁剪窗口大小 (默认 256)
    :param stride: 滑动步长
    :return: 裁剪出的图像块列表
    """
    image = Image.open(image_path)
    # print(f"image size: {image.size}")
    width, height= image.size
    patches = []
    positions = []

    if height == 1920:
        for y in range(0, height - patch_size + 1, stride_1920):
            for x in range(0, width - patch_size + 1, stride_1080):
                patch = F.crop(image, y, x, patch_size, patch_size)
                patches.append(patch)
                positions.append((x, y))  # 记录 patch 位置
    elif height == 1080:
        for y in range(0, height - patch_size + 1, stride_1080):
            for x in range(0, width - patch_size + 1, stride_1920):
                patch = F.crop(image, y, x, patch_size, patch_size)
                patches.append(patch)
                positions.append((x, y))  # 记录 patch 位置

    return patches, positions

def process_patches(model, patches, transform):
    """
    用模型处理每个裁剪出的 patch
    :param model: 预训练模型
    :param patches: 裁剪出的图像块 (列表)
    :param transform: 图像预处理变换
    :return: 预测结果
    """
    model.eval()
    results = []
    to_tensor = ToTensor()  # 用于将 PIL Image 转换为 PyTorch Tensor

    with torch.no_grad():
        for patch in patches:
            patch_tensor = to_tensor(patch).unsqueeze(0)  # 变成 (1, C, H, W)
            output = model(patch_tensor)
            results.append(output)

    return results


def merge_patches(patches, positions, image_size, patch_size=256, stride_1080=206, stride_1920=208):
    """
    将多个 patch 还原回完整图片
    :param patches: 处理后的 patch (pil格式)
    :param positions: patch 对应的 (x, y) 位置
    :param image_size: 原始图像大小 (H, W)
    :param patch_size: 每个 patch 的大小 (默认 256)
    :param stride: 滑动步长 (默认 128)
    :return: 还原的完整图像
    """
    W,H = image_size
    C = 3  # RGB 通道数
    output = torch.zeros((C, H, W))  # 改为 3 通道输出 (C, H, W)
    count = torch.zeros((H, W))  # 计数器仍为单通道（空间维度）
    to_tensor = ToTensor()  # 用于将 PIL Image 转换为 PyTorch Tensor
    # print(positions)

    for patch, (x, y) in zip(patches, positions):
        # print(f"(x,y):{x,y}")
        # 将 PIL Image 转换为 PyTorch Tensor (C, H, W)
        patch = to_tensor(patch)  # 自动归一化到 [0, 1]
        # print(f"patch_shape: {patch.shape}")

        # 如果 patch 有 batch 维度（如 (1, C, H, W)），去掉它
        if patch.dim() == 4:
            patch = patch.squeeze(0)  # (C, H, W)

        output[:, y:y + patch_size, x:x + patch_size] += patch  # 对所有通道操作
        count[y:y + patch_size, x:x + patch_size] += 1


    # 将计数器的值扩展到 3 通道，然后做除法
    count_expanded = count.unsqueeze(0).expand(C, -1, -1)  # (C, H, W)
    output /= count_expanded  # 逐通道除以重叠次数
    output = output.squeeze(0)

    # 将张量转换为 PIL Image
    output = ToPILImage()(output.clamp(0, 1))  # 确保值在 [0, 1] 范围内
    return output

def main():
    noised_img_dir = 'data_noised'
    files = [f for f in os.listdir(noised_img_dir) if not os.path.isdir(os.path.join(noised_img_dir,f))]
    image_paths = [os.path.join(noised_img_dir,f) for f in files]
    for image_path in image_paths:
        dict = {}
        image = Image.open(image_path)
        print(f"noised image size: {image.size}")
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        print(f"image name: {image_name}")
        dict['image_name'] = image_name
        dict['image_size'] = image.size
        save_dir = os.path.join(noised_img_dir,image_name)
        os.makedirs(save_dir, exist_ok=True)
        patches, positions = sliding_window_crop(image_path)
        # 保存划分的块patch
        for i,patch in enumerate(patches):
            save_path = os.path.join(save_dir,f'{i:03}.png')
            patch.save(save_path)
        # 保存每个块的位置
        print(f"patch positions: {positions}")
        dict['pos'] = positions
        json_save_path = os.path.join(save_dir, f'pos_{image_name}.json')
        with open(json_save_path, 'w', encoding='utf-8') as f:
            json.dump(dict, f, ensure_ascii=False) # 去掉缩进参数indent来不换行
    print('Done!')

if __name__ == '__main__':
    main()
