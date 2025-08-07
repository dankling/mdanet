import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from lib.dataset import Data
from net import Mnet
import time

# 设置 GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == '__main__':
    model_path = 'model/pearlfish1.pth'
    out_path = 'logo'
    heatmap_out_path = 'logoheatmap'  # 新增热图输出目录
    data = Data(root='datasets/logo/', cla=4, mode='test')  # 数据集加载
    state_dict = torch.load(model_path)

    loader = DataLoader(data, batch_size=1, shuffle=False)
    net = Mnet().cuda()  # 加载模型
    net.load_state_dict(state_dict)
    
    # 确保输出目录存在
    if not os.path.exists(out_path): 
        os.makedirs(out_path)
    if not os.path.exists(heatmap_out_path):  # 确保热图输出目录存在
        os.makedirs(heatmap_out_path)

    img_num = len(loader)
    net.eval()

    print(f"Total images to process: {img_num}")
    with torch.no_grad():
        for idx, (rgb, mask, cla, (H, W), name) in enumerate(loader):
            start_time = time.time()
            print(f"Processing {idx + 1}/{img_num}: {name[0]}")

            # 确保文件路径的后缀是.jpg
            image_path = os.path.join('datasets/logo/RGB', name[0].replace('.png', '.jpg'))  # 将.png替换为.jpg

            # 加载RGB图像 
            rgb_image = cv2.imread(image_path)

            if rgb_image is None:
                print(f"Error loading image: {name[0]}")
                continue  # 跳过此图像
            
            # 模型预测
            score1, score2, score3, score4 = net(rgb.cuda().float())
            score = F.interpolate(score1, size=(H, W), mode='bilinear', align_corners=True)
            pred = torch.sigmoid(score).cpu().numpy().squeeze()

            # 设置阈值
            threshold = 0.5
            pred = (pred > threshold).astype(np.uint8) * 255  # 应用阈值并转换为 0/255

            # 保存预测结果
            output_file = os.path.join(out_path, name[0][:-4] + '.png')
            cv2.imwrite(output_file, pred)
            
            # 生成热图
            heatmap = np.squeeze(score.cpu().numpy())  # 获取分数用于生成热图
            heatmap = np.maximum(heatmap, 0)  # 确保热图没有负值
            
            # 强制设置一个最小值（例如0.1），以确保背景部分也能显示
            heatmap[heatmap < 0.1] = 0.1  # 设置一个阈值，增强背景部分
            
            # 线性归一化（确保背景信息不会丢失）
            heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))
            
            # 将热图转换为伪彩色图像
            heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)

            # 调整热力图尺寸以匹配RGB图像
            heatmap_colored_resized = cv2.resize(heatmap_colored, (rgb_image.shape[1], rgb_image.shape[0]))

            # 确保热力图是3通道，如果是单通道，则转换
            if len(rgb_image.shape) == 2:  # 如果是灰度图
                rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_GRAY2BGR)
            if len(heatmap_colored_resized.shape) == 2:  # 如果是灰度图
                heatmap_colored_resized = cv2.cvtColor(heatmap_colored_resized, cv2.COLOR_GRAY2BGR)

            # 设置透明度
            alpha = 0.5  # 热力图透明度
            beta = 1 - alpha  # RGB图像透明度

            # 进行加权叠加
            overlay_image = cv2.addWeighted(rgb_image, alpha, heatmap_colored_resized, beta, 0)

            # 保存叠加图像
            overlay_output_file = os.path.join(heatmap_out_path, name[0][:-4] + '_overlay.png')
            cv2.imwrite(overlay_output_file, overlay_image)
            
            # 保存热图
            heatmap_output_file = os.path.join(heatmap_out_path, name[0][:-4] + '_heatmap.png')
            cv2.imwrite(heatmap_output_file, heatmap_colored)
            
            # 统计时间
            elapsed_time = time.time() - start_time
            print(f"Saved to {output_file}. Time taken: {elapsed_time:.2f} seconds")
            print(f"Heatmap saved to {heatmap_output_file}")
            print(f"Overlay saved to {overlay_output_file}")

    print("All images processed.")
