#! python3
# -*- encoding: utf-8 -*-
'''
@File    :   segment.py
@Time    :   2024/11/28 23:25:21
@Author  :   ChenYang 
@Version :   1.0
@Contact :   cheny975@mail2.sysu.edu.cn
'''

import laspy
import numpy as np
from scipy.ndimage import gaussian_filter, label
from scipy.ndimage import watershed_ift
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage as ndi
import open3d as o3d
import random

# 读取点云文件
print("Reading point cloud file...")
input_file = "data/Forest.las"
las = laspy.read(input_file)
# 提取点云坐标和分类信息
x = las.x
y = las.y
z = las.z
classification = las.classification
# 将点分类为地面点和非地面点
ground_points = classification == 2
nonground_points = classification == 1

# 分别提取地面和非地面点的坐标和高程
x_ground, y_ground, z_ground = x[ground_points], y[ground_points], z[ground_points]
x_nonground, y_nonground, z_nonground = x[nonground_points], y[nonground_points], z[nonground_points]

# 生成地形模型 (DEM) 的网格
print("Generating Digital Terrain Model (DEM)...")
grid_size = 0.5  # 网格大小
x_min, x_max = np.min(x), np.max(x)
y_min, y_max = np.min(y), np.max(y)
x_bins = np.arange(x_min, x_max + grid_size, grid_size)
y_bins = np.arange(y_min, y_max + grid_size, grid_size)

# 将地面点投影到网格并生成DEM
print("Projecting ground points to grid...")
ground_grid = np.full((len(y_bins), len(x_bins)), np.nan)
for i in range(len(x_ground)):
    x_idx = int((x_ground[i] - x_min) // grid_size)
    y_idx = int((y_ground[i] - y_min) // grid_size)
    ground_grid[y_idx, x_idx] = (
        z_ground[i]
        if np.isnan(ground_grid[y_idx, x_idx])
        else min(ground_grid[y_idx, x_idx], z_ground[i])
    )

# 填补DEM中的空值（使用邻近值填充）
ground_grid = ndi.generic_filter(ground_grid, np.nanmin, size=30)
# plt.figure()
# plt.imshow(ground_grid, cmap='viridis')
# plt.colorbar(label='Height (m)')
# plt.title('Digital Elevation Model (DEM)')
# plt.xlabel('X Coordinate')
# plt.ylabel('Y Coordinate')
# plt.show()

# 将非地面点投影到网格并减去DEM生成CHM
print("Generating Canopy Height Model (CHM)...")
chm = np.full((len(y_bins), len(x_bins)), np.nan) 
for i in range(len(x_nonground)):
    x_idx = int((x_nonground[i] - x_min) // grid_size)
    y_idx = int((y_nonground[i] - y_min) // grid_size)
    if not np.isnan(ground_grid[y_idx, x_idx]):
        tree_height = z_nonground[i] - ground_grid[y_idx, x_idx]
        chm[y_idx, x_idx] = (
            tree_height
            if np.isnan(chm[y_idx, x_idx])
            else max(chm[y_idx, x_idx], tree_height)
        )

chm_smooth = np.nan_to_num(chm, nan=0.0)
# plt.figure()
# plt.imshow(chm_smooth, cmap='viridis')
# plt.colorbar(label='Height (m)')
# plt.title('Canopy Height Model (CHM)')
# plt.xlabel('X Coordinate')
# plt.ylabel('Y Coordinate')
# plt.show()
# 检测局部极值点作为种子点
circus = 10
coordinates = peak_local_max(
    chm_smooth, footprint=np.ones((circus, circus)), labels=~np.isnan(chm_smooth))

# 初始化一个与 chm_smooth 相同大小的布尔矩阵
local_max = np.zeros_like(chm_smooth, dtype=bool)

# 根据坐标将对应位置设为 True
local_max[tuple(coordinates.T)] = True

# 使用标记的局部极值点作为种子点进行分水岭分割
print("Performing watershed segmentation...")
markers, _ = ndi.label(local_max)
labels = watershed(-chm_smooth, markers, mask=~np.isnan(chm_smooth))

# 将分割结果保存为点云标签
output_labels = np.full(len(x), -1, dtype=int)
nonground_indices = np.where(nonground_points)[0]

for i, (xi, yi) in enumerate(zip(x_nonground, y_nonground)):
    x_idx = int((xi - x_min) // grid_size)
    y_idx = int((yi - y_min) // grid_size)
    if 0 <= x_idx < labels.shape[1] and 0 <= y_idx < labels.shape[0]:
        output_labels[nonground_indices[i]] = labels[y_idx, x_idx]

# 获取唯一的标签
unique_labels = np.unique(output_labels)
unique_labels = unique_labels[unique_labels != -1]  # 排除背景标签

# 创建一个随机颜色映射
random.seed(42)  # 固定随机种子以确保结果可重复
label_mapping = {label: random.randint(1, 255) for label in unique_labels}

# 将 output_labels 映射到新的颜色值，默认值 0
remapped_output_labels = np.vectorize(lambda x: label_mapping.get(x, 0))(output_labels)

# 保存分割结果为txt文件
output_file = f"Segmented_Forest_{circus}.txt"
with open(output_file, 'w') as f:
    f.write("X Y Z Classification Remapped_Classification\n")
    for i in range(len(x)):
        f.write(f"{x[i]} {y[i]} {z[i]} {output_labels[i]} {remapped_output_labels[i]}\n")

print(f"Segmentation saved to {output_file}.")

# 输出分割出的树木个数
num_trees = len(unique_labels)
print(f"Number of segmented trees: {num_trees}")