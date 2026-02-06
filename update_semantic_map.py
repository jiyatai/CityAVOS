import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import cv2
from collections import Counter
from matplotlib.lines import Line2D
import os

# Global variables to hold merged point cloud data
merged_world_points = None
merged_labels = None
camera_params_list = []
depth_image_paths = []

def generate_semantic_map(x_min, x_max, y_min, y_max, z_min, z_max, interval, interval_z):
    """生成三维网格点，每个点包含xyz坐标和label值"""
    points = []
    x_start, x_end = min(x_min, x_max), max(x_min, x_max)
    y_start, y_end = min(y_min, y_max), max(y_min, y_max)
    z_start, z_end = min(z_min, z_max), max(z_min, z_max)

    for x in range(x_start, x_end + interval, interval):
        for y in range(y_start, y_end + interval, interval):
            for z in range(z_start, z_end + interval_z, interval_z):
                point = np.array([float(x), float(y), float(z), 0])  # [x, y, z, label]
                points.append(point)

    return np.array(points, dtype=np.float64)

def generate_coginitive_map(x_min, x_max, y_min, y_max, z_min, z_max, interval, interval_z):
    """生成三维网格点，每个点包含xyz坐标和label值"""
    points = []
    x_start, x_end = min(x_min, x_max), max(x_min, x_max)
    y_start, y_end = min(y_min, y_max), max(y_min, y_max)
    z_start, z_end = min(z_min, z_max), max(z_min, z_max)

    for x in range(x_start, x_end + interval, interval):
        for y in range(y_start, y_end + interval, interval):
            for z in range(z_start, z_end + interval_z, interval_z):
                point = np.array([float(x), float(y), float(z), 0, 1])  # [x, y, z, label]
                points.append(point)

    return np.array(points, dtype=np.float64)

def add_label_to_semantic_map(world_points, labels, semantic_map, cognitive_map, scores_rel):
    """对world_points四舍五入后进行归类，每一个相同值所对应的标签最多的为真实标签，将真实标签添加至semantic_map中对应的点上"""
    # 创建一个字典来存储每个点的标签
    label_dict = {}

    # 将世界坐标点四舍五入到最近的网格点
    for point, label in zip(world_points, labels):
        rounded_point = tuple(np.round(point).astype(int))  # 四舍五入并转换为元组作为字典的键
        if rounded_point in label_dict:
            label_dict[rounded_point].append(label)
        else:
            label_dict[rounded_point] = [label]

    # 确定每个网格点的主导标签
    for key, value in label_dict.items():
        dominant_label = Counter(value).most_common(1)[0][0]  # 获取最常见的标签
        # 更新语义图中对应点的标签
        idx = np.where((semantic_map[:, 0] == key[0]) & (semantic_map[:, 1] == key[1]) & (semantic_map[:, 2] == key[2]))
        if idx[0].size > 0:
            semantic_map[idx[0][0], 3] = dominant_label  # 更新语义图中的标签
            cognitive_map[idx[0][0], 3] = scores_rel[int(dominant_label)-1] * cognitive_map[idx[0][0], 4]  # 更新认知图中的标签

    return semantic_map, cognitive_map


#
def visualize_semantic_map(semantic_map, step, if_figure_plot,
                           filename_prefix='D:/JYT/code/output/semantic_map/semantic_map'):
    """
    可视化3D语义图 (优化版)
    特点：窗口常驻、非阻塞、只渲染有效点
    """

    # --- 1. 初始化绘图窗口 (单例模式) ---
    # 使用函数属性来存储 fig 和 ax，避免使用全局变量，同时保证只初始化一次
    if not hasattr(visualize_semantic_map, 'fig'):
        plt.ion()  # 开启交互模式，非阻塞
        visualize_semantic_map.fig = plt.figure(figsize=(10, 8))
        visualize_semantic_map.ax = visualize_semantic_map.fig.add_subplot(111, projection='3d')

        # 预定义颜色映射，避免每次循环重复定义
        visualize_semantic_map.color_map = {
            0: (1, 1, 1, 0),  # 透明
            1: (1, 1, 0.5, 1),  # 黄色 车
            2: (0.25, 1, 0.25, 1),  # 绿色 树
            3: (0.25, 1, 1, 1),  # 青色 人行道
            4: (0, 0.5, 1, 1),  # 蓝色 建筑
            5: (0.5, 0.5, 0.5, 1),  # 灰色 路
            6: (1, 0.25, 1, 1),  # 品红色 标志
            7: (1, 0.25, 0.25, 1)  # 红色 广告牌
        }

    fig = visualize_semantic_map.fig
    ax = visualize_semantic_map.ax
    color_map = visualize_semantic_map.color_map

    # 清空当前轴的内容，而不是关闭窗口
    ax.clear()

    # --- 2. 数据处理与优化 ---
    grid_keys = semantic_map[:, :3]
    dominant_labels = semantic_map[:, 3]

    # 【优化】只绘制非透明的点 (label != 0)
    # 3D散点图渲染非常耗时，过滤掉空点可以极大提升FPS
    valid_mask = dominant_labels != 0

    if np.any(valid_mask):
        valid_points = grid_keys[valid_mask]
        valid_labels = dominant_labels[valid_mask]

        # 向量化生成颜色列表 (比列表推导式稍快，且代码更整洁)
        colors = [color_map[int(label)] for label in valid_labels]

        # --- 3. 绘图 ---
        ax.scatter(valid_points[:, 0], valid_points[:, 1], valid_points[:, 2], c=colors, marker='o', s=20)
    else:
        # 如果没有点，画一个空的防止报错
        pass

    # --- 4. 设置视图属性 (每次clear后都需要重新设置) ---
    ax.set_box_aspect([1, 1, 0.5])
    # ax.set_aspect('equal') # 3D图中某些版本matplotlib不支持此参数，box_aspect通常足够

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'3D Semantic Map - Step {step}')

    # 关闭网格和坐标轴
    ax.grid(False)
    ax.set_axis_off()

    # --- 5. 显示与保存 ---

    # 确保保存目录存在
    save_dir = os.path.dirname(filename_prefix)
    if not os.path.exists(save_dir) and save_dir != '':
        os.makedirs(save_dir)

    filename = f"{filename_prefix}_step_{step}.png"
    plt.savefig(filename, bbox_inches='tight')

    if if_figure_plot:
        # 刷新窗口事件，暂停极短时间让GUI更新，但不阻塞
        plt.draw()
        plt.pause(0.01)


#
def visualize_cognitive_map(pos, look, cognitive_map, scores_rel, step, max_cluster_center, if_figure_plot,
                            filename_prefix='D:/JYT/code/output/cognitive_map/cognitive_map'):

    grid_keys = cognitive_map[:, :3]  # 获取x, y, z坐标

    dominant_labels = cognitive_map[:, 3]  # 获取标签

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    colors = np.array([(1, 1 - label, 1 - label, label) for label in dominant_labels])

    ax.scatter(grid_keys[:, 0], grid_keys[:, 1], grid_keys[:, 2], c=colors, marker='o')

    ax.scatter(pos[0], pos[1], pos[2], color='blue', s=50, marker='o')  # 蓝点，大小为100

    scaled_look = look / np.linalg.norm(look) * 10
    ax.quiver(pos[0], pos[1], pos[2], scaled_look[0], -scaled_look[1], scaled_look[2], color='red',
          arrow_length_ratio=0.1)  # 红色箭头

    if max_cluster_center is not None:
        # 添加光晕效果
        ax.scatter(max_cluster_center[0], max_cluster_center[1], max_cluster_center[2],
                   color='yellow', s=300, marker='o', alpha=0.3, zorder=9)

        # 主标志（金色五角星）
        ax.scatter(max_cluster_center[0], max_cluster_center[1], max_cluster_center[2],
                   color='gold', s=200, marker='*', edgecolors='black', linewidths=2,
                   label='Target Cluster', zorder=10)

        # 添加文本标注（可选）
        ax.text(max_cluster_center[0], max_cluster_center[1], max_cluster_center[2] + 0.5,
                'Target', color='black', fontsize=10, fontweight='bold',
                ha='center', va='bottom', zorder=11)

        ax.legend(loc='upper right', fontsize=10)

    ax.set_box_aspect([1, 1, 0.5])  # 设置xyz三个方向的比例
    ax.set_aspect('equal', adjustable='box')  # 确保xy平面保持真实比例

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Cognitive Map')

    ax.grid(False)  # 关闭网格
    ax.set_axis_off()  # 隐藏坐标轴

    if if_figure_plot:
        plt.show()

    filename = f"{filename_prefix}_step_{step}.png"  # 使用step命名文件

    plt.draw()

    plt.savefig(filename, bbox_inches='tight')  # 保存为图片，文件名为filename

    plt.close()


def update_semantic_map(semantic_map, cognitive_map, depth_image_path, camera_position, euler_angles, depth_enhanced, scores_rel):

    # Create camera parameters
    rotation_matrix = create_rotation_matrix(*euler_angles)
    extrinsic_matrix = create_extrinsic_matrix(camera_position, rotation_matrix)

    camera_params = {
        'intrinsic_matrix': np.array([
            [320, 0, 320],  # fx, 0, cx
            [0, 320, 240],  # 0, fy, cy
            [0, 0, 1]       # 0, 0, 1
        ]),
        'extrinsic_matrix': extrinsic_matrix,
        'width': 640,
        'height': 480
    }

    # Read depth image
    depth_image = read_depth_image(depth_image_path)

    world_points, labels = depth_to_world_coordinates(depth_image, camera_params, depth_enhanced)
    # print(world_points)
    # print(labels)

    updated_semantic_map, updated_cognitive_map = add_label_to_semantic_map(world_points, labels, semantic_map, cognitive_map, scores_rel)

    return updated_semantic_map, updated_cognitive_map

def update_coginitive_map(coginitive_map, depth_image_path, camera_position, euler_angles, depth_enhanced):

    # Create camera parameters
    rotation_matrix = create_rotation_matrix(*euler_angles)
    extrinsic_matrix = create_extrinsic_matrix(camera_position, rotation_matrix)

    camera_params = {
        'intrinsic_matrix': np.array([
            [320, 0, 320],  # fx, 0, cx
            [0, 320, 240],  # 0, fy, cy
            [0, 0, 1]       # 0, 0, 1
        ]),
        'extrinsic_matrix': extrinsic_matrix,
        'width': 640,
        'height': 480
    }

    # Read depth image
    depth_image = read_depth_image(depth_image_path)

    world_points, labels = depth_to_world_coordinates(depth_image, camera_params, depth_enhanced)
    # print(world_points)
    # print(labels)

    updated_coginitive_map = add_label_to_semantic_map(world_points, labels, coginitive_map)

    return updated_coginitive_map


def calculate_depth_weighted_center():
    """
    计算以深度为权重的点云中心坐标

    返回:
    depth_weighted_center: 以深度为权重的中心坐标 [x, y, z]
    """
    global merged_world_points, merged_labels

    if merged_world_points is None:
        print("尚未添加任何点云数据")
        return None

    # 使用深度（标签）作为权重
    weights = merged_labels / np.sum(merged_labels)

    # 计算加权中心
    depth_weighted_center = np.sum(
        merged_world_points * weights[:, np.newaxis],
        axis=0
    )

    return depth_weighted_center


def create_rotation_matrix(roll, pitch, yaw):
    """
    创建3D旋转矩阵
    """
    roll_rad = np.deg2rad(roll)
    pitch_rad = np.deg2rad(pitch)
    yaw_rad = np.deg2rad(yaw)

    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll_rad), -np.sin(roll_rad)],
        [0, np.sin(roll_rad), np.cos(roll_rad)]
    ])

    Ry = np.array([
        [np.cos(pitch_rad), 0, np.sin(pitch_rad)],
        [0, 1, 0],
        [-np.sin(pitch_rad), 0, np.cos(pitch_rad)]
    ])

    Rz = np.array([
        [np.cos(yaw_rad), -np.sin(yaw_rad), 0],
        [np.sin(yaw_rad), np.cos(yaw_rad), 0],
        [0, 0, 1]
    ])

    return Rz @ Ry @ Rx

def create_extrinsic_matrix(position, rotation_matrix):
    """
    创建完整的外参矩阵
    """
    extrinsic_matrix = np.eye(4)
    extrinsic_matrix[:3, :3] = rotation_matrix
    extrinsic_matrix[:3, 3] = position
    return extrinsic_matrix

def read_depth_image(file_path):
    """
    读取深度图
    """
    try:
        depth_image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        if depth_image.dtype == np.uint16:
            depth_image = depth_image.astype(np.float32) / 1000.0
        return depth_image
    except Exception as e:
        print(f"读取深度图时出错: {e}")
        return None

def depth_to_world_coordinates(depth_image, camera_params, depth_enhanced):
    """
    将深度图转换为世界坐标系下的坐标
    """
    height, width = depth_image.shape
    x, y = np.meshgrid(np.arange(width), np.arange(height))

    x_flat = x.flatten()
    y_flat = y.flatten()
    depth_flat = depth_image.flatten()
    labels_flat = depth_enhanced.flatten()

    non_zero_mask = labels_flat != 0
    x_filtered = x_flat[non_zero_mask]
    y_filtered = y_flat[non_zero_mask]
    depth_filtered = depth_flat[non_zero_mask]
    labels_filtered = labels_flat[non_zero_mask]

    K = camera_params['intrinsic_matrix']
    K_inv = np.linalg.inv(K)
    extrinsic_matrix = camera_params['extrinsic_matrix']

    pixel_coords_normalized = np.column_stack([
        x_filtered,
        y_filtered,
        np.ones_like(x_filtered)
    ])

    camera_coords = K_inv @ pixel_coords_normalized.T
    camera_coords *= depth_filtered

    camera_coords_homo = np.vstack([
        camera_coords,
        np.ones((1, camera_coords.shape[1]))
    ])

    world_coords_homo = extrinsic_matrix @ camera_coords_homo
    world_points = world_coords_homo[:3, :].T

    return world_points, labels_filtered

