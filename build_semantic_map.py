import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import open3d as o3d
import cv2
import os

class IncrementalPointCloudMerger:
    def __init__(self):
        """
        初始化增量式点云合并器
        """
        self.merged_world_points = None
        self.merged_labels = None
        self.camera_params_list = []
        self.depth_image_paths = []

    def add_image(self, depth_image_path, camera_position, euler_angles, depth_enhanced=None):
        """
        增量添加图像到点云

        参数:
        depth_image_path: 深度图像路径
        camera_position: 相机位置 [x, y, z]
        euler_angles: 相机旋转角度 [roll, pitch, yaw]
        depth_enhanced: 可选的增强深度图数据，如果为None则尝试从文件加载
        """
        # 创建相机参数
        rotation_matrix = self._create_rotation_matrix(*euler_angles)
        extrinsic_matrix = self._create_extrinsic_matrix(camera_position, rotation_matrix)

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

        # 读取深度图
        depth_image = self._read_depth_image(depth_image_path)

        # 如果没有提供depth_enhanced，尝试从文件加载
        if depth_enhanced is None:
            try:
                depth_enhanced = np.load(os.path.splitext(depth_image_path)[0] + '_enhanced.npy')
            except Exception as e:
                print(f"加载增强深度图失败: {e}")
                return

        if depth_image is not None and depth_enhanced is not None:
            # 计算世界坐标和标签
            world_points, labels = self._depth_to_world_coordinates(depth_image, camera_params, depth_enhanced)

            # 合并点云
            if self.merged_world_points is None:
                self.merged_world_points = world_points
                self.merged_labels = labels
            else:
                self.merged_world_points = np.vstack([self.merged_world_points, world_points])
                self.merged_labels = np.concatenate([self.merged_labels, labels])

            # 记录路径和参数
            self.depth_image_paths.append(depth_image_path)
            self.camera_params_list.append(camera_params)

            print(f"已添加图像: {depth_image_path}")
            print(f"当前点云总点数: {len(self.merged_world_points)}")

    def calculate_depth_weighted_center(self):
        """
        计算以深度为权重的点云中心坐标

        返回:
        depth_weighted_center: 以深度为权重的中心坐标 [x, y, z]
        """
        if self.merged_world_points is None:
            print("尚未添加任何点云数据")
            return None

            # 使用深度（标签）作为权重
        weights = self.merged_labels / np.sum(self.merged_labels)

        # 计算加权中心
        depth_weighted_center = np.sum(
            self.merged_world_points * weights[:, np.newaxis],
            axis=0
        )

        return depth_weighted_center


    def visualize_merged_pointcloud(self):
        """
        可视化合并的点云数据
        """
        if self.merged_world_points is None:
            print("尚未添加任何点云数据")
            return

        plt.figure(figsize=(15, 10))
        ax = plt.axes(projection='3d')

        # 创建蓝色渐变颜色映射
        # 使用 0-1 的值映射蓝色深浅
        colors = plt.cm.Blues(self.merged_labels)

        scatter = ax.scatter(
            self.merged_world_points[:, 0],
            self.merged_world_points[:, 1],
            self.merged_world_points[:, 2],
            c=colors,
            alpha=0.6,
            s=2
        )

        ax.set_title('Merged 3D Point Cloud with Color Intensity')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # 添加颜色条
        plt.colorbar(
            plt.cm.ScalarMappable(
                cmap=plt.cm.Blues,
                norm=plt.Normalize(vmin=0, vmax=1)
            ),
            ax=ax,
            label='Intensity'
        )

        plt.tight_layout()
        plt.show()

        # Open3D 可视化
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.merged_world_points)

        # 使用相同的颜色映射
        color_list = np.array([color[:3] for color in colors])  # 只取RGB通道
        pcd.colors = o3d.utility.Vector3dVector(color_list)

        o3d.visualization.draw_geometries([pcd])

    def _create_rotation_matrix(self, roll, pitch, yaw):
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

    def _create_extrinsic_matrix(self, position, rotation_matrix):
        """
        创建完整的外参矩阵
        """
        extrinsic_matrix = np.eye(4)
        extrinsic_matrix[:3, :3] = rotation_matrix
        extrinsic_matrix[:3, 3] = position
        return extrinsic_matrix

    def _read_depth_image(self, file_path):
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

    def _depth_to_world_coordinates(self, depth_image, camera_params, depth_enhanced):
        """
        将深度图转换为世界坐标系下的点云坐标
        """
        height, width = depth_image.shape
        x, y = np.meshgrid(np.arange(width), np.arange(height))

        x_flat = x.flatten()
        y_flat = y.flatten()
        depth_flat = depth_image.flatten()
        labels_flat = depth_enhanced[:, :, 1].flatten()

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

def main():
    # 创建增量式点云合并器
    merger = IncrementalPointCloudMerger()

    # 依次添加图像
    depth_enhanced1 = np.load("inputs/depth_image2_enhanced.npy")
    merger.add_image(
        "inputs/depth_image1.png",
        np.array([63.957007, -41.789424, 9.422275]),
        [-90, 0, 180],
        depth_enhanced1
    )

    depth_enhanced2 = np.load("inputs/depth_image2_enhanced.npy")
    merger.add_image(
        "inputs/depth_image2.png",
        np.array([63.957007, -41.789424, 9.422275]),
        [-90, 0, 270],
        depth_enhanced2
    )

    # merger.add_image(
    #     "inputs/depth_image3.png",
    #     np.array([63.957007, -41.789424, 9.422275]),
    #     [-90, 0, 0]
    # )
    #
    # merger.add_image(
    #     "inputs/depth_image4.png",
    #     np.array([63.957007, -41.789424, 9.422275]),
    #     [-90, 0, 90]
    # )

    # 可视化合并的点云
    merger.visualize_merged_pointcloud()

if __name__ == "__main__":
    main()