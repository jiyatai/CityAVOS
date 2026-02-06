import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.colors import Normalize

def generate_uncertainty_map(x_min, x_max, y_min, y_max, z_min, z_max, interval, interval_z):
    """生成三维网格点，每个点有六个深度值"""
    points = []
    x_start, x_end = min(x_min, x_max), max(x_min, x_max)
    y_start, y_end = min(y_min, y_max), max(y_min, y_max)
    z_start, z_end = min(z_min, z_max), max(z_min, z_max)

    for x in range(x_start, x_end + interval, interval):
        for y in range(y_start, y_end + interval, interval):
            for z in range(z_start, z_end + interval_z, interval_z):
                point = np.array([float(x), float(y), float(z)] + [1.0])  # [x, y, z, depth1, depth2, depth3, depth4, depth5, depth6]
                points.append(point)

    return np.array(points, dtype=np.float64)

def uncertainty_map_update(points, observer_pos, look_direction, step_x, fov=90, max_distance=100):
    """获取当前视野内的可见面并更新深度值"""
    look_direction = look_direction * [1, -1, 1]

    look_direction_normalized = look_direction / np.linalg.norm(look_direction)

    updated_points = points.copy()
    cos_fov = np.cos(np.radians(fov / 2))

    for i, point in enumerate(points):
        point_pos = point[:3]  # 获取点的位置 [x, y, z]
        to_point = point_pos - observer_pos  # 从观察者到目标的向量
        distance = np.linalg.norm(to_point)  # 计算与观察者的距离

        if distance > max_distance:
            continue

        if distance > 0:  # 计算可见性
            to_point_normalized = to_point / distance
            cos_angle = np.dot(to_point_normalized, look_direction_normalized)

            if cos_angle > cos_fov:  # 如果点落在视场范围内
                face_depths = updated_points[i][3:]  # 获取当前面的深度值

                log_curr_dist = np.exp(-distance*0.15/step_x)

                # 更新深度值
                face_depths = max(0, face_depths * (1 - log_curr_dist))

                updated_points[i][3:] = face_depths

    return updated_points


def cognitive_map_denoising(points, observer_pos, look_direction, step_x, fov=120, max_distance=2000):
    """获取当前视野内的可见面并更新深度值"""
    look_direction = look_direction * [1, -1, 1]
    look_direction_normalized = look_direction / np.linalg.norm(look_direction)

    updated_points = points.copy()
    cos_fov = np.cos(np.radians(fov / 2))

    for i, point in enumerate(points):
        point_pos = point[:3]  # 获取点的位置 [x, y, z]
        to_point = point_pos - observer_pos  # 从观察者到目标的向量
        distance = np.linalg.norm(to_point)  # 计算与观察者的距离

        if distance > step_x * 1.9:
            continue

        if distance > 0:  # 计算可见性
            to_point_normalized = to_point / distance
            cos_angle = np.dot(to_point_normalized, look_direction_normalized)

            if cos_angle > cos_fov:  # 如果点落在视场范围内
                updated_points[i][3] = 0
                updated_points[i][4] = 0  # 更新深度值

    return updated_points



def visualize_uncertainty_map(points, observer_pos, look_direction, step, if_figure_plot, filename_prefix='D:/JYT/code/output/uncertainty_map/uncertainty_map'):
    """可视化当前三维点的深度值，每个点表示为10x10x5的立方体"""
    look_direction = look_direction * [1, -1, 1]
    fig = plt.figure(figsize=(15, 12))
    ax = fig.add_subplot(111, projection='3d')

    # 创建颜色映射和归一化
    cmap = plt.cm.Blues
    norm = Normalize(vmin=0, vmax=1)

    # 立方体的尺寸
    cube_x, cube_y, cube_z = 5, 5, 5

    # 绘制每个点为立方体
    for point in points:
        x, y, z = point[:3]
        depths = point[3:]  # 获取六个面的深度值
        # 创建立方体的顶点
        x_vertices = [x, x, x + cube_x, x + cube_x, x, x, x + cube_x, x + cube_x]
        y_vertices = [y, y + cube_y, y, y + cube_y, y, y + cube_y, y, y + cube_y]
        z_vertices = [z, z, z, z, z + cube_z, z + cube_z, z + cube_z, z + cube_z]

        # 定义立方体的面
        vertices = np.column_stack((x_vertices, y_vertices, z_vertices))

        color = cmap(depths)

        # 使用深度值作为颜色强度
        for j in range(6):  # 6个面
            faces = [
                [vertices[0], vertices[1], vertices[3], vertices[2]],  # 底面
                [vertices[4], vertices[5], vertices[7], vertices[6]],  # 顶面
                [vertices[0], vertices[4], vertices[6], vertices[2]],  # 前面
                [vertices[1], vertices[5], vertices[7], vertices[3]],  # 后面
                [vertices[0], vertices[1], vertices[5], vertices[4]],  # 左面
                [vertices[2], vertices[3], vertices[7], vertices[6]]  # 右面
            ]

            # 绘制每个面
            face_collection = Poly3DCollection([faces[j]], alpha=0.2, facecolor=color, edgecolor='k', linewidth=0.5)
            ax.add_collection3d(face_collection)

    # 绘制观察者位置和方向
    ax.scatter(*observer_pos, c='red', s=100, label='Observer')

    # 绘制观察方向
    ax.quiver(
        observer_pos[0], observer_pos[1], observer_pos[2],  # 起点坐标
        look_direction[0], look_direction[1], look_direction[2],  # 方向向量
        color='red',
        label='View Direction',
        length=20  # 添加长度参数
    )

    # 设置坐标轴标签和标题
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Grid Points Represented as Cubic Volumes with Depth Values')

    # 添加颜色条
    plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                 ax=ax, label='Depth Value', shrink=0.8, aspect=20)

    # 设置坐标轴范围
    x_min, x_max = points[:, 0].min() - 10, points[:, 0].max() + 20
    y_min, y_max = points[:, 1].min() - 10, points[:, 1].max() + 20
    z_min, z_max = points[:, 2].min() - 10, points[:, 2].max() + 20

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)

    # ax.legend()
    plt.tight_layout()

    ax.grid(False)  # 关闭网格
    ax.set_axis_off()  # 隐藏坐标轴

    if if_figure_plot:
        plt.show()

    # 生成文件名
    filename = f"{filename_prefix}_step_{step}.png"  # 使用step命名文件

    # 保存图片
    plt.savefig(filename, bbox_inches='tight')  # 保存为图片，文件名为filename

    plt.close()

def is_target_visible(target_pos, observer_pos, look_direction, fov=120, max_distance=20):
    """判断目标位置是否在可见范围内"""
    look_direction = look_direction * [1, -1, 1]
    look_direction_normalized = look_direction / np.linalg.norm(look_direction)
    to_target = target_pos - observer_pos
    distance = np.linalg.norm(to_target)

    if distance > max_distance:
        return False  # 超出最大距离

    if distance > 0:  # 计算可见性
        to_target_normalized = to_target / distance
        cos_angle = np.dot(to_target_normalized, look_direction_normalized)
        cos_fov = np.cos(np.radians(fov / 2))

        if cos_angle > cos_fov:  # 如果目标在视场范围内
            return True

    return False

def is_target_visible2(target_pos, observer_pos, look_direction, step_size, fov=120):
    """判断目标位置是否在可见范围内"""
    look_direction = look_direction * [1, -1, 1]
    look_direction_normalized = look_direction / np.linalg.norm(look_direction)
    to_target = target_pos - observer_pos
    distance = np.linalg.norm(to_target)

    if distance > 2 * step_size:
        return False  # 超出最大距离

    if distance > 0:  # 计算可见性
        to_target_normalized = to_target / distance
        cos_angle = np.dot(to_target_normalized, look_direction_normalized)
        cos_fov = np.cos(np.radians(fov / 2))

        if cos_angle > cos_fov:  # 如果目标在视场范围内
            return True

    return False

def is_target_visible3(target_pos, observer_pos):
    """判断目标位置是否在可见范围内"""

    to_target = target_pos - observer_pos
    distance = np.linalg.norm(to_target)

    if distance < 15:
        return True  # 超出最大距离
    else:
        return False



def main():
    # 初始化点数据
    points = generate_grid_points(6330, 6410, -4210, -4140, 1, 16, 10, 5)
    observer_pos = [6350, -4160, 6]
    look_direction = [1, 0, 0]
    total_depth = np.sum(points[:, 3:])
    print("总深度:", total_depth)
    # 获取可见面并更新深度值
    points = get_visible_faces_and_observe(points, observer_pos, look_direction, fov=45, max_distance=2000)

    visualize_depth_values_cubic(points, observer_pos, look_direction)

    total_depth = np.sum(points[:, 3:])
    print("更新后总深度:", total_depth)

    print(f"总点数: {len(points)}")

if __name__ == "__main__":
    main()