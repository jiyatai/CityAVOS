import numpy as np
import math
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import DBSCAN
from llm_agent import chat_with_llm_images
from update_uncertainty_map import uncertainty_map_update

# 定义动作集合
action_set = ["Go Up", "Go Down", "Turn Left", "Turn Right", "Go Forward", "Go Left", "Go Right", "Stop"]


def rad_to_deg(rad):
    angle_rad = np.deg2rad(rad)
    # 计算 x 和 y 分量
    x = np.cos(angle_rad)
    y = np.sin(angle_rad)
    # 创建方向向量，z 分量为 0
    direction = np.array([x, y, 0])
    return direction


def action_value_choose(drone, points, scene, total_uncertainty):
    points_now = points.copy()
    uncertainty_now = np.sum(points[:, 3:])
    total_depth = [0, 0, 0, 0, 0, 0, 0]
    [x_min, x_max, y_min, y_max, z_min, z_max, step_x, step_z] = [scene["x_min"], scene["x_max"], scene["y_min"],
                                                                  scene["y_max"], scene["z_min"], scene["z_max"],
                                                                  scene["step_x"], scene["step_z"]]

    # 检查位置是否超出边界的函数
    def is_out_of_bounds(pos):
        return (pos[0] < x_min or pos[0] > x_max or
                pos[1] < y_min or pos[1] > y_max or
                pos[2] < z_min or pos[2] > z_max)

    # 位置变换和观测
    positions = [
        drone.pos * [1, -1, -1] + [0, 0, step_z],  # 上
        drone.pos * [1, -1, -1] - [0, 0, step_z],  # 下
        drone.pos * [1, -1, -1],
        drone.pos * [1, -1, -1],
        drone.pos * [1, -1, -1] + [step_x * np.cos(math.radians(drone.ori[2])),
                                   -step_x * np.sin(math.radians(drone.ori[2])), 0],
        drone.pos * [1, -1, -1] + [step_x * np.cos(math.radians(drone.ori[2] - 90)),
                                   -step_x * np.sin(math.radians(drone.ori[2] - 90)), 0],
        drone.pos * [1, -1, -1] + [step_x * np.cos(math.radians(drone.ori[2] + 90)),
                                   -step_x * np.sin(math.radians(drone.ori[2] + 90)), 0]
    ]

    look_directions = [
        np.round(rad_to_deg(drone.ori[2])).astype(int),  # 上
        np.round(rad_to_deg(drone.ori[2])).astype(int),  # 下
        np.round(rad_to_deg(drone.ori[2] - 90)).astype(int),  # 左
        np.round(rad_to_deg(drone.ori[2] + 90)).astype(int),  # 右
        np.round(rad_to_deg(drone.ori[2])).astype(int),  # 前进
        np.round(rad_to_deg(drone.ori[2])).astype(int),  # 前进
        np.round(rad_to_deg(drone.ori[2])).astype(int)  # 前进
    ]

    # 遍历并计算深度
    for i in range(7):
        # 检查是否超出边界
        if is_out_of_bounds(positions[i]):
            total_depth[i] = 10000
        else:
            # 注意：这里调用了 update_uncertainty_map 中的函数
            points_now = uncertainty_map_update(points, positions[i], look_directions[i], step_x, fov=60,
                                                       max_distance=1000)
            total_depth[i] = np.sum(points_now[:, 3:])

    index = np.argmin(total_depth)

    temp = (uncertainty_now - total_depth[index]) / total_uncertainty
    print(temp)

    if temp > 0.1:
        return action_set[np.argmin(total_depth)]
    else:
        return None

def action_to_pos(drone, pos_target, scene):
    """
    根据无人机当前位置和目标位置，计算下一步的最佳动作索引。
    Action Set: ["Go Up", "Go Down", "Turn Left", "Turn Right", "Go Forward", "Go Left", "Go Right"]
    Indices:    0,       1,         2,           3,            4,            5,         6
    """

    # 1. 提取场景边界和步长
    x_min, x_max = scene["x_min"], scene["x_max"]
    y_min, y_max = scene["y_min"], scene["y_max"]
    z_min, z_max = scene["z_min"], scene["z_max"]
    step_x = scene["step_x"]  # 假设水平移动统一使用 step_x
    step_z = scene["step_z"]

    # 辅助函数：检查位置是否在边界内
    def is_valid(pos):
        return (x_min <= pos[0] <= x_max and
                y_min <= pos[1] <= y_max and
                z_min <= pos[2] <= z_max)

    # 获取当前状态
    curr_pos = np.array(drone.pos * [1, -1, -1])
    curr_yaw = drone.ori[2]
    target_pos = np.array(pos_target)

    # ---------------------------------------------------------
    # 第一步：调整高度 (Z轴) - 优先级最高
    # ---------------------------------------------------------
    z_diff = target_pos[2] - curr_pos[2]
    z_threshold = step_z / 2.0  # 设置阈值防止震荡

    if z_diff > z_threshold:
        # 预测向上移动后的位置
        next_pos_up = curr_pos.copy()
        next_pos_up[2] += step_z
        # 只有在不出界的情况下才执行
        if is_valid(next_pos_up):
            return 0  # Go Up

    if z_diff < -z_threshold:
        # 预测向下移动后的位置
        next_pos_down = curr_pos.copy()
        next_pos_down[2] -= step_z
        if is_valid(next_pos_down):
            return 1  # Go Down

    # ---------------------------------------------------------
    # 第二步：转向
    # ---------------------------------------------------------

    # 计算目标相对于当前位置的角度
    dx = target_pos[0] - curr_pos[0]
    dy = target_pos[1] - curr_pos[1]
    target_angle = math.degrees(math.atan2(dy, dx))

    # 计算角度差 (-180 到 180)
    angle_diff = target_angle + curr_yaw
    while angle_diff > 180: angle_diff -= 360
    while angle_diff < -180: angle_diff += 360

    if angle_diff < -45:
        return 3  # Turn Left
    if angle_diff > 45:
        return 2  # Turn Right


    # ---------------------------------------------------------
    # 第三步：移动
    # ---------------------------------------------------------

    # 计算当前水平距离
    curr_dist_xy = np.linalg.norm(target_pos[:2] - curr_pos[:2])

    # 如果已经非常接近目标（小于半个步长），则停止或微调（此处可视需求返回特定动作，这里假设继续尝试对齐）
    if curr_dist_xy < step_x / 2.0:
        # 如果高度也对齐了，理论上应该悬停，这里默认不做动作或继续保持
        return 4

    moves = []

    # 定义移动逻辑 (需与 Drone 类中的数学逻辑保持一致)
    # Forward
    dx_f = step_x * np.cos(math.radians(curr_yaw))
    dy_f = step_x * np.sin(math.radians(curr_yaw))
    pos_fwd = curr_pos.copy()
    pos_fwd[0] += dx_f
    pos_fwd[1] -= dy_f
    moves.append({'action': 4, 'pos': pos_fwd, 'dist': np.linalg.norm(target_pos[:2] - pos_fwd[:2])})

    # Left (Yaw - 90)
    dx_l = step_x * np.cos(math.radians(curr_yaw - 90))
    dy_l = step_x * np.sin(math.radians(curr_yaw - 90))
    pos_left = curr_pos.copy()
    pos_left[0] += dx_l
    pos_left[1] -= dy_l
    moves.append({'action': 5, 'pos': pos_left, 'dist': np.linalg.norm(target_pos[:2] - pos_left[:2])})

    # Right (Yaw + 90)
    dx_r = step_x * np.cos(math.radians(curr_yaw + 90))
    dy_r = step_x * np.sin(math.radians(curr_yaw + 90))
    pos_right = curr_pos.copy()
    pos_right[0] += dx_r
    pos_right[1] -= dy_r
    moves.append({'action': 6, 'pos': pos_right, 'dist': np.linalg.norm(target_pos[:2] - pos_right[:2])})

    # 过滤掉会导致出界的动作
    valid_moves = [m for m in moves if is_valid(m['pos'])]

    if valid_moves:
        # 找出距离目标最近的动作
        best_move = min(valid_moves, key=lambda x: x['dist'])

        # 如果最佳移动能显著减小距离（比当前距离更近），则执行该移动
        # 这里的 "显著" 可以只是简单的 < curr_dist_xy，或者加一点余量
        if best_move['dist'] < curr_dist_xy:
            return best_move['action']



def calculate_angle(pos_drone, pos_target):
    delta_x = pos_target[0] - pos_drone[0]
    delta_y = pos_target[1] - pos_drone[1]
    angle = math.degrees(math.atan2(delta_y, delta_x))  # 计算目标点的角度
    return angle % 360  # 确保角度在0-360度之间


def plot_drone_path(drone_positions):
    """
    绘制无人机移动路径
    """
    # 提取x, y, z坐标
    x_coords = [pos[0] for pos in drone_positions]
    y_coords = [pos[1] for pos in drone_positions]
    z_coords = [pos[2] for pos in drone_positions]

    # 创建3D图形
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制路径线
    ax.plot(x_coords, y_coords, z_coords, 'b-', linewidth=2, label='Drone Path')

    # 绘制起点和终点
    ax.scatter(x_coords[0], y_coords[0], z_coords[0], color='green', s=100, label='Start', marker='^')
    ax.scatter(x_coords[-1], y_coords[-1], z_coords[-1], color='red', s=100, label='End', marker='v')

    # 标记每个位置点
    ax.scatter(x_coords, y_coords, z_coords, color='purple', s=50, alpha=0.6)

    # 设置标签和标题
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Drone Movement Path')
    ax.legend()

    # 添加网格
    ax.grid(True)

    plt.tight_layout()
    plt.show()


def get_action_label():
    while True:
        try:
            # 获取用户输入
            action_label = int(input("请输入动作标签 (0-6): "))  # Prompt for input
            if 0 <= action_label <= 6:  # Validate the input
                return action_label
            else:
                print("无效输入，请输入0到6之间的数字。")  # Invalid input message
        except ValueError:
            print("无效输入，请输入一个数字。")  # Handle non-integer input


def get_action_from_llm(adviser_cognitive_map, adviser_uncertainty_map, target_text, target_image_path, rgb_path,
                        Attraction_Value):
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # 构建 prompt
            if adviser_uncertainty_map:

                if adviser_cognitive_map:
                    prompt = f"""  
                                
                                You are operating a drone to search for a visual target in an urban space. For each step, you will receive the following inputs:
                                -Image_RGB_inputs: An RGB image representing your current view, which is the first image.
                                -Image_Object: An image of the object you are searching for, which is the second image.
                                -Text_Object: The object you are searching for is {target_text}.
                                Guidelines:
                                -Exploitation advice: Select action "{adviser_cognitive_map}" helps you to approach the target with a probability of {Attraction_Value * 100}%.
                                -Exploration advice: Select action "{adviser_uncertainty_map}" helps you explore the surrounding environment, which is more important.
                                Select your action follow the guidelines above. Only return the name of the action you selected.
                                """
                else:
                    prompt = f"""  
                    
                                You are operating a drone to search for a visual target in an urban space. For each step, you will receive the following inputs:
                                -Image_RGB_inputs: An RGB image representing your current view, which is the first image.
                                -Image_Object: An image of the object you are searching for, which is the second image.
                                -Text_Object: The object you are searching for is {target_text}.
                                Guidelines:
                                -Exploration advice: Select action "{adviser_uncertainty_map}" helps you search for the target.
                                -You can follow the exploration advice or you can also follow your own ideas.
                                Select your action follow the guidelines above. Only return the name of the action you selected.
                                """
            else:
                if adviser_cognitive_map:
                    prompt = f"""  
                                You are operating a drone to search for a visual target in an urban space. For each step, you will receive the following inputs:
                                -Image_RGB_inputs: An RGB image representing your current view, which is the first image.
                                -Image_Object: An image of the object you are searching for, which is the second image.
                                -Text_Object: The object you are searching for is {target_text}.
                                Guidelines:
                                -Exploitation advice: Select action "{adviser_cognitive_map}" helps you to approach the target  with a probability of {Attraction_Value * 100}%.
                                -You can follow the exploitation advice or you can also follow your own ideas.
                                Select your action following guidelines above, Only return the name of the action you selected.
                                """
                else:
                    prompt = f"""  
                                You are operating a drone to search for a visual target in an urban space. For each step, you will receive the following inputs:
                                -Image_RGB_inputs: An RGB image representing your current view, which is the first image.
                                -Image_Object: An image of the object you are searching for, which is the second image.
                                -Text_Object: The object you are searching for is {target_text}.
                                Guidelines:
                                -Follow your own ideas.
                                -You can select on action from ("Turn Left", "Turn Right", "Go Forward", "Go Left", "Go Right")
                                Select your action following guidelines above, Only return the name of the action you selected.
                                """


            # 提取返回的动作
            print(prompt)
            chosen_action = chat_with_llm_images(prompt, [("./" + target_image_path), rgb_path])
            print(chosen_action)

            # 验证动作是否在动作集中
            if chosen_action in action_set:
                # 返回动作在列表中的索引
                return action_set.index(chosen_action)
            else:
                print(f"Attempt {attempt + 1}: Invalid action chosen. Retrying...")
                time.sleep(1)  # 短暂延迟后重试

        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(2)  # 错误后延迟重试

        # 如果重试仍失败，返回默认动作的索引
    print("Failed to get a valid action after multiple attempts. Defaulting to 'Stop'.")
    return action_set.index('Stop')




def find_max_cluster_center(cognitive_map, eps=1.0, min_samples=3):
    """
    从cognitive_map中提取第四维值最大的点，并对这些点进行聚类，返回最大簇的聚类中心。

    参数:
        cognitive_map: np.array, 形状为 (N, 4)
        eps: DBSCAN的半径参数
        min_samples: DBSCAN的最小样本数参数

    返回:
        max_value: 第四维的最大值
        cluster_center: 最大簇的中心坐标 (x, y, z)
    """

    # 1. 输入校验
    if cognitive_map is None or len(cognitive_map) == 0:
        return None, None

    # 2. 找出第四维的最大值
    max_value = np.max(cognitive_map[:, 3])

    # 3. 找出具有最大值的点 (使用 np.isclose 解决浮点数精度问题)
    # atol 是绝对容差，根据数据量级调整，通常 1e-8 足够
    mask = np.isclose(cognitive_map[:, 3], max_value, atol=1e-8)
    max_value_points = cognitive_map[mask][:, :3]

    # 4. 边界情况处理：如果没有点或点很少
    num_points = len(max_value_points)
    if num_points == 0:
        return max_value, None

    # 如果点数少于聚类要求的最小样本数，直接计算这些点的中心作为结果
    # 避免 DBSCAN 将其标记为噪声导致返回 None
    if num_points < min_samples:
        return max_value, np.mean(max_value_points, axis=0)

    # 5. 使用 DBSCAN 聚类
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(max_value_points)

    # 6. 找出最大簇
    unique_clusters = np.unique(clusters)

    # 过滤掉噪声点 (-1)
    valid_clusters = unique_clusters[unique_clusters != -1]

    if len(valid_clusters) == 0:
        # 情况A: 所有点都被标记为噪声 (-1)
        # 策略: 这种情况下，说明点很分散。
        # 建议直接返回所有最大值点的几何中心作为“最大簇中心”的兜底
        return max_value, np.mean(max_value_points, axis=0)

    # 情况B: 存在有效的簇，寻找包含点数最多的簇
    best_cluster_label = -1
    max_size = -1

    for cluster_label in valid_clusters:
        size = np.sum(clusters == cluster_label)
        if size > max_size:
            max_size = size
            best_cluster_label = cluster_label

    # 7. 计算最大簇的聚类中心
    largest_cluster_points = max_value_points[clusters == best_cluster_label]
    cluster_center = np.mean(largest_cluster_points, axis=0)

    return max_value, cluster_center