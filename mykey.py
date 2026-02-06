import sys
import cv2
import time
import airsim
import pygame
import numpy as np 
import pprint
from vln.agent_control import AirsimAgent
import os
from PIL import Image


def take_photo(drone, log_dir):
    # 检查log_dir是否存在, 如果不存在则创建
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 获取log_dir下的所有子文件夹
    subfolders = [f.name for f in os.scandir(log_dir) if f.is_dir()]

    # 如果没有子文件夹，则创建第一个文件夹，命名为0
    if not subfolders:
        new_folder_name = "0"
    else:
        # 找到子文件夹中的最大编号，创建下一个编号的文件夹
        existing_numbers = sorted(int(name) for name in subfolders if name.isdigit())
        new_folder_name = str(existing_numbers[-1] + 1)

    # 创建新的子文件夹
    new_folder_path = os.path.join(log_dir, new_folder_name)
    os.makedirs(new_folder_path)

    names = ['Forward', 'Forward Left', 'Left', 'Rear Left', 'Rear', 'Rear Right', 'Right', 'Forward Right', 'Top Down']
    for i in range(8):
        img1 = drone.get_front_image() # 获取前景图
        img1 = Image.fromarray(img1, 'RGB')
        img1.save(os.path.join(new_folder_path, '%d_%s.png' % (i, names[i])))
        drone.turnLeft(np.pi/4)

    i = 8
    img1 = drone.get_xyg_image(image_type=0, cameraID="3")
    img1 = Image.fromarray(img1, 'RGB')
    img1.save(os.path.join(new_folder_path, '%d_%s.png' % (i, names[i])))


def scan_photo(drone, log_dir):
    # 检查log_dir是否存在, 如果不存在则创建
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 获取log_dir下的所有子文件夹
    subfolders = [f.name for f in os.scandir(log_dir) if f.is_dir()]

    # 如果没有子文件夹，则创建第一个文件夹，命名为0
    if not subfolders:
        new_folder_name = "0"
    else:
        # 找到子文件夹中的最大编号，创建下一个编号的文件夹
        existing_numbers = sorted(int(name) for name in subfolders if name.isdigit())
        new_folder_name = str(existing_numbers[-1] + 1)

    # 创建新的子文件夹
    new_folder_path = os.path.join(log_dir, new_folder_name)
    os.makedirs(new_folder_path)
    str_time = time.strftime("%d%H%M%S", time.localtime(time.time()))
    img1 = drone.get_front_image()  # 获取前景图
    img1 = Image.fromarray(img1, 'RGB')
    img1.save(os.path.join(new_folder_path, f'{str_time}.png'))

# ---------- Pygame Settings ---------- #
pygame.init()
screen = pygame.display.set_mode((320, 240))  # 设置屏幕大小
pygame .display .set_caption('keyboard ctrl')  # 设置窗口标题
screen .fill((0, 0, 0))  # 屏幕填充RGB颜色
# ---------- AirSim Settings ---------- #
vehicle_name = ""
AirSim_client = airsim.MultirotorClient()
AirSim_client.confirmConnection()
AirSim_client.enableApiControl(True, vehicle_name=vehicle_name)
AirSim_client.armDisarm(True, vehicle_name=vehicle_name)
AirSim_client.takeoffAsync(vehicle_name=vehicle_name).join()

# 基础的控制速度(m/s)
vehicle_velocity = 40.0
# 设置加速度
speedup_ratio = 100.0
# 设置临时加速
speedup_flag = False
# 基础偏航速率
vehicle_yaw_rate = 84.0
i=0
image_types = {
            "scene": airsim.ImageType.Scene,
            "depth": airsim.ImageType.DepthVis,
            "seg": airsim.ImageType.Segmentation,
            "normals": airsim.ImageType.SurfaceNormals,
            "segmentation": airsim.ImageType.Segmentation,
            "disparity": airsim.ImageType.DisparityNormalized
        }

log_dir = 'SCAN'
os.makedirs(log_dir, exist_ok=True)
drone_handControl = AirsimAgent(None, None, None, AirSim_client)


while True:
    yaw_rate = 0.0
    velocity_x = 0.0
    velocity_y = 0.0
    velocity_z = 0.0
    time.sleep(0.02)
    for event in pygame .event .get():
        if event.type == pygame.QUIT:
            sys.exit()
    scan_wrapper = pygame.key.get_pressed()
    # 按下空格键加速10倍
    if scan_wrapper[pygame.K_SPACE]:
        scale_ratio = speedup_ratio
    else:
        scale_ratio = 1
    # 根据"A"和"D"设置yaw轴速率
    if scan_wrapper[pygame .K_a] or scan_wrapper[pygame .K_d]:
        yaw_rate = (scan_wrapper[pygame .K_d] - scan_wrapper[pygame .K_a]) * scale_ratio * vehicle_yaw_rate
        # 根据"Up"和"Down"设置pitch轴速率
    if scan_wrapper[pygame .K_UP] or scan_wrapper[pygame .K_DOWN]:
        velocity_x = (scan_wrapper[pygame .K_UP] - scan_wrapper[pygame .K_DOWN]) * scale_ratio
        # 根据"Left"和"Right"设置roll轴速率
    if scan_wrapper[pygame.K_LEFT] or scan_wrapper[pygame.K_RIGHT]:
        velocity_y = -(scan_wrapper[pygame.K_LEFT] - scan_wrapper[pygame.K_RIGHT]) * scale_ratio
    # 根据"W"和"S"设置z轴速率
    if scan_wrapper[pygame.K_w] or scan_wrapper[pygame.K_s]:
        velocity_z = -(scan_wrapper[pygame.K_w] - scan_wrapper[pygame.K_s]) * scale_ratio
    # 设置速度控制以及设置偏航控制
    AirSim_client.moveByVelocityBodyFrameAsync(vx=velocity_x, vy=velocity_y, vz=velocity_z, duration=0.02,
            yaw_mode=airsim .YawMode(True, yaw_or_rate=yaw_rate),vehicle_name=vehicle_name)
    # 按f 保存图片
    if scan_wrapper[pygame.K_f]:
        scan_photo(drone_handControl, log_dir)
    if scan_wrapper[pygame.K_p]:
        print(1)
        print(AirSim_client.simGetGroundTruthKinematics())
    


    # press 'Esc' to quit
    if scan_wrapper[pygame.K_ESCAPE]:
        pygame.quit()
        sys.exit()
