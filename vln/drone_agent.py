
import airsim
from scipy.spatial.transform import Rotation as R
import numpy as np
import math
import sys
import time
sys.path.append('..')
from airsim_utils.coord_transformation import quaternion2eularian_angles


AirSimImageType = {
    0: airsim.ImageType.Scene,
    1: airsim.ImageType.DepthPlanar,
    2: airsim.ImageType.DepthPerspective,
    3: airsim.ImageType.DepthVis,
    4: airsim.ImageType.DisparityNormalized,
    5: airsim.ImageType.Segmentation,
    6: airsim.ImageType.SurfaceNormals,
    7: airsim.ImageType.Infrared
}


class AirsimAgent:
    def __init__(self, cfg, query_func, prompt_template):
        self.query_func = query_func
        self.prompt_template = prompt_template
        self.landmarks = None
        self.client = airsim.MultirotorClient()
        self.actions = []
        self.states = []
        self.pos = np.array([6380.7007, -4178.9424, -10.422275])
        self.ori = np.array([0, 0, -90])
        self.step_distance = 10
        self.cfg = cfg
        self.rotation = R.from_euler("X", -np.pi).as_matrix()
        self.gt_height = 0.0
        self.velocity = 3
        self.panoid_yaws = [-np.pi / 2, -np.pi / 4, 0, np.pi / 4, np.pi / 2]
        self.init_config()

    def init_config(self):
        print("Initializing - init_config()")
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

        # self.client.moveToZAsync(300, 6).join()  # 上升到3m高度
        # # self.client.moveToPositionAsync(10, 0, -9, 7).join()
        # self.client.moveByRollPitchYawZAsync(0, 0, 0, -9, 2).join()
        time.sleep(2)

        cur_pos, cur_rot = self.get_current_state()
        print("initial position: {}, initial rotation: {}".format(cur_pos, cur_rot))

    def setVehiclePose(self, pose: np.ndarray) -> None:
        '''
        pose为[pos, rot]
        rot接受欧拉角或者四元数，
        如果len(pose) == 6,则认为rot为欧拉角,单位为弧度, [pitch, roll, yaw]
        如果len(pose) == 7,则认为rot为四元数, [x, y, z, w]
        '''
        pos = pose[:3]
        rot = pose[3:]

        if len(rot) == 3:
            rot = np.deg2rad(rot)
            air_rot = airsim.to_quaternion(rot[0], rot[1], rot[2])
        elif len(rot) == 4:
            air_rot = airsim.Quaternionr()
            air_rot.x_val = rot[0]
            air_rot.y_val = rot[1]
            air_rot.z_val = rot[2]
            air_rot.w_val = rot[3]
        else:
            raise ValueError(f"Expected rotation shape is (4,) or (3, ), got ({len(rot)},)")

        air_pos = airsim.Vector3r(pos[0], pos[1], pos[2])
        air_pose = airsim.Pose(air_pos, air_rot)
        self.client.simSetVehiclePose(air_pose, True)
        self.gt_height = float(air_pos.z_val)
        # print(f"gt z:{self.gt_height}")
        # print(f"set pose: {pos}")

    def global2body_rotation(self, global_rot, body_rot):
        # todo: assert shape
        global2body_rot = global_rot.dot(body_rot)
        return global2body_rot

    def bodyframe2worldframe(self, bodyframe):
        if type(bodyframe) is not np.ndarray:
            bodyframe = np.array(bodyframe)

        cur_pos, cur_rot = self.get_current_state()
        cur_rot = R.from_euler("XYZ", cur_rot).as_matrix()
        global2body_rot = self.global2body_rotation(cur_rot, self.rotation)
        worldframe = global2body_rot.dot(bodyframe) + cur_pos

        return worldframe

    # position is in current body frame
    def moveToPosition(self, position):
        pos_world = self.bodyframe2worldframe(position)
        print(f"next position in world coords: {pos_world}")
        self.client.moveToPositionAsync(float(pos_world[0]), float(pos_world[1]), float(pos_world[2]),
                                        self.velocity).join()

    def go_up(self, distance):
        """
        无人机向上移动
        :param distance: 移动距离，默认使用预设步长
        """
        if distance is None:
            distance = 10

        # 计算新位置（Z轴向上）
        self.pos[2] = self.pos[2] - distance

        # 移动到新位置
        self.MovetoPose(np.concatenate([self.pos, self.ori]).tolist())
        print(f"向上移动 {distance} 米")

    def go_down(self, distance):
        """
        无人机向下移动
        :param distance: 移动距离，默认使用预设步长
        """
        if distance is None:
            distance = 10

            # 获取当前位置

        # 计算新位置（Z轴向上）
        self.pos[2] = self.pos[2] + distance

        # 移动到新位置
        self.MovetoPose(np.concatenate([self.pos, self.ori]).tolist())
        print(f"向下移动 {distance} 米")


    def go_left(self, distance=None):
        """
        无人机向前移动（沿当前朝向）
        :param distance: 移动距离，默认使用预设步长
        """
        if distance is None:
            distance = self.step_distance

        # 计算前进方向（假设使用欧拉角，第三个角度为偏航角）
        yaw = self.ori[2]-90  # 偏航角

        # 计算新位置
        self.pos[0] = self.pos[0] + distance * np.cos(math.radians(yaw))
        self.pos[1] = self.pos[1] + distance * np.sin(math.radians(yaw))
        print(np.cos(math.radians(yaw)))
        print(np.sin(math.radians(yaw)))

        # 移动到新位置
        self.MovetoPose(np.concatenate([self.pos, self.ori]).tolist())
        print(f"向前移动 {distance} 米")


    def go_right(self, distance):
        """
        无人机向前移动（沿当前朝向）
        :param distance: 移动距离，默认使用预设步长
        """
        if distance is None:
            distance = self.step_distance

        # 计算前进方向（假设使用欧拉角，第三个角度为偏航角）
        yaw = self.ori[2]+90  # 偏航角

        # 计算新位置
        self.pos[0] = self.pos[0] + distance * np.cos(math.radians(yaw))
        self.pos[1] = self.pos[1] + distance * np.sin(math.radians(yaw))
        print(np.cos(math.radians(yaw)))
        print(np.sin(math.radians(yaw)))

        # 移动到新位置
        self.MovetoPose(np.concatenate([self.pos, self.ori]).tolist())
        print(f"向前移动 {distance} 米")

    def go_forward(self, distance):
        """
        无人机向前移动（沿当前朝向）
        :param distance: 移动距离，默认使用预设步长
        """
        if distance is None:
            distance = self.step_distance

        # 计算前进方向（假设使用欧拉角，第三个角度为偏航角）
        yaw = self.ori[2]  # 偏航角

        # 计算新位置
        self.pos[0] = self.pos[0] + distance * np.cos(math.radians(yaw))
        self.pos[1] = self.pos[1] + distance * np.sin(math.radians(yaw))

        # 移动到新位置
        self.MovetoPose(np.concatenate([self.pos, self.ori]).tolist())
        print(f"向前移动 {distance} 米")

    def turn_left(self, angle=None):
        """
        无人机向左转
        :param angle: 旋转角度，默认使用预设角度
        """
        if angle is None:
            angle = 90

        # 计算新的偏航角（逆时针旋转）
        self.ori[2] = self.ori[2] - angle

        # 移动到新位置（保持原位置）
        self.MovetoPose(np.concatenate([self.pos, self.ori]).tolist())
        print(f"向左转 {angle} 度")

    def turn_left_angle(self, angle):
        """
        无人机向左转
        :param angle: 旋转角度，默认使用预设角度
        """

        # 计算新的偏航角（逆时针旋转）
        self.ori[2] = self.ori[2] - angle

        # 移动到新位置（保持原位置）
        self.MovetoPose(np.concatenate([self.pos, self.ori]).tolist())
        print(f"向左转 {angle} 度")

    def turn_right(self, angle=None):
        """
        无人机向右转
        :param angle: 旋转角度，默认使用预设角度
        """
        if angle is None:
            angle = 90

        # 计算新的偏航角（逆时针旋转）
        self.ori[2] = self.ori[2] + angle

        # 移动到新位置（保持原位置）
        self.MovetoPose(np.concatenate([self.pos, self.ori]).tolist())
        print(f"向右转 {angle} 度")

    def take_action(self, action_label, x_interval, z_interval):
        if action_label == 0:
            self.go_up(z_interval)
        elif action_label == 1:
            self.go_down(z_interval)
        elif action_label == 2:
            self.turn_left()
        elif action_label == 3:
            self.turn_right()
        elif action_label == 4:
            self.go_forward(x_interval)
        elif action_label == 5:
            self.go_left(x_interval)
        elif action_label == 6:
            self.go_right(x_interval)
        else:
            print("无效的操作标签")

    def get_panorama_images(self, image_type=0):
        panorama_images = []
        new_yaws = []
        cur_pos, cur_rot = self.get_current_state()
        cur_yaw_body = -cur_rot[2]  # current yaw in body frame

        for angle in self.panoid_yaws:
            yaw = cur_yaw_body + angle
            self.client.moveByRollPitchYawZAsync(0, 0, float(yaw), float(cur_pos[2]), 2).join()
            image = self.get_front_image(image_type)
            panorama_images.append(image)

        self.client.moveByRollPitchYawZAsync(0, 0, float(cur_yaw_body), float(cur_pos[2]), 2).join()

        return panorama_images

    def get_front_image(self, image_type=0):
        # todo
        responses = self.client.simGetImages(
            [airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])  # if image_type == 0:
        response = responses[0]
        if image_type == 0:
            img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
            img_out = img1d.reshape(response.height, response.width, 3)
            img_out = img_out[:, :, [2, 1, 0]]
        else:
            # todo: add other image type
            img_out = None
        return img_out

    def MovetoPose(self, pose: np.ndarray) -> None:
        '''
        pose为[pos, rot]
        rot接受欧拉角或者四元数，
        如果len(pose) == 6,则认为rot为欧拉角,单位为弧度, [pitch, roll, yaw]
        如果len(pose) == 7,则认为rot为四元数, [x, y, z, w]
        '''
        pos = pose[:3]
        rot = pose[3:]

        if len(rot) == 3:
            rot = np.deg2rad(rot)
            air_rot = airsim.to_quaternion(rot[0], rot[1], rot[2])
        elif len(rot) == 4:
            air_rot = airsim.Quaternionr()
            air_rot.x_val = rot[0]
            air_rot.y_val = rot[1]
            air_rot.z_val = rot[2]
            air_rot.w_val = rot[3]
        else:
            raise ValueError(f"Expected rotation shape is (4,) or (3, ), got ({len(rot)},)")

        air_pos = airsim.Vector3r(pos[0], pos[1], pos[2])
        air_pose = airsim.Pose(air_pos, air_rot)
        self.client.simSetVehiclePose(air_pose, True)
        self.gt_height = float(air_pos.z_val)

    def get_observation(self):
        # todo
        # "1"rgb “0” depth

        # 获取rgb
        responses = self.client.simGetImages(
            [airsim.ImageRequest(1, airsim.ImageType.Scene, False, False)])  # if image_type == 0:
        response = responses[0]
        img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
        img_out1 = img1d.reshape(response.height, response.width, 3)

        # 获取depth
        responses2 = self.client.simGetImages(
            [airsim.ImageRequest(0, airsim.ImageType.DepthPlanar, True, False)])
        img_out2 = np.array(responses2[0].image_data_float).reshape(responses2[0].height, responses2[0].width)
        img_depth_vis = img_out2 / 100
        img_depth_vis[img_depth_vis > 1] = 1.
        img_out2 = img_depth_vis * 100

        return img_out1, img_out2



    def get_xyg_image(self, image_type, cameraID):
        # todo
        # "3"地面 “4”后面 “2”前面
        if image_type == 0:
            responses = self.client.simGetImages(
                [airsim.ImageRequest(cameraID, airsim.ImageType.Scene, False, False)])  # if image_type == 0:
            response = responses[0]

            img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
            img_out = img1d.reshape(response.height, response.width, 3)
            # img_out = img_out[:, :, [2, 1, 0]]

        elif image_type == 1:
            # todo: add other image type

            # 获取DepthVis深度可视图
            responses = self.client.simGetImages([
                airsim.ImageRequest(cameraID, airsim.ImageType.DepthPlanar, True, False)])
            img_depth_planar = np.array(responses[0].image_data_float).reshape(responses[0].height, responses[0].width)
            # 2. 距离100米以上的像素设为白色（此距离阈值可以根据自己的需求来更改）
            img_depth_vis = img_depth_planar / 100
            img_depth_vis[img_depth_vis > 1] = 1.
            # # 3. 转换为整形
            img_out = (img_depth_vis * 255).astype(np.uint8)

            # responses = self.client.simGetImages([
            #     airsim.ImageRequest('front_center', airsim.ImageType.DepthVis, False, False)])
            # img_1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
            # img_depthvis_bgr = img_1d.reshape(responses[0].height, responses[0].width, 3)

            # responses = self.client.simGetImages(
            #     [airsim.ImageRequest("0", airsim.ImageType.DepthVis, True, False)])  # if image_type == 0:
            # response = responses[0]

            # img1d = (np.array(response.image_data_float)*255).astype(int)
            # img_out = img1d.reshape(response.height, response.width)

        elif image_type == 11:

            # 获取DepthVis深度可视图
            responses = self.client.simGetImages([
                airsim.ImageRequest(cameraID, airsim.ImageType.DepthPerspective, True, False)])
            img_out = np.array(responses[0].image_data_float).reshape(responses[0].height, responses[0].width)
            # print(img_out)

            # 2. 距离100米以上的像素设为白色（此距离阈值可以根据自己的需求来更改）
            img_depth_vis = img_out / 100
            img_depth_vis[img_depth_vis > 1] = 1.
            # 3. 转换为整形
            img_out = (img_depth_vis * 255).astype(np.uint8)
        elif image_type == 2:
            # responses = self.client.simGetImages([airsim.ImageRequest(cameraID, airsim.ImageType.Segmentation, pixels_as_float=False, compress=True)])
            # # img_depth_planar = np.array(responses[0].image_data_float).reshape(responses[0].height, responses[0].width)
            #
            #
            # responses = self.client.simGetImages([airsim.ImageRequest(cameraID, airsim.ImageType.Segmentation, False, False)])
            # img1d = np.fromstring(responses[0].image_data_uint8, dtype=np.uint8)  # get numpy array
            # img_rgb = img1d.reshape(responses[0].height, responses[0].width, 3)  # reshape array to 3 channel image array H X W X 3

            responses = self.client.simGetImages([
                airsim.ImageRequest(cameraID, airsim.ImageType.Segmentation, pixels_as_float=False, compress=True),
                airsim.ImageRequest(cameraID, airsim.ImageType.Segmentation, pixels_as_float=False, compress=False), ])

            f = open('imgs/seg.png', 'wb')
            f.write(responses[0].image_data_uint8)
            f.close()

            img1d = np.fromstring(responses[1].image_data_uint8, dtype=np.uint8)  # get numpy array
            img_out = img1d.reshape(responses[1].height, responses[1].width, 3)

            # img_out = None

        else:
            return None
        return img_out

    def get_current_state(self):
        # get world frame pos and orientation
        # orientation is in roll, pitch, yaw format
        state = self.client.simGetGroundTruthKinematics()
        pos = state.position.to_numpy_array()
        ori = quaternion2eularian_angles(state.orientation)
        # ori = state.orientation

        return pos, ori

    def get_obstacle_dis(self):
        '''
        获取正前方障碍物距离
        '''
        depth = self.get_xyg_image(image_type=1, cameraID="0")
        time.sleep(0.1)
        width,height = depth.shape
        distance = depth[2*width//5:3*width//5,height//2+1:3*height//5]/255*100
        # print("center:",depth[width//2,height//2]/255*100)
        return np.min(distance)


if __name__ == "__main__":
    drone = AirsimAgent(None, None, None)
    drone.get_panorama_images()
