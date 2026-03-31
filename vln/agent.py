# from vln.env import get_nav_from_actions
# from vln.prompt_builder import get_navigation_lines
import airsim
from scipy.spatial.transform import Rotation as R
import numpy as np
import cv2
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

class Agent:
    def __init__(self, query_func, env, instance, prompt_template):
        self.query_func = query_func
        self.env = env
        self.instance = instance
        self.dataset_name = instance['dataset_name']
        self.landmarks = instance['landmarks']
        self.traffic_flow = instance.get('traffic_flow')
        self.init_prompt = prompt_template.format(instance['navigation_text'])
        self.is_map2seq = instance['is_map2seq']

    def run(self, max_steps, verbatim=False):
        actions = ['init']
        if self.dataset_name == 'map2seq':
            actions.append('forward')

        navigation_lines = list()
        is_actions = list()

        query_count = 0
        nav = get_nav_from_actions(actions, self.instance, self.env)

        step_id = 0
        hints_action = None
        while step_id <= max_steps:
            if verbatim:
                print('Number of Steps:', len(nav.actions))

            new_navigation_lines, new_is_actions = get_navigation_lines(nav,
                                                                        self.env,
                                                                        self.landmarks,
                                                                        self.traffic_flow,
                                                                        step_id=step_id
                                                                        )
            navigation_lines = navigation_lines[:-1] + new_navigation_lines
            is_actions = is_actions[:-1] + new_is_actions
            step_id = len(nav.actions)

            navigation_text = '\n'.join(navigation_lines)
            prompt = self.init_prompt + navigation_text
            # print(navigation_text)

            action, queried_api, hints_action = self.query_next_action(prompt, hints_action, verbatim)
            query_count += queried_api

            action = nav.validate_action(action)

            if action == 'stop':
                nav.step(action)
                prompt += f' {action}\n'
                break

            nav.step(action)
            if verbatim:
                print('Validated action', action)

                # print('actions', actions)
                print('query_count', query_count)

        del hints_action

        new_navigation_lines, new_is_actions = get_navigation_lines(nav,
                                                                    self.env,
                                                                    self.landmarks,
                                                                    self.traffic_flow,
                                                                    step_id=step_id,
                                                                    )
        navigation_lines = navigation_lines[:-1] + new_navigation_lines
        is_actions = is_actions[:-1] + new_is_actions

        return nav, navigation_lines, is_actions, query_count

    def query_next_action(self, prompt, hints=None, verbatim=True):
        output, queried_api, hints = self.query_func(prompt, hints)
        try:
            predicted = self.extract_next_action(output, prompt)
        except Exception as e:
            print('extract_next_action error: ', e)
            print('returned "forward" instead')
            predicted_sequence = output[len(prompt):]
            predicted = 'forward'
            print('predicted_sequence', predicted_sequence)
        if verbatim:
            print('Predicted Action:', predicted)
        return predicted, queried_api, hints

    @staticmethod
    def extract_next_action(output, prompt):
        assert output.startswith(prompt)
        predicted_sequence = output[len(prompt):]
        predicted = predicted_sequence.strip().split()[0]
        predicted = predicted.lower()
        if predicted in {'forward', 'left', 'right', 'turn_around', 'stop'}:
            return predicted

        predicted = ''.join([i for i in predicted if i.isalpha()])
        if predicted == 'turn':
            next_words = predicted_sequence.strip().split()[1:]
            next_predicted = next_words[0]
            next_predicted = ''.join([i for i in next_predicted if i.isalpha()])
            next_predicted = next_predicted.lower()
            predicted += ' ' + next_predicted
        return predicted


class LLMAgent(Agent):

    def __init__(self, llm, env, instance, prompt_template):
        self.llm = llm
        self.env = env
        self.instance = instance
        self.dataset_name = instance['dataset_name']

        self.landmarks = instance['landmarks']
        self.traffic_flow = instance.get('traffic_flow')

        self.init_prompt = prompt_template.format(instance['navigation_text'])

        cache_key = f'{self.dataset_name}_{instance["idx"]}'

        def query_func(prompt, hints=None):
            queried_api = 0
            output = self.llm.get_cache(prompt, cache_key)
            if output is None:
                print('query API')
                output = self.llm.query_api(prompt)
                queried_api += 1
                self.llm.add_to_cache(output, cache_key)
                print('api sequence')
            return output, queried_api, dict()

        super().__init__(query_func, env, instance, prompt_template)


class AirsimAgent:
    def __init__(self, cfg, query_func, prompt_template):
        self.query_func = query_func
        self.prompt_template = prompt_template
        self.landmarks = None
        self.client = airsim.MultirotorClient()
        self.actions = []
        self.states = []
        self.cfg = cfg
        self.rotation = R.from_euler("X", -np.pi).as_matrix()
        self.velocity = 4
        self.panoid_yaws = [-np.pi/2, -np.pi/4, 0, np.pi/4, np.pi/2]

        self.init_config()
    def init_config(self):
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

        # self.client.moveToZAsync(-5, 1).join()  # 上升到3m高度
        self.client.moveToPositionAsync(10, 0, -9, 7).join()
        self.client.moveByRollPitchYawZAsync(0, 0, 0, -9, 2).join()
        time.sleep(2)

        cur_pos, cur_rot = self.get_current_state()
        print("initial position: {}, initial rotation: {}".format(cur_pos, cur_rot))

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
        print(pos_world)
        self.client.moveToPositionAsync(float(pos_world[0]), float(pos_world[1]), float(pos_world[2]), self.velocity).join()

    def moveBackForth(self, distance):
        pos = [distance, 0, 0]
        self.moveToPosition(pos)

    def moveHorizontal(self, distance):
        pos = [0, distance, 0]
        self.moveToPosition(pos)

    def moveVertical(self, distance):
        pos = [0, 0, distance]
        self.moveToPosition(pos)

    # yaw is in current body frame, radian unit
    def moveByYaw(self, yaw):
        cur_pos, cur_rot = self.get_current_state()     # get rotation in world frame
        cur_yaw_body = -cur_rot[2]   # current yaw in local body frame
        new_yaw_body = cur_yaw_body+yaw

        # print("new yaw body: {}, current yaw: {}".format(new_yaw_body, cur_yaw_body))

        # moveByRollPitchYaw is on current body frame
        self.client.moveByRollPitchYawZAsync(0, 0, float(new_yaw_body), float(cur_pos[2]), 2).join()

    def get_panorama_images(self, image_type=0):
        panorama_images = []
        new_yaws = []
        cur_pos, cur_rot = self.get_current_state()
        cur_yaw_body = -cur_rot[2]   # current yaw in body frame

        for angle in self.panoid_yaws:
            yaw = cur_yaw_body + angle
            self.client.moveByRollPitchYawZAsync(0, 0, float(yaw), float(cur_pos[2]), 2).join()
            image = self.get_front_image(image_type)
            panorama_images.append(image)

        self.client.moveByRollPitchYawZAsync(0, 0, float(cur_yaw_body), float(cur_pos[2]), 2).join()

        return panorama_images

    def get_front_image(self, image_type=0):
        # todo
        responses = self.client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])        # if image_type == 0:
        response = responses[0]
        if image_type == 0:
            img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
            img_out = img1d.reshape(response.height, response.width, 3)
        else:
            # todo: add other image type
            img_out = None
        return img_out

    def get_xyg_image(self, image_type, cameraID):
        # todo
        # "3"地面 “4”后面 “2”前面
        if image_type == 0:
            responses = self.client.simGetImages([airsim.ImageRequest(cameraID, airsim.ImageType.Scene, False, False)])        # if image_type == 0:
            response = responses[0]

            img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
            img_out = img1d.reshape(response.height, response.width, 3)
        elif image_type == 1:
            # todo: add other image type

            # 获取DepthVis深度可视图
            responses = self.client.simGetImages([
                airsim.ImageRequest(cameraID, airsim.ImageType.DepthPlanar, True, False)])
            img_depth_planar = np.array(responses[0].image_data_float).reshape(responses[0].height, responses[0].width)
            # 2. 距离100米以上的像素设为白色（此距离阈值可以根据自己的需求来更改）
            img_depth_vis = img_depth_planar / 100
            img_depth_vis[img_depth_vis > 1] = 1.
            # 3. 转换为整形
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
        else:
            return None
        return img_out

    def get_current_state(self):
        # get world frame pos and orientation
        # orientation is in roll, pitch, yaw format
        state = self.client.simGetGroundTruthKinematics()
        pos = state.position.to_numpy_array()
        ori = quaternion2eularian_angles(state.orientation)

        return pos, ori



    # def update_coord_rot(self, axis, angle, intrinsic_rot=True):
    #     if intrinsic_rot:
    #         assert axis in ["X", "Y", "Z"]
    #         rot_mat = R.from_euler(axis, angle).as_matrix()
    #         self.coord_rot = self.coord_rot.dot(rot_mat)
    #     else:
    #         assert axis in ["x", "y", "z"]
    #         rot_mat = R.from_euler(axis, angle).as_matrix()
    #         self.coord_rot = rot_mat.dot(self.coord_rot)


if __name__ == "__main__":
    drone = AirsimAgent(None, None, None)
    drone.get_panorama_images()
    # drone.moveByYaw(np.pi/4)
    # img1 = drone.get_front_image()
    #
    # drone.moveBackForth(5)
    # img2 = drone.get_front_image()
    #
    # drone.moveHorizontal(5)
    # img3 = drone.get_front_image()
    #
    # drone.moveBackForth(-5)
    # drone.get_front_image()
    #
    # drone.moveHorizontal(-5)
    # drone.get_front_image()

    # import matplotlib.pyplot as plt
    # img = plt.imread("../figures/scene.png")
    # # img = cv2.imread("../figures/scene.png")
    # plt.imshow(img)
    # plt.show()

    # cv2.imshow("img", img)
    # cv2.waitKey()
