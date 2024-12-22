import os
import numpy as np
import pybullet as p
import pybullet_data
import time
from pybullet_utils import bullet_client
from scipy.spatial.transform import Rotation as R
from stable_baselines3 import PPO
from env import Env
from abc import ABC, abstractmethod

class BaseAlgorithm(ABC):
    @abstractmethod 
    def get_action(self, observation):
        """
        输入观测值，返回动作
        Args:
            observation: numpy array of shape (1, 12) 包含:
                - 6个关节角度 (归一化到[0,1])
                - 3个目标位置坐标
                - 3个障碍物位置坐标
        Returns:
            action: numpy array of shape (6,) 范围在[-1,1]之间
        """
        pass

class MyCustomAlgorithm(BaseAlgorithm):
    def __init__(self):
        # 自定义初始化
        pass
        
    def get_action(self, observation):
        # 输入观测值，返回动作
        action = np.random.uniform(-1, 1, 6)
        return action

# 示例：使用PPO预训练模型
class PPOAlgorithm(BaseAlgorithm):
    def __init__(self):
        self.model = PPO.load("ppo_model.zip", device="mps")

    def get_velocity(self):
        '''获取目标物速度'''
        linear_velocity, angular_velocity = p.getBaseVelocity(2)
        return linear_velocity

    def get_expected_pos(self,target_initpos):
        '''获取目标物预期位置(80步后)'''
        weight=0.083 #权重，指每一步目标物移动的距离=速度*0.083
        x_init=target_initpos[0]
        z_init=target_initpos[2]
        linear_velocity=self.get_velocity()
        x_velocity=linear_velocity[0]
        z_velocity=linear_velocity[2]
        x_exp=x_init+x_velocity*80*weight #不考虑反弹，因为70步根本不可能反弹
        if z_init+z_velocity*weight*80<0.1:
            #目标物高度如果过低很容易发生反弹，还是要考虑一下的
            step_stage_z=abs((0.1-z_init)/(z_velocity*weight))
            z_exp=0.1+abs((80-step_stage_z)*(weight*z_velocity))
        else:
            z_exp=z_init+z_velocity*80*weight
        return [x_exp,target_initpos[1],z_exp]
    
    def get_pos(self):  
        gripper_pos = p.getLinkState(0, 6)[0]
        relative_position = np.array([0, 0, 0.15])
        rotation = R.from_quat(p.getLinkState(0, 7)[1])
        rotated_relative_position = rotation.apply(relative_position)
        gripper_centre_pos = np.array(gripper_pos) + rotated_relative_position
        return gripper_centre_pos
    
    def get_observation(self):
        gripper_centre_pos = self.get_pos()
        obs_gripper_centre_pos = np.array([(gripper_centre_pos[0]+0.922)/1.844,
                                    (gripper_centre_pos[1]+0.922)/1.844,
                                    (gripper_centre_pos[2]+0.5)/1],dtype=np.float32)
        dis_to_target=np.linalg.norm(gripper_centre_pos-self.target_pos)
        target_position = self.target_pos
        target_position[0]=target_position[0]+0.5
        target_position[1]=(target_position[1]-0.8)*10
        target_position[2]=(target_position[2]-0.1)*2.5
        obstacle1_position = self.barrier_pos
        obstacle1_position[0]=(obstacle1_position[0]+0.4)*1.25
        obstacle1_position[1]=obstacle1_position[1]-0.6
        obstacle1_position[2]=(obstacle1_position[2]-0.1)*5
        self.observation = np.hstack((obs_gripper_centre_pos,self.joint_angles, target_position, obstacle1_position,dis_to_target)).flatten()
        self.observation = self.observation.astype(np.float32)
        self.observation = self.observation.reshape(1,15)
        return self.observation
    
    
    def get_action(self,observation):
        self.joint_angles=observation[0][0:5]
        self.target_pos=observation[0][6:9]
        self.barrier_pos=observation[0][9:12]
        action, _ = self.model.predict(self.get_observation())
        action=np.clip(action,-1,1)
        action=np.append(action,0)
        return action
    

