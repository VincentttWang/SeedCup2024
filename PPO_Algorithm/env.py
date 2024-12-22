import os
import gymnasium
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
import math
from pybullet_utils import bullet_client
from scipy.spatial.transform import Rotation as R

class Env(gymnasium.Env):
    def __init__(self,is_senior=False,seed=102, gui=False):
        super(Env, self).__init__()
        np.random.seed(seed)
        self.seed=seed
        self.is_senior = is_senior
        self.step_num = 0
        self.max_steps = 200
        self.p = bullet_client.BulletClient(connection_mode=p.GUI if gui else p.DIRECT)
        self.p.resetDebugVisualizerCamera(cameraDistance=1.5,  # 根据需要调整距离
        cameraYaw=0,
        cameraPitch=-60.00,  # 设置俯仰角为 -90度以实现俯视
        cameraTargetPosition=(0, 0, 0))  
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1) #是否渲染
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0) #是否打开控件
        p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, 1) #是否使用核显渲染
        self.p.setGravity(0, 0, -9.81)
        self.p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.init_env()
        
        low_action = np.array([-1.0,-1.0,-1.0,-1.0,-1.0])
        high_action = np.array([1.0,1.0,1.0,1.0,1.0])
        self.action_space = spaces.Box(low=low_action, high=high_action, dtype=np.float32)

        low = np.zeros((1,15),dtype=np.float32)
        high = np.ones((1,15),dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)


    def init_env(self):
        np.random.seed(self.seed)  
        self.fr5 = self.p.loadURDF("fr5_description/urdf/fr5v6.urdf", useFixedBase=True, basePosition=[0, 0, 0],
                                   baseOrientation=p.getQuaternionFromEuler([0, 0, np.pi]), flags=p.URDF_USE_SELF_COLLISION)
        self.table = self.p.loadURDF("table/table.urdf", basePosition=[0, 0.5, -0.63],
                                      baseOrientation=p.getQuaternionFromEuler([0, 0, np.pi / 2]))
        collision_target_id = self.p.createCollisionShape(shapeType=p.GEOM_CYLINDER, radius=0.02, height=0.05)
        self.target = self.p.createMultiBody(baseMass=0, baseCollisionShapeIndex=collision_target_id, basePosition=[0.5, 0.8, 2])
        collision_obstacle_id = self.p.createCollisionShape(shapeType=p.GEOM_SPHERE, radius=0.1)
        self.obstacle1 = self.p.createMultiBody(baseMass=0, baseCollisionShapeIndex=collision_obstacle_id, basePosition=[0.5, 0.5, 2])

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

    def switch_type(self,a,b,x_velocity):
        target_angle=np.arctan(a[0]/a[1])/np.pi*360
        barrier_angle=np.arctan(b[0]/b[1])/np.pi*360
        extreme_deltaangle=15 #最大容错角
        if(target_angle > barrier_angle+extreme_deltaangle or (((target_angle > barrier_angle+10 and x_velocity>=-0.002 ) or (target_angle > barrier_angle+8 and x_velocity>0)) and a[2]>b[2]-0.1)):
            return 2
        elif((target_angle<barrier_angle-15) or(target_angle<barrier_angle-12 and x_velocity<0.01) or (target_angle < barrier_angle-6 and x_velocity<-0.01) or (target_angle < barrier_angle-9 and x_velocity<-0.005) )and target_angle<36:
            return 3
        elif (a[2]-b[2]>0.07 and b[2]<=0.20) or (a[2]-b[2]>0.10 and b[2]<=0.225) or (a[2]-b[2]>0.12 and b[2]<0.26) or a[2]-b[2]>0.15:
            return 4
        else:
            return 1
    
    def reset(self,seed=None):
        self.step_num = 0
        self.total_reward= 0
        self.success_reward = 0
        self.distance_reward = 0
        self.step_penalty = 0
        self.obstacle_penalty=0
        self.terminated = False
        self.done = False
        self.obstacle_contact = False
        neutral_angle = [-49.45849125928217, -57.601209583849, -138.394013961943, -164.0052115563118, -49.45849125928217, 0, 0, 0]
        neutral_angle = [x * math.pi / 180 for x in neutral_angle]
        self.p.setJointMotorControlArray(self.fr5, [1, 2, 3, 4, 5, 6, 8, 9], p.POSITION_CONTROL, targetPositions=neutral_angle)

        self.goalx = np.random.uniform(-0.2, 0.2, 1)
        self.goaly = np.random.uniform(0.8, 0.9, 1)
        self.goalz = np.random.uniform(0.1, 0.3, 1)
        self.target_position = [self.goalx[0], self.goaly[0], self.goalz[0]]
        self.p.resetBasePositionAndOrientation(self.target, self.target_position, [0, 0, 0, 1])

        self.obstacle1_position = [np.random.uniform(-0.2, 0.2, 1) + self.goalx[0], 0.6, np.random.uniform(0.1, 0.3, 1)]
        self.p.resetBasePositionAndOrientation(self.obstacle1, self.obstacle1_position, [0, 0, 0, 1])

        # 设置目标朝x z平面赋予随机速度
        self.random_velocity = np.random.uniform(-0.02, 0.02, 2)
        self.x_velocity = self.random_velocity[0]
        self.z_velocity = self.random_velocity[1]
        self.p.resetBaseVelocity(self.target, linearVelocity=[self.random_velocity[0], 0, self.random_velocity[1]])
        
        for _ in range(100):
            self.p.stepSimulation()

        infos = {}
        infos['is_success'] = False
        infos['reward'] = 0
        infos['step_num'] = 0
        if self.switch_type(self.get_expected_pos(self.get_observation()[0][8:11]),self.get_observation()[0][11:14],self.get_velocity()[0]) in [1,2]:
            self.terminated=True
            return self.get_observation(),{}
        else:
            return self.get_observation(),{}

    def get_observation(self):
        joint_angles = [self.p.getJointState(self.fr5, i)[0] * 180 / np.pi for i in range(1, 6)]
        obs_joint_angles = ((np.array(joint_angles, dtype=np.float32) / 180) + 1) / 2
        target_position = np.array(self.p.getBasePositionAndOrientation(self.target)[0])
        gripper_centre_pos = self.get_pos()
        obs_gripper_centre_pos = np.array([(gripper_centre_pos[0]+0.922)/1.844,
                                    (gripper_centre_pos[1]+0.922)/1.844,
                                    (gripper_centre_pos[2]+0.5)/1],dtype=np.float32)
        dis_to_target=self.get_dis()
        target_position[0]=target_position[0]+0.5
        target_position[1]=(target_position[1]-0.8)*10
        target_position[2]=(target_position[2]-0.1)*2.5
        obstacle1_position = np.array(self.p.getBasePositionAndOrientation(self.obstacle1)[0])
        obstacle1_position[0]=(obstacle1_position[0]+0.4)*1.25
        obstacle1_position[1]=obstacle1_position[1]-0.6
        obstacle1_position[2]=(obstacle1_position[2]-0.1)*5
        self.observation = np.hstack((obs_gripper_centre_pos,obs_joint_angles, target_position, obstacle1_position,dis_to_target)).flatten()
        self.observation = self.observation.astype(np.float32)
        self.observation = self.observation.reshape(1,15)
        return self.observation

    def get_pos(self):  
        gripper_pos = self.p.getLinkState(self.fr5, 6)[0]
        relative_position = np.array([0, 0, 0.15])
        rotation = R.from_quat(self.p.getLinkState(self.fr5, 7)[1])
        rotated_relative_position = rotation.apply(relative_position)
        gripper_centre_pos = np.array(gripper_pos) + rotated_relative_position
        return gripper_centre_pos
          
    def step(self, action,seed=None):
        
        truncated=self.terminated
        if not truncated and not self.done:
            joint_angles = [self.p.getJointState(self.fr5, i)[0] for i in range(1, 7)]
            action = np.clip(action, -1, 1)
            action=np.append(action,0)
            fr5_joint_angles = np.array(joint_angles) + (np.array(action[:6]) / 180 * np.pi)
            gripper = np.array([0, 0])
            angle_now = np.hstack([fr5_joint_angles, gripper])
            self.p.setJointMotorControlArray(self.fr5, [1, 2, 3, 4, 5, 6, 8, 9], p.POSITION_CONTROL, targetPositions=angle_now)

            for _ in range(20):
                self.p.stepSimulation()

            # 检查目标位置并反向速度
            target_position = self.p.getBasePositionAndOrientation(self.target)[0]
            if target_position[0] > 0.5 or target_position[0] < -0.5:
                self.p.resetBaseVelocity(self.target, linearVelocity=[-self.random_velocity[0], 0, self.random_velocity[1]])
                self.x_velocity = -self.x_velocity
            if target_position[2] > 0.5 or target_position[2] < 0.1:
                self.p.resetBaseVelocity(self.target, linearVelocity=[self.random_velocity[0], 0, -self.random_velocity[1]])
                self.z_velocity = -self.z_velocity
            
            self.step_num = self.step_num+1
            
        reward=self.reward()
        self.get_observation()
        
        info = {}
        done=self.done
        if truncated and self.step_num==0:
            reward=0
        self.reset_episode()
        return self.get_observation(),reward,self.done,truncated,info

    def get_dis(self):
        gripper_pos = self.p.getLinkState(self.fr5, 6)[0]
        relative_position = np.array([0, 0, 0.15])
        rotation = R.from_quat(self.p.getLinkState(self.fr5, 7)[1])
        rotated_relative_position = rotation.apply(relative_position)
        gripper_centre_pos = np.array(gripper_pos) + rotated_relative_position
        target_position = np.array(self.p.getBasePositionAndOrientation(self.target)[0])
        return np.linalg.norm(gripper_centre_pos - target_position)
    
    def get_dis_to_obstacle(self):
        obstacle_pos=np.array(self.p.getBasePositionAndOrientation(self.obstacle1)[0])
        return np.linalg.norm(self.get_pos()-obstacle_pos)

    def reward(self):
        self.total_reward=0
        
        # 获取与桌子和障碍物的接触点
        table_contact_points = self.p.getContactPoints(bodyA=self.fr5, bodyB=self.table)
        obstacle1_contact_points = self.p.getContactPoints(bodyA=self.fr5, bodyB=self.obstacle1)

        for contact_point in table_contact_points or obstacle1_contact_points:
            link_index = contact_point[3]
            if link_index not in [0, 1]:
                self.obstacle_contact = True

        if not self.done:
            if self.get_dis() < 0.05 and self.step_num <= 200:
                grasp_reward = 1000
                self.total_reward+=grasp_reward
                self.done = True
            elif self.step_num >= 200:
                grasp_reward = 0
                self.terminated = True
            else:
                grasp_reward = 0
    
        # 接近目标物的奖励（基于差分）
        if self.step_num >1:
            if self.get_dis()>0.4:
                target_reward = 1000*(self.last_distance - self.get_dis())
            elif self.get_dis()>0.2:
                target_reward = 2000*(self.last_distance - self.get_dis())
            else:
                target_reward = 3000*(self.last_distance - self.get_dis())
        else:
            target_reward = 0
        self.last_distance = self.get_dis()
    
        # 避开障碍物的奖励（负奖励）
        dis_to_obstacle=np.linalg.norm(self.get_pos() - self.get_dis_to_obstacle())  #机械臂末端到障碍物的距离
        obstacle_penalty = -100 if self.obstacle_contact else 0
        if dis_to_obstacle < 0.16 and not self.obstacle_contact:
            obstacle_penalty = -20
        if obstacle_penalty<self.obstacle_penalty:
            self.obstacle_penalty=obstacle_penalty
            self.total_reward+=obstacle_penalty      
              
        # 总奖励
        gamma=0.99
        self.total_reward += (gamma**self.step_num)*(target_reward)
        #self.total_reward +=target_reward
        return self.total_reward
    

    def reset_episode(self):
        total_steps = self.step_num
        total_distance = self.get_dis()
        final_score =self.total_reward
        if total_steps >0:
            print(f"steps:", total_steps, "Distance:", total_distance, "Score:", final_score)

    def close(self):
        self.p.disconnect()
