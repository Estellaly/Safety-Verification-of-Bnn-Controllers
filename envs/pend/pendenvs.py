from pickle import NONE
import numpy as np
import torch
import time
import torch
import os
import casadi as ca

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


class Zones():  ## 定义一个区域 ：有两种，长方体或者球
    def __init__(self, shape, center=None, r=0.0, low=None, up=None, inner=True):
        self.shape = shape
        self.inner = inner
        if shape == 'ball':
            self.center = np.array(center)
            self.r = r  ##半径
        elif shape == 'box':
            self.low = np.array(low)
            self.up = np.array(up)
            self.center = (self.low + self.up) / 2  ## 外接球中心
            self.r = np.sqrt(sum(((self.up - self.low) / 2) ** 2))  ## 外接球半径
        else:
            raise ValueError('没有形状为{}的区域'.format(shape))

def angular(x):
    # x = float(x)
    while x -np.pi / 2<0:
        x += 2 * np.pi
    while x > 5 * np.pi / 2:
        x -= 2 * np.pi
    if x > 3 * np.pi / 2:
        return (x - 2 * np.pi) / (np.pi / 2)
    if x > np.pi / 2:
        return 2 - x / (np.pi / 2)
    return x / (np.pi / 2)

class PendEnv():
    def __init__(self, g=10.0,Q=None,R= NONE,constraint_dim=None):
        self.has_render = True
        self.max_speed = 8
        self.max_torque = 2.5
        self.dt = 0.05
        self.g = g
        self.m = 0.8  # 1.0
        self.l = 1.0
        self.steps = 0


        self.n_dim = 2  # 变量个数
        self.u_dim = 1  # 控制维度
        high = np.array([2.8, 4], dtype=np.float32)
        self.D_zones = Zones('box', low=-high, up=high)  # 不变式区域
        self.D2_zones = Zones('box', low=[-0.9,-2], up=[0.9,2])  # 不变式区域
        init_train = np.array([np.pi / 6, 0.2], np.float32)
        self.I_zones = Zones('box', low=-init_train, up=init_train)
         ## 初始区域
        # self.G_zones = Zones('box', low=[-init_train], up=[])
        self.U_zones = Zones('box', low=[-0.9,-2], up=[0.9,2],inner=False)
        self.isSafe = lambda x: np.abs(x[0]) > 0.9 or  np.abs(x[1]) >2 ## 非安全区域
        self.f = self.step    # 微分方程
        self.u_bound = [1]  # 输出范围为 [-u,u]
        self.path = 'ex1/model'  # 存储路径
        self.id = 5 # 标识符
        self.Q = np.eye(self.n_dim) if Q is None else Q  # state每个维度的权重系数
        self.R =np.eye(self.u_dim) if R is None else R  # U每个维度的权重系数
        self.constraint_dim= constraint_dim if constraint_dim is not None else self.n_dim #需要约束的前constraint_dim个维度，有些方程中某些变量不需要受限
    def step(self,u):
        x = self.state
        self.steps += 1
        th,thdot = x[0],x[1]  # th := theta
        g = self.g
        m = self.m
        l = self.l
        dt = self.dt

        # u = np.clip(u, -1, 1)[0]
        u=np.clip(u[0], -1, 1)
        self.last_u = u  # for rendering
        # costs = angle_normalize(th) ** 2 + 0.1 * thdot ** 2 + 0.001 * (u ** 2)
        # th1 =  ca.sin(th + np.pi)
        th1 =  ca.sin(th)
        newthdot = (
            thdot
            + (
                -3 * g * th1 / (2 * l) 
                + 7.5 * u / (m * l ** 2)  
            )*dt
        )
        newth = th + newthdot * dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)

        self.state = np.array([newth, newthdot])
        # print("state: ", self.state)
        done = self.steps >= 200 or np.abs(newth) > np.pi / 2
        return self.state,done
    def df(self,x,u):
        th,thdot = x[0],x[1]  # th := theta
        g = self.g
        m = self.m
        l = self.l
        dt = self.dt

        # u = np.clip(u, -1, 1)[0]
        u=u[0]
        self.last_u = u  # for rendering
        # costs = angle_normalize(th) ** 2 + 0.1 * thdot ** 2 + 0.001 * (u ** 2)
        # th1 =  ca.sin(th + np.pi)
        th1 =  ca.sin(th)
        ddot = 3 * g * th1 / (2 * l) + 7.5 * u / (m * l ** 2)  


        newthdot = (
            thdot
            + (
                3 * g * th1 / (2 * l) 
                + 7.5 * u / (m * l ** 2)  
            )*dt
        )
        dth=newthdot * dt

        return [dth,ddot]

    def sample_init(self):
        s = np.array([np.random.random() - 0.5 for _ in range(self.n_dim)])  ##边长为1，中心在原点的正方体的内部
        idx=np.random.randint(self.n_dim)
        # is_up=np.random.randint(2)
        # s[idx]=0.5 if is_up else -0.5 #将某一维度变为边界值，使其采样落到表面
        s = s * (self.I_zones.up - self.I_zones.low) + self.I_zones.center
        return s.tolist()
    def sample_unsafe(self,n=500):
        th1 = np.random.uniform(-1, 1, n // 2)
        thdot1 = np.random.choice([-2, 2], n // 2)
        th2 = np.random.choice([0.9, -0.9], n // 2)
        thdot2 = np.random.uniform(-2, 2, n // 2)
        unsafe = np.concatenate([np.stack([th1, thdot1], 1), np.stack([th2, thdot2], 1)], 0)
        return unsafe.tolist()
    def reset(self, state=None):
        if state is None:
            state = self.sample_init()
        self.state = state
        self.steps = 0
        return np.array(self.state)



# def angle_normalize(x):
#     return ((x + np.pi) % (2 * np.pi)) - np.pi


