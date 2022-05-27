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


class CarEnv():
    def __init__(self,Q=None, R=None,constraint_dim=None):
        self.safety_ball = 1
        self.stepSize =None
        self.s =None
        self.speed_y = 1
        self.speed_x = 1
        self.n_obs=3
        self.n_dim = 3  # 变量个数
        self.u_dim = 1  # 控制维度
        self.D_zones = Zones('box', low=[-2,-2,-1], up=[2,2,5])  # 不变式区域
        self.I_zones = Zones('box', low=[-2,-2,5], up=[2,2,5])  ## 初始区域
        self.U_zones = lambda x: np.abs(x[0]-x[1])<=1.0 ## 非安全区域
        self.f1=[lambda x, u: u[0],
               lambda x, u: 0,
               lambda x,u: -1
               ]
        self.f = self.step   # 微分方程
        self.u_bound = [1]  # 输出范围为 [-u,u] ?怎么设置
        self.path = 'ex1/model'  # 存储路径
        self.id = 6 # 标识符
        self.Q = np.eye(self.n_dim) if Q is None else Q  # state每个维度的权重系数
        self.R = np.eye(self.u_dim) if R is None else R  # U每个维度的权重系数
        self.constraint_dim= constraint_dim if constraint_dim is not None else self.n_dim #需要约束的前constraint_dim个维度，有些方程中某些变量不需要受限
    def step(self,u):
        x = self.s
        x0=x[0]+u[0]*self.speed_x
        x2=x[2]-self.speed_y
        self.s = np.array([x0,x[1],x2])
        self.stepSize+=1
        done = False
        if self.stepSize>200: done= True
        if abs(x0-x[1])<1 or x2<=0: done = True
        return self.s,0,done,0
    def df(self,x,u):
        x0=u[0]*self.speed_x
        x2=-self.speed_y
        return [x0,0,x2]
    def clipx(self,u,low,up):
        if u<=low:
            return low
        elif u >=up:
            return up
        else:return u
    def sample_unsafe(self):
        px = np.random.random()*(self.I_zones.up[0]-self.I_zones.low[0])+self.I_zones.low[0]
        ax = np.random.random()*(self.I_zones.up[1]-self.I_zones.low[1])+self.I_zones.low[1]
        ay = 0
        return [px,ax,ay]
    def sample_unsafe2(self):
        px = np.random.random()*(self.I_zones.up[0]-self.I_zones.low[0])+self.I_zones.low[0]
        ax = np.random.random()*(self.I_zones.up[1]-self.I_zones.low[1])+self.I_zones.low[1]
        # ay = np.random.randint(0, self.I_zones.up[2] + 1)
        ay=0
        while (abs(ax-px)>1):
            ax = np.random.random()*(self.I_zones.up[1]-self.I_zones.low[1])+self.I_zones.low[1]
        return [px,ax,ay]
    def sample_init(self,nums):
        zones = self.I_zones
        lows = np.linspace(zones.low[0],zones.up[0],nums) 
        ups = np.linspace(zones.low[1],zones.up[1],nums)
        points = np.transpose([np.repeat(lows, len(ups)), np.tile(ups, len(lows))])
        ay = [5]*points.shape[0]
        ay = np.array(ay).reshape(-1,1)
        res= np.hstack((points,ay))
        return res.tolist()
    def reset(self,s=None):
        if s is not None:
            self.s = np.array(s)
        else:
            self.s = np.array([np.random.random() - 0.5 for _ in range(self.n_obs)])  ##边长为1，中心在原点的正方体的内部
            self.s = self.s * (self.I_zones.up - self.I_zones.low) + self.I_zones.center
            self.s[2]=5
        self.stepSize=0

        return self.s



        







