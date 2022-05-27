import numpy as np
import time
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


class LdsEnv():
    def __init__(self,D_zones,I_zones,Q=None, R=None,constraint_dim=None):
        self.steps = None
        self.state = None
        self.n_dim = 2  # 变量个数
        self.u_dim = 1  # 控制维度
        self.D_zones = D_zones  # 不变式区域
        self.I_zones = I_zones  ## 初始区域
        self.U_zones=Zones('box', low=[-1.2, -1.2], up=[1.2, 1.2],inner=False)
        self.isSafe = lambda x: np.abs(x[0]) >=1.2 or  np.abs(x[1]) >=1.2 ## 非安全区域
        self.f = self.step   # 微分方程
        self.u_bound = [1]  # 输出范围为 [-u,u]
        self.path = 'ex1/model'  # 存储路径
        self.id = 4 # 标识符
        self.Q = np.eye(self.n_dim) if Q is None else Q  # state每个维度的权重系数
        self.R = np.eye(self.u_dim) if R is None else R  # U每个维度的权重系数
        self.constraint_dim= constraint_dim if constraint_dim is not None else self.n_dim #需要约束的前constraint_dim个维度，有些方程中某些变量不需要受限
    def step(self,u):
        self.steps += 1
        u = self.clipx(u[0],-1,1)
        x = self.state
        # u=u[0]
        x[0]+=x[1]*0.3+u*0.05
        x[1]+=u*0.2
        self.state = np.array([x[0], x[1]])
        crash = np.abs(x[0]) > 1.2 or np.abs(x[1]) > 1.2
        done = self.steps >= 200 or crash
        return self.state,done

    def df(self,x,u):
        d0=x[1]*0.3+self.clipx(u[0],-1,1)*0.05
        d1=self.clipx(u[0],-1,1)*0.2
        return [d0,d1]
    def clipx(self,u,low,up):
        if u<=low:
            return low
        elif u >=up:
            return up
        else:return u
    def sample_unsafe(self,n):
        y1 = np.random.uniform(-1.2, 1.2, n // 2)
        x1 = np.random.choice([-1.2, 1.2], n // 2)
        x2 = np.random.uniform(-1.2, 1.2, n // 2)
        y2 = np.random.choice([-1.2, 1.2], n // 2)
        unsafe = np.concatenate([np.stack([x1, y1], 1), np.stack([x2, y2], 1)], 0)
        return unsafe.tolist()
    def sample_init(self):
        s = np.array([np.random.random() - 0.5 for _ in range(self.n_dim)])  ##边长为1，中心在原点的正方体的内部
        idx=np.random.randint(self.n_dim)
        is_up=np.random.randint(2)
        s[idx]=0.5 if is_up else -0.5 #将某一维度变为边界值，使其采样落到表面
        s = s * (self.I_zones.up - self.I_zones.low) + self.I_zones.center
        return s
    def reset(self, state=None):
        if state is None:
            state = self.sample_init()
        self.state = state
        self.steps = 0
        return self.state

def getEnv():
    return LdsEnv(D_zones=Zones('box', low=[-1.2, -1.2], up=[1.2, 1.2]),
    I_zones = Zones('box', low=[-0.4, -0.4], up=[0.4, 0.4]))




