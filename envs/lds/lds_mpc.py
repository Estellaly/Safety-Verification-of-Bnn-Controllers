import imp
from re import L
import casadi as ca
import numpy as np
from envs.ldsenv.ldsenv import getEnv
# from ldsenv import getEnv
# from Plot import plot
import math

class MPC():
    N = 30 # MPC步长大小
    eps=2e-2 #误差控制
    def __init__(self, example):
        self.dt = 0.04
        self.x_dim = example.n_dim  #变量个数 
        self.u_dim = example.u_dim #控制维度
        self.D_zones = example.D_zones#不变区域
        self.I_zones = example.I_zones #初始区域
        self.U_zones = example.U_zones #非安全区域 ,返回布尔值，
        self.f = example.f #微分方程
        self.path = example.path 
        self.u_bound = example.u_bound 
        self.R = example.R
        self.Q = example.Q
        self.constraint_dim=example.constraint_dim
        self.opti = ca.Opti()
        self.U = self.opti.variable(self.N, self.u_dim)  # 每一步控制器的参数
        self.x_0 = self.opti.parameter(self.x_dim).T  #初始状态
        self.__add_subject()

    def get_next_state(self, x, u):
        x1= x[1]+u[0]*0.2
        x0=x[0]+x1*0.3+u[0]*0.05
        
        return ca.hcat([x0,x1])
    def clipx(self,u,low,up):
        if u<=low:
            return low
        elif u >=up:
            return up
        else:
            return u

    def __add_subject(self): #添加约束
        # self.x_final = self.opti.parameter(self.x_dim)

        obj = 0
        x_i=self.x_0
        for i in range(self.N):
            x_i = self.get_next_state(x_i, self.U[i, :])
            if self.D_zones.shape == 'box':#对不变式区域进行约束
                for j in range(self.constraint_dim):
                    self.opti.subject_to(self.opti.bounded(self.D_zones.low[j], x_i[j], self.D_zones.up[j]))
            else:
                center=self.D_zones.center[:self.constraint_dim].reshape(-1,self.constraint_dim)
                self.opti.subject_to(self.opti.bounded(0,ca.sumsqr(x_i[:self.constraint_dim]-center),(self.D_zones.r)**2))


            for j in range(self.constraint_dim):
                self.opti.subject_to(ca.fabs(x_i[0])<=1.0)



            obj = obj +ca.mtimes([x_i,self.Q,x_i.T]) + ca.mtimes(
                [self.U[i, :], self.R, self.U[i, :].T]) #目标函数

            # obj = obj + 1- ca.mtimes([x_i,self.Q,x_i.T])

        for i in range(self.u_dim):
            self.opti.subject_to(self.opti.bounded(-self.u_bound[i], self.U[:, i], self.u_bound[i]))

        opts_setting = {'ipopt.max_iter': 400, 'ipopt.print_level': 0, 'print_time': 0, 'ipopt.acceptable_tol': 1e-8,
                        'ipopt.acceptable_obj_change_tol': 1e-6}

        self.opti.minimize(-1*obj)
        self.opti.solver('ipopt', opts_setting)

    def sample_init(self):
        """在初始区域内随机生成初始点"""
        current_state = np.array([np.random.random() - 0.5 for _ in range(self.x_dim)])  ##边长为1，中心在原点的正方体的内部 【-0.5 0.5】
        if self.I_zones.shape == 'ball':
            ## 在超球内进行采样：将正方体进行归一化，变成对单位球的表面采样，再对其半径进行采样。
            # current_state *= 2  ## 变成半径为1
            # # current_state = current_state / np.sqrt(sum(current_state ** 2)) * np.sqrt(
            # #     self.I_zones.r)  # * np.random.random() ** (1 / self.n_obs)  ##此时球心在原点
            # # ## random()^(1/d) 是为了均匀采样d维球体
            # current_state = current_state*self.I_zones.r # * np.random.random() ** (1 / self.n_obs)  ##此时球心在原点
            # ## random()^(1/d) 是为了均匀采样d维球体
            # current_state += self.I_zones.center
            theta = np.random.random() * 2 * np.pi
            r = np.random.uniform(0, self.I_zones.r ** 2)
            x = math.cos(theta) * (r ** 0.5) + self.I_zones.center[0]
            y = math.sin(theta) * (r ** 0.5) + self.I_zones.center[1]
            current_state =[x,y]
        else:
            current_state = current_state * (self.I_zones.up - self.I_zones.low) + self.I_zones.center
        return current_state
    def sovle(self,current_state=None):
        print("init:",current_state)
        if current_state is None:
            current_state=self.sample_init()
            print("init:",current_state)
        # self.opti.set_value(self.x_final, final_state)
        time_out = 60
        count = 0
        state = [current_state]
        U_mpc=[]
        while (len(state)<=200 and count * self.dt < time_out):
            self.opti.set_value(self.x_0, current_state)  # 初始化初始位置
            sol = self.opti.solve()

            u = sol.value(self.U).reshape(self.N,-1)
            current_state = self.get_next_state(current_state, u[0, :])
            current_state = sol.value(current_state)
            state.append(current_state)
            U_mpc.append(u[0,:])
            count += 1

            if np.abs(current_state[0])>1.0:
                print('crashing')
                break
        # print('error:', np.linalg.norm(current_state - final_state))
        print('step:',count)
        return np.array(state)[:-1],U_mpc


# from Plot import plot
if __name__ == '__main__':
    ex = getEnv()
    mpc = MPC(ex)
    trace,U_mpc = mpc.sovle()
    print(trace)
    print(trace.shape)
    # plot(ex,trace,'env2_mpc.png')
