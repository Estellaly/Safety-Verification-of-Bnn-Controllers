import imp
from re import L
import casadi as ca
import numpy as np
from  car import CarEnv
# from envs.carEnv.car import CarEnv
import matplotlib.pyplot as plt
import os
# from ldsenv import getEnv
# from Plot import plot
import math

class MPC():
    N = 10 # MPC步长大小
    eps=2e-3 #误差控制
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
        x0=x[0]+u[0]*1
        x2=x[2]-0.1
        # x0,x1,x2 = self.f(x,u)
        return ca.hcat([x0,x[1],x2])


    def __add_subject(self): #添加约束
        # self.x_final = self.opti.parameter(self.x_dim)

        obj = 0
        x_i=self.x_0
        # print(x_i.shape)
        for i in range(self.N):
            x_i = self.get_next_state(x_i, self.U[i, :])
            if self.D_zones.shape == 'box':#对不变式区域进行约束
                for j in range(self.constraint_dim):
                    # print(j)
                    self.opti.subject_to(self.opti.bounded(self.D_zones.low[j], x_i[j], self.D_zones.up[j]))
            else:
                center=self.D_zones.center[:self.constraint_dim].reshape(-1,self.constraint_dim)
                self.opti.subject_to(self.opti.bounded(0,ca.sumsqr(x_i[:self.constraint_dim]-center),(self.D_zones.r)**2))

            self.opti.subject_to(ca.fabs(x_i[0]-x_i[1])>=1)
            # self.opti.subject_to((x_i[2]>=0))

            obj = obj -0.5*ca.fabs(x_i[0]-x_i[1])+0.1*ca.mtimes([x_i,self.Q,x_i.T]) + 0.1*ca.mtimes(
                [self.U[i, :], self.R, self.U[i, :].T]) #目标函数
            # obj+=0.1*ca.fabs(x_i[0]-x_i[1])
            # obj = obj -0.1*ca.fabs(x_i[0]-x_i[1])+ca.mtimes(
            #     [self.U[i, :], self.R, self.U[i, :].T]) #目标函数
            # obj = obj + 1- ca.mtimes([x_i,self.Q,x_i.T])

        for i in range(self.u_dim):
            self.opti.subject_to(self.opti.bounded(-self.u_bound[i], self.U[:, i], self.u_bound[i]))

        opts_setting = {'ipopt.max_iter': 400, 'ipopt.print_level': 0, 'print_time': 0, 'ipopt.acceptable_tol': 1e-8,
                        'ipopt.acceptable_obj_change_tol': 1e-6}

        self.opti.minimize(obj)
        self.opti.solver('ipopt', opts_setting)

    def sample_init(self):
        """在初始区域内随机生成初始点"""
        px=np.random.random()*(self.I_zones.up[0]-self.I_zones.low[0])+self.I_zones.low[0]
        ax=np.random.random()*(self.I_zones.up[1]-self.I_zones.low[1])+self.I_zones.low[1]
        while np.abs(px-ax)<=1:
            ax=np.random.random()*(self.I_zones.up[1]-self.I_zones.low[1])+self.I_zones.low[1]
        ay=5
        return np.array([px,ax,ay])
    def sovle(self,current_state=None):
        # print("init:",current_state)
        if current_state is None:
            current_state=self.sample_init()
            # print("init:",current_state)
        # self.opti.set_value(self.x_final, final_state)
        time_out = 60
        count = 0
        state = [current_state]
        U_mpc=[]
        while (len(state)<=50 and count * self.dt < time_out):
            self.opti.set_value(self.x_0, current_state)  # 初始化初始位置
            sol = self.opti.solve()
            u = sol.value(self.U).reshape(self.N,-1)
            current_state = self.get_next_state(current_state, u[0, :])
            current_state = sol.value(current_state)
            state.append(current_state)
            U_mpc.append(u[0,:])
            count += 1
            # print("state:",current_state)
            # print("u:",u[0,:])
            # print('---')
            # if np.abs(np.abs(current_state[0]-current_state[1])-1)<1e-6 and current_state[2]<=0:
            if current_state[2]<=0:
                # print('crashing')
                break
        # print('error:', np.linalg.norm(current_state - final_state))
        # print('step:',count)
        return np.array(state)[:-1],U_mpc


# from Plot import plot
if __name__ == '__main__':
    ex = CarEnv()
    mpc = MPC(ex)
    trace,U_mpc = mpc.sovle()
    # print(trace)
    # print(trace.shape)
    import joblib
    import numpy as np
    from car import CarEnv
    ex = CarEnv()
    mpc = MPC(ex)
    path=f'./new'
    def generate_data(path):
        N=int(2e4)
        X=[]
        Y=[]
        # if not os.path.exists(path):
        #     os.mkdir(path)
        path=path+'traces_{}.list'.format(N)
        if os.path.exists(path):
            # X,Y=joblib.load(path)
            pass
        else:
            while len(X)<N:
            # for i in range(10):
                x,y = mpc.sovle()
                # ax.plot(x[:,0], x[:,1], x[:,2], color="tab:blue")
                X.extend(list(x))
                Y.extend(list(y))
                print(id,":",len(X))
            joblib.dump([X,Y],path)
        # plt.savefig('./Image/car.png')

        return np.array(X),np.array(Y)
# for i in range(1):
#     x,U_mpc = mpc.sovle()
#     ax.plot(x[:,0], x[:,1], x[:,1], color="tab:blue")

    generate_data(path)
