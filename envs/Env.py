import numpy as np
import time
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

class Zones(): 
    def __init__(self, shape, center=None, r=0.0, low=None, up=None, inner=True):
        self.shape = shape
        self.inner = inner
        if shape == 'ball':
            self.center = np.array(center)
            self.r = r 
        elif shape == 'box':
            self.low = np.array(low)
            self.up = np.array(up)
            self.center = (self.low + self.up) / 2  
            self.r = np.sqrt(sum(((self.up - self.low) / 2) ** 2)) 
        else:
            raise ValueError('error{}'.format(shape))

class Example():
    def __init__(self, n_obs, u_dim, D_zones, I_zones, G_zones, U_zones, f, u, path, dense, units, activation, id, k):
        self.n_obs = n_obs
        self.u_dim = u_dim 
        self.D_zones = D_zones  
        self.I_zones = I_zones  
        self.G_zones = G_zones 
        self.U_zones = U_zones  
        self.f = f 
        self.u = u  
        self.path = path  
        self.dense = dense  
        self.units = units  
        self.activation = activation  
        self.k = k 
        self.id = id  
        self.Q = np.eye(n_obs) 
        self.R = np.eye(u_dim) 
        self.constraint_dim = self.n_obs
class Env():
    def __init__(self, example):
        self.n_obs = example.n_obs
        self.u_dim = example.u_dim
        self.D_zones = example.D_zones
        self.I_zones = example.I_zones
        self.G_zones = example.G_zones
        self.U_zones = example.U_zones
        self.f = example.f
        self.path = example.path
        self.u = example.u

        self.dense = example.dense  
        self.units = example.units
        self.activation = example.activation  
        self.id = example.id
        self.dt = 0.05
        self.k = example.k
        self.Q = np.eye(example.n_obs) 
        self.R = np.eye(example.u_dim)
        self.constraint_dim = self.n_obs
        self.dic = dict()

    def reset(self, s=None):

        if s is not None:
            self.s = np.array(s)
        else:
            self.s = np.array([np.random.random() - 0.5 for _ in range(self.n_obs)])
            if self.I_zones.shape == 'ball':
                
                self.s *= 2  
                self.s = self.s / np.sqrt(sum(self.s ** 2)) * np.sqrt(self.I_zones.r)
                self.s += self.I_zones.center

            else:
                if self.id==10 or self.id==7:
                    idx=np.random.randint(self.n_obs)
                    self.s[idx]=np.random.randint(2)-0.5
                self.s = self.s * (self.I_zones.up - self.I_zones.low) + self.I_zones.center

        return self.s
    def isSafe(self,x):
        x=x.reshape(-1)
        if self.U_zones.shape == 'box':

                Unsafe = sum([self.U_zones.low[i] <= x[i] <= self.U_zones.up[i] for i in range(self.n_obs)])
                if (Unsafe != self.n_obs)^ self.U_zones.inner:

                    return True
        elif self.U_zones.shape == 'ball':
                if self.U_zones.inner==False:
                    Unsafe = sum((x - self.U_zones.center) ** 2) > self.U_zones.r 
                else:
                    Unsafe = sum((x - self.U_zones.center) ** 2) < self.U_zones.r 
                if Unsafe:
                   return True
        return False

    def sample_unsafe(self):
        s = np.array([np.random.random() - 0.5 for _ in range(self.n_obs)]) 
        if self.U_zones.shape == 'ball':
            s *= 2  
            s =s / np.sqrt(sum(s ** 2)) * np.sqrt(self.U_zones.r) 
            s += self.U_zones.center

        else:
            idx=np.random.randint(self.n_obs)
            is_up=np.random.randint(2)
            s[idx]=0.5 if is_up else -0.5 
            s = s * (self.U_zones.up - self.U_zones.low) + self.U_zones.center
        return s
    def sample_init(self):
        s = np.array([np.random.random() - 0.5 for _ in range(self.n_obs)]) 
        if self.I_zones.shape == 'ball':
            s *= 2  
            s =s / np.sqrt(sum(s ** 2)) * np.sqrt(self.I_zones.r)  
            s += self.I_zones.center
        else:
            idx=np.random.randint(self.n_obs)
            is_up=np.random.randint(2)
            s[idx]=0.5 if is_up else -0.5 
            s = s * (self.I_zones.up - self.I_zones.low) + self.I_zones.center
        return s

    def sample_domain(self):
        s = np.array([np.random.random() - 0.5 for _ in range(self.n_obs)])  
        if self.D_zones.shape == 'ball':
            s *= 2 
            s = s / np.sqrt(sum(s ** 2)) * np.sqrt(self.D_zones.r)*np.random.random()^(1/self.n_obs)  
            s += self.D_zones.center

        else:
            s = s * (self.D_zones.up - self.D_zones.low) + self.D_zones.center

        return s
    def sample_goal(self):
        s = np.array([np.random.random() - 0.5 for _ in range(self.n_obs)])
        if self.G_zones.shape == 'ball':

            s *= 2
            s = s / np.sqrt(sum(s ** 2)) * np.sqrt(self.G_zones.r)

            s += self.G_zones.center
        else:
            idx = np.random.randint(self.n_obs)
            is_up = np.random.randint(2)
            s[idx] = 0.5 if is_up else -0.5
            s = s * (self.G_zones.up - self.G_zones.low) + self.G_zones.center

        return s
    def sample_goal_outer(self):
        s = np.array([np.random.random() - 0.5 for _ in range(self.n_obs)])
        if self.G_zones.shape == 'ball':

            s *= 2
            s = s / np.sqrt(sum(s ** 2)) * np.sqrt(self.I_zones.r)
            r_sample=np.random.random()*self.I_zones.r
            if r_sample<self.G_zones.r:
                r_sample+=self.G_zones.r
            s*=r_sample
            s += self.G_zones.center
        else:
            idx = np.random.randint(self.n_obs)
            is_up = np.random.randint(2)
            s[idx] = 0.5 if is_up else -0.5
            s = s * (self.G_zones.up - self.G_zones.low) + self.G_zones.center
        return s
    def step(self, u):
        isSafe=True

        self.ds = np.array([F(self.s, u) for F in self.f])
        last_s = self.s
        self.s = self.s + self.ds * self.dt
        done = False
        if self.D_zones.shape == 'box':
            if self.id == 0:
                done = bool(self.s[0] < self.D_zones.low[0]
                            or self.s[0] > self.D_zones.up[0]
                            or self.s[2] < self.D_zones.low[2]
                            or self.s[2] > self.D_zones.up[2])
            else:
                for i in range(self.n_obs):
                    if self.s[i] < self.D_zones.low[i] or self.s[i] > self.D_zones.up[i]:
                        done = True
            pass

        else:
            t = np.sqrt(self.D_zones.r / sum(self.s ** 2))
            if t < 1:
                self.s = self.s * t
                done = True
        if self.id == 0:
            x, x_dot, theta, theta_dot = self.s
            r1 = (self.D_zones.up[0] - abs(x)) / self.D_zones.up[0] - 0.8
            r2 = (self.D_zones.up[2] - abs(theta)) / self.D_zones.up[2] - 0.5
            reward = r1 + r2

        else:
            if self.G_zones.shape == 'box':
                if self.id==15:
                    reward= -np.sqrt(sum((self.s[:2] - self.G_zones.center[:2]) ** 2)) + np.sqrt(
                    sum((self.G_zones.up[:2] - self.G_zones.low[:2]) ** 2 / 4))
                else:
                    reward = -np.sqrt(sum((self.s - self.G_zones.center) ** 2)) + np.sqrt(
                        sum((self.G_zones.up - self.G_zones.low) ** 2 / 4))

            else:
                if self.id!=5:
                    if self.id==4:
                        reward = -np.sqrt(sum((self.s[:2] - self.G_zones.center[:2]) ** 2)) + np.sqrt(
                            self.G_zones.r)

                    else:
                        reward = -np.sqrt(sum((self.s - self.G_zones.center) ** 2)) + np.sqrt(
                            self.G_zones.r)
                else:
                    reward=sum((last_s-self.G_zones.center)**2)-sum((self.s-self.G_zones.center)**2)
        if sum((self.s-self.G_zones.center)**2)<self.G_zones.r:
            done = True
            isSafe = True
        if self.id == 2:
            reward /= 10

        if self.id>=3:
            if self.U_zones.shape == 'box':

                Unsafe = sum([self.U_zones.low[i] <= self.s[i] <= self.U_zones.up[i] for i in range(self.n_obs)])

                if (Unsafe != self.n_obs)^ self.U_zones.inner:

                    done=True
                    pass
                if self.id==9:
                    gass=np.exp(-sum([(self.s[i]-self.U_zones.center[i])**2/(self.U_zones.up[i]-self.U_zones.low[i])**2 for i in range(self.n_obs)]))/3 #正太分布
                    reward-=gass
                else:
                    if self.id!=7:
                        reward -= np.sqrt(sum((self.s - self.U_zones.center) ** 2)) * 1
           
            elif self.U_zones.shape == 'ball':
                if self.U_zones.inner==False:
                    Unsafe = sum((self.s - self.U_zones.center) ** 2) > self.U_zones.r  # 在圆外
                else:
                    Unsafe = sum((self.s - self.U_zones.center) ** 2) < self.U_zones.r  # 在圆内
                if Unsafe:
                    done=True
                    isSafe=False
                    pass


        return self.s, reward, done, reward
g = 9.8
pi = np.pi
m = 0.1
l = 0.5
mt = 1.1
from numpy import sin, cos, tan
def get_Env(id):
    examples = {
        7:Example(
            n_obs=3,
            u_dim=1,
            D_zones=Zones(shape='box', low=[-5]*3, up=[5]*3),
            I_zones=Zones(shape='ball', center=[-0.75,-1,-0.4],r=0.35**2),
            G_zones=Zones(shape='ball', center=[0, 0, 0], r=0.1**2),
            U_zones=Zones(shape='ball', center=[-0.3,-0.36,0.2],r=0.30**2,inner=True),
            f=[lambda x, u: x[2] + 8 * x[1],
               lambda x, u: -x[1] + x[2],
               lambda x, u: -x[2] - x[0] ** 2 + u[0], 
               ],
            u=3,
            path='ex7/model',
            dense=5,
            units=50 ,
            activation='relu',
            id=7,
            k=50
        ),#Academic 3D
        9: Example(
            n_obs=2,
            u_dim=1,
            D_zones=Zones('box',low=[-2,-2],up=[2,2]),
            I_zones=Zones('box', low=[-0.51,0.49],up=[-0.49,0.51]),
            G_zones=Zones('box', low=[-0.05,-0.05],up=[0.05,0.05]),
            U_zones=Zones('box', low=[-0.4,0.2],up=[0.1,0.35]),
            f=[lambda x, u: x[1],
               lambda x, u: (1-x[0]**2)*x[1]-x[0]+u[0]
              ],
            u=3,
            path='ex9/model',
            dense=5,
            units=64,
            activation='relu',
            id=9,
            k=50
        ),#Oscillator
        10:Example(
            n_obs=2,
            u_dim=1,
            D_zones=Zones('box',low=[-6,-7*pi/10],up=[6,7*pi/10]),
            I_zones=Zones('box', low=[-1,-pi/16],up=[1,pi/16]),
            G_zones=Zones('ball', center=[0,0],r=0.1**2),
            U_zones=Zones('box', low=[-5,-pi/2],up=[5,pi/2],inner=False),
            f=[lambda x, u:sin(x[1]),
               lambda x, u: -u[0]
              ],
            u=3,
            path='ex10/model',
            dense=5,
            units=30,
            activation='relu',
            id=10,
            k=50
        ),#Dubins' Car
       12:Example(
           n_obs=3,
           u_dim=1,
           D_zones=Zones('box', low=[-2.2]*3, up=[2.2]*3),
           I_zones=Zones('box', low=[-0.2]*3, up=[0.2]*3),
           G_zones=Zones('box', low=[-0.1]*3, up=[0.1]*3),
           U_zones=Zones('box', low=[-2.2]*3, up=[-2]*3),
           f=[lambda x, u: x[1],
              lambda x, u: 30*sin(x[0])+300*cos(x[0])*tan(x[2])+15*cos(x[0])/cos(x[2])**2*u[0],
              lambda x, u: u[0],
              ],
           u=3,
           path='ex12/model',
           dense=5,
           units=40,
           activation='relu',
           id=12,
           k=50
       ),#Bicycle Steering
    13:Example(
        n_obs=7,
        u_dim=1,
        D_zones=Zones('box', low=np.array([1.2,1.05,1.5,2.4,1,0.1,0.45])-5, up=np.array([1.2,1.05,1.5,2.4,1,0.1,0.45])+5),
        I_zones=Zones('box', low=np.array([1.2,1.05,1.5,2.4,1,0.1,0.45])-0.05, up=np.array([1.2,1.05,1.5,2.4,1,0.1,0.45])+0.05),
        G_zones=Zones('box', low=np.array([0.87,0.37,0.56, 2.75, 0.22, 0.08,0.27])-0.1, up=np.array([0.87,0.37,0.56, 2.75, 0.22, 0.08,0.27])+0.1),
        U_zones=Zones('box', low=np.array([1.2,1.05,1.5,2.4,1,0.1,0.45])-4.5, up=np.array([1.2,1.05,1.5,2.4,1,0.1,0.45])+4.5,inner=False),
        f=[lambda x, u: 1.4*x[2]-0.9*x[0],
           lambda x, u: 2.5*x[4]-1.5*x[1]+u[0],
           lambda x, u: 0.6*x[6]-0.8*x[1]*x[2],
           lambda x,u:2-1.3*x[2]*x[3],
           lambda x,u:0.7*x[0]-x[3]*x[4],
           lambda x,u:0.3*x[0]-3.1*x[5],
           lambda x,u:1.8*x[5]-1.5*x[1]*x[6],
           ],
        u=0.3,
        path='ex13/model',
        dense=5,
        units=50,
        activation='relu',
        id=13,
        k=50
    ),#LALO20
    20: Example( #example 8 
        n_obs=4,
        u_dim=1,
        D_zones=Zones('ball', center=[0, 0, 0, 0], r=16),
        I_zones=Zones('box', low=[-0.2, -0.2, -0.2, -0.2], up=[0.2, 0.2, 0.2, 0.2]),
        U_zones=Zones('ball', center=[-2, -2, -2, -2], r=1),
        G_zones=None,
        f=[lambda x, u: -x[0] - x[3] + u[0],
        lambda x, u: x[0] - x[1] + x[0] ** 2 + u[0],
        lambda x, u: -x[2] + x[3] + x[1] ** 2,
        lambda x, u: x[0] - x[1] - x[3] + x[2] ** 3 - x[3] ** 3],
        u=1,
        path='ex20/model',
        dense=5,
        units=30,
        activation='relu',
        id=20,
        k=100
        )
    }

    return Env(examples[id])
