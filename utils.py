from __future__ import division, print_function

from envs.lds.ldsenv import getEnv as LDSENV
from envs.pend.pendenvs import PendEnv 
from envs.car.car import CarEnv
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from safeWeight import set_model_path,gen_samples
import joblib
import sympy as sym
from envs.Env import get_Env
from sympy import *

def initial(env,first_layer,degree,sample_num):
    if env=='lds':
        example = LDSENV()
    elif env =='pend':
        example = PendEnv()
    elif env=='car':
        example = CarEnv()
    else:
        example = get_Env(int(env))
    if env.isdigit():
        path1 =f'./data/{env}/weights/min_{env}.npz'
        poly_path=f"./data/{env}/{env}_poly_{degree}.pkl"
        data_path=f'./data/{env}/traces_100000.list'
    else:
        if first_layer:
            path1 =f'./data/{env}/weights/{env}_all.npz'
            poly_path=f"./data/{env}/{env}_poly_{degree}_all.pkl"
            data_path=f'./data/{env}/{env}_traces_all_100000.list'
            save_path = f'./data/{env}/weights/safeWeight_{env}_all.npz'
        else:
            path1 =f'./data/{env}/weights/{env}.npz'
            poly_path=f"./data/{env}/{env}_poly_{degree}.pkl"
            data_path=f'./data/{env}/{env}_traces_100000.list'
            save_path = f'./data/{env}/weights/safeWeight_{env}.npz'
    set_model_path(path1)
    poly_reg =PolynomialFeatures(degree=degree) #三次多项式
    lin_reg_2=joblib.load(poly_path)
    x_st ,y_st= joblib.load(data_path)
    x_test ,y_test=np.array(x_st),np.array(y_st)
    y_test = y_test.reshape(-1,1)
    if env=='20':
        x_test ,y_test = x_test[40000:,:] ,y_test[40000:,:]
    else:
        x_test ,y_test = x_test[80000:,:] ,y_test[80000:,:]
    gen_samples(first_layer,sample_num)
    return example,poly_reg,lin_reg_2,x_test,y_test


def savePoly(lin_reg_2,dim,degree):
    x1 = sym.symbols('x0')
    x2 = sym.symbols('x1')
    coef = lin_reg_2.coef_

    inter=  lin_reg_2.intercept_

    if dim ==2:
        if degree==2:
            res= inter + 1*coef[:,0]+ x1*coef[:,1]+ x2*coef[:,2]+ x1*x1*coef[:,3]+ x1*x2*coef[:,4]+ x2*x2*coef[:,5]
        elif degree==3:
            coef = coef.reshape(-1)
            res= inter + 1*coef[0]+ x1*coef[1]+ x2*coef[2]+x1*x1*coef[3]+ x1*x2*coef[4]+x2*x2*coef[5]+x1*x1*x2*coef[6]+x1*x2*x2*coef[7]+x1*x1*x1*coef[8]+x2*x2*x2*coef[9]
    elif dim ==3:
        x3 = sym.symbols('x2')
        if degree==2:
            res= inter + 1*coef[:,0]+ x1*coef[:,1]+ x2*coef[:,2]+x3*coef[:,3]+ x1*x1*coef[:,4]+ x1*x2*coef[:,5] +x1*x3*coef[:,6]+x2*x2*coef[:,7]+x2*x3*coef[:,8]+x3*x3*coef[:,9]
        else:
            coefs=np.array([1,x1,x2,x3,x1**2,x1*x2,x1*x3,x2*x2,x2*x3,x3**2,x1**3,x1**2*x2,x1**2*x3,x1*x2**2,x1*x2*x3,x1*x3**2,x2**3,x2**2*x3,x2*x3**2,x3**3])
            res= inter + np.dot(coefs,coef.T)
    elif dim ==4:
        x3 = sym.symbols('x2')
        x4 = sym.symbols('x3')
        coefs=np.array([1,x1,x2,x3,x4,x1**2,x1*x2,x1*x3,x1*x4,x2*x2,x2*x3,x2*x4,x3**2,x3*x4,x4**2])
        res= inter + np.dot(coefs,coef.T)
    elif dim ==7:
        x=symbols(['x{}'.format(i)for i in range(dim)])
        xs =[1]
        xs.extend(x)
        for i in range(dim):
            for j in range(i,dim):
                xs.append(x[i]*x[j])
        coefs=np.array(xs)
        res= inter + np.dot(coefs,coef.T)
    return str(res[0])


