from __future__ import division, print_function

from sympy import *
from envs.Env import get_Env
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from safeWeight import compute_p_one,compute_p,set_model_path,gen_samples,get_L,get_AL,interval_bound_propagation_2layer_L
import joblib
import sympy as sym
import argparse
from multiprocessing.dummy import Pool as ThreadPool
import time
import json

parser = argparse.ArgumentParser()
parser.add_argument("--id", default=9, type=int)
parser.add_argument("--xmargin", default=0.001,type = float)
parser.add_argument("--wmargin", default=0.5,type = float)
parser.add_argument("--u", default=0.1,type = float)
parser.add_argument("--umargin", default=0.1,type = float)
parser.add_argument("--degree", default=2,type = int)
parser.add_argument("--num", default=1000,type = int)
args = parser.parse_args()
degree = args.degree
id = int(args.id)
path1 =f'./data/{id}/weights/min_{id}.npz'
poly_path=f"./data/{id}/{id}_poly_{degree}.pkl"
data_path=f'./data/{id}/traces_100000.list'
example = get_Env(id)
savePath= f"./data/{id}/safeWeight_{id}.npz"


poly_reg =PolynomialFeatures(degree=degree) 
lin_reg_2=joblib.load(poly_path)
def poly_predict(x):
  x1 =poly_reg.fit_transform(x)
  y1 =  lin_reg_2.predict(x1) 
  return y1

def savePoly():
    x1 = sym.symbols('x0')
    x2 = sym.symbols('x1')
    coef = lin_reg_2.coef_
    inter=  lin_reg_2.intercept_
    if example.n_obs ==2:
        res= inter + 1*coef[:,0]+ x1*coef[:,1]+ x2*coef[:,2]+ x1*x1*coef[:,3]+ x1*x2*coef[:,4]+ x2*x2*coef[:,5]
    elif example.n_obs ==3:
        x3 = sym.symbols('x2')
        if degree==2:
            res= inter + 1*coef[:,0]+ x1*coef[:,1]+ x2*coef[:,2]+x3*coef[:,3]+ x1*x1*coef[:,4]+ x1*x2*coef[:,5] +x1*x3*coef[:,6]+x2*x2*coef[:,7]+x2*x3*coef[:,8]+x3*x3*coef[:,9]
        else:
            coefs=np.array([1,x1,x2,x3,x1**2,x1*x2,x1*x3,x2*x2,x2*x3,x3**2,x1**3,x1**2*x2,x1**2*x3,x1*x2**2,x1*x2*x3,x1*x3**2,x2**3,x2**2*x3,x2*x3**2,x3**3])
            res= inter + np.dot(coefs,coef.T)
    elif example.n_obs ==4:
        x3 = sym.symbols('x2')
        x4 = sym.symbols('x3')
        coefs=np.array([1,x1,x2,x3,x4,x1**2,x1*x2,x1*x3,x1*x4,x2*x2,x2*x3,x2*x4,x3**2,x3*x4,x4**2])
        res= inter + np.dot(coefs,coef.T)
    elif example.n_obs ==7:
        x=symbols(['x{}'.format(i)for i in range(example.n_obs)])
        xs =[1]
        xs.extend(x)
        for i in range(example.n_obs):
            for j in range(i,example.n_obs):
                xs.append(x[i]*x[j])
        coefs=np.array(xs)
        res= inter + np.dot(coefs,coef.T)
    print(res)

    return str(res[0])

poly_s = savePoly()
x_st ,y_st= joblib.load(data_path)
x_test ,y_test=np.array(x_st),np.array(y_st)
y_test = y_test.reshape(-1,1)
x_test ,y_test=x_test[80000:,:] ,y_test[80000:,:]
x_margin=args.xmargin

length = x_test.shape[0]


sample_number = args.num
set_model_path(path1)
gen_samples(1,sample_number)
u=args.u
validId=[]
print("x_margin:",x_margin)
totalP=0
w_margin=args.wmargin
finals=[]
items = list(range(sample_number))
def isValid(index):
    isSafe = True
    if index%100==0:
        print(index)
    for i in range(length//20):
        x_init=(x_test[i,:]).reshape(1,-1)
        samplePredict = poly_predict(x_init)
        
        isPass,diff,l_poly,l,final = get_AL(x_init.reshape(1,example.n_obs),x_margin,samplePredict.reshape(1),u,w_margin,index,poly_s)
        i+=20
        finals.append(final)
        if not isPass:
            isSafe = False
            break

    if isSafe:
        validId.append(index)

start = time.time()
print(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))
pool = ThreadPool(20)
pool.map(isValid,items)
pool.close()
pool.join()
end = time.time()

print(str(id)," u:",u,',have ',len(validId),'valid weight') 
u+=args.umargin
if len(validId)!=0:
    totalP,maxSigma= compute_p(validId,w_margin,savePath)
    print(str(id),"finalP:",totalP,"u:",u, "x_margin:",x_margin,"w_margin:",w_margin,"sigma:",maxSigma,"time:",end-start)
    result_dict = {
        "id": id,
        "runtimes": end-start,
        "finalP": totalP,
        "u": u,
        "x_margin": x_margin,
        "w_margin":w_margin,
        "sigma":maxSigma
    }
    # filename = f"./data/{id}/{id}.json"
    # with open(filename, "a+",newline='\n') as f:
    #     json.dump(result_dict, f)

