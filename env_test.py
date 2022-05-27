from __future__ import division, print_function

from envs.lds.ldsenv import getEnv as LDSENV
from envs.pend.pendenvs import PendEnv 
from envs.car.car import CarEnv
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from safeWeight import interval_all_L,compute_p,set_model_path,compute_p_one,gen_samples,interval_2layer_L
import joblib
import sympy as sym
from sympy import *
import argparse
from multiprocessing.dummy import Pool as ThreadPool
import time
import json

parser = argparse.ArgumentParser()
parser.add_argument("--env", default='car', type=str)
parser.add_argument("--isfirst", default=0,type=int)
parser.add_argument("--xmargin", default=0.0005,type = float)
parser.add_argument("--wmargin", default=0.5,type = float)
parser.add_argument("--u", default=0.1,type = float)
parser.add_argument("--umargin", default=0.1,type = float)
parser.add_argument("--degree", default=2,type = int)
parser.add_argument("--num", default=1000,type = int)

args = parser.parse_args()
degree = args.degree
first_layer=args.isfirst
env = args.env
if env=='lds':
    example = LDSENV()
elif env =='pend':
    example = PendEnv()
else:
    example = CarEnv()

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
    if example.n_dim ==2:
        if degree==2:
            res= inter + 1*coef[:,0]+ x1*coef[:,1]+ x2*coef[:,2]+ x1*x1*coef[:,3]+ x1*x2*coef[:,4]+ x2*x2*coef[:,5]
        elif degree==3:
            coef = coef.reshape(-1)
            res= inter + 1*coef[0]+ x1*coef[1]+ x2*coef[2]+x1*x1*coef[3]+ x1*x2*coef[4]+x2*x2*coef[5]+x1*x1*x2*coef[6]+x1*x2*x2*coef[7]+x1*x1*x1*coef[8]+x2*x2*x2*coef[9]
    elif example.n_dim ==3:
        x3 = sym.symbols('x2')
        if degree==2:
            res= inter + 1*coef[:,0]+ x1*coef[:,1]+ x2*coef[:,2]+x3*coef[:,3]+ x1*x1*coef[:,4]+ x1*x2*coef[:,5] +x1*x3*coef[:,6]+x2*x2*coef[:,7]+x2*x3*coef[:,8]+x3*x3*coef[:,9]
        else:
            coefs=np.array([1,x1,x2,x3,x1**2,x1*x2,x1*x3,x2*x2,x2*x3,x3**2,x1**3,x1**2*x2,x1**2*x3,x1*x2**2,x1*x2*x3,x1*x3**2,x2**3,x2**2*x3,x2*x3**2,x3**3])
            res= inter + np.dot(coefs,coef.T)
    r=res[0].xreplace({n : round(n, 4) for n in res[0].atoms(Number)})
    print(r)
    return str(res[0])

poly_s = savePoly()
x_st ,y_st= joblib.load(data_path)
x_test ,y_test=np.array(x_st),np.array(y_st)
y_test = y_test.reshape(-1,1)
x_test ,y_test = x_test[80000:,:] ,y_test[80000:,:]
x_margin=args.xmargin

length = x_test.shape[0]
w_margin=args.wmargin
sample_number =args.num
set_model_path(path1)
gen_samples(first_layer,sample_number)
u=args.u
print("x_margin:",x_margin,"env:",env)
totalP=0
w_margin=args.wmargin
validId=[]
finals = []
totalP = 0
def isValid(index): 
    if index%100==0:
        print(index)
    isSafe = True
    i=0
    # for j in range(length//20):
    for i in[0]:
        x_init=(x_test[i,:]).reshape(1,-1)
        if env!='car' and (example.isSafe(x_init.reshape(2,-1)) or np.abs(y_test[i,0])>1):
            i+=1
            continue
        samplePredict = poly_predict(x_init)
        if not first_layer:
            isPass,final = interval_2layer_L(x_init.reshape(1,example.n_dim),x_margin,samplePredict.reshape(1),u,w_margin,index,poly_s)
        else:
            isPass,final = interval_all_L(x_init.reshape(1,example.n_dim),x_margin,samplePredict.reshape(1),u,w_margin,index,poly_s)
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
items = list(range(sample_number))
pool.map(isValid,items)
pool.close()
pool.join()
end = time.time()

print(env," u:",u,',have ',len(validId),' valid weight') 
if len(validId)!=0:
    if first_layer:
        totalP,maxSigma= compute_p(validId,w_margin,save_path)
    else:
        totalP,maxSigma= compute_p_one(validId,w_margin,env,save_path)
    if first_layer:
        print(env,"all layer")
    else:
        print(env,"one layer")
    print(env,"finalP:","isfirst_layer:",first_layer,totalP,"u:",u, "x_margin:",x_margin,"w_margin:",w_margin,"runtime:",end-start,"margin:",maxSigma)
    result_dict = {
        "id": env,
        "isfirst_layer:":first_layer,
        "runtimes": end-start,
        "finalP": totalP,
        "u": u,
        "x_margin": x_margin,
        "w_margin":w_margin,
        "sigma":maxSigma
    }
    # filename = f"./envs/{env}env/{env}.json"
    # with open(filename, "a+",newline='\n') as f:
    #     json.dump(result_dict, f)

