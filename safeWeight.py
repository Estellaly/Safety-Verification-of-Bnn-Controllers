
from os import pread
import numpy as np
from tqdm import trange
import edward as ed
import tensorflow as tf
import scipy.optimize as opt
import numpy as np
import sympy
from bound.ibp_p import ibp_poly
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
width = 0
def set_width(w):
    global width
    width = w

def set_BNN(bnn_n):
   global bnn_net 
   bnn_net = bnn_n

model_path = "ERR - NO MODEL SET. Call set_model_path function."
golbal_interval=[]
def set_model_path(m):
    global model_path
    model_path = m
def compute_p_one(id,w_margin,env,savepath):
    [sW_1, sb_1]=GLOBAL_samples
    vW_1, vb_1 = [], []
    res=np.load(model_path)
    mW_0, mb_0, mW_1, mb_1, dW_1, db_1=res['mW_0'],res['mb_0'],res['mW_1'],res['mb_1'],res['dW_1'],res['db_1']
    for item in id:
        vW_1.append(sW_1[item])
        vb_1.append(sb_1[item])
    pW_1,maxs1,safeweight1 = compute_interval_probs_weight(np.asarray(vW_1), marg=w_margin, mean=mW_1, std=dW_1)
    pb_1 ,maxs2,safeweight2 = compute_interval_probs_bias(np.asarray(vb_1), marg=w_margin, mean=mb_1, std=db_1)
    p = 0.0
    maxss=np.array([maxs1,maxs2])
    for i in pW_1.flatten():
        p+=math.log(i)
    for i in pb_1.flatten():
        p+=math.log(i)
    #print( math.exp(p))
    # np.savez_compressed(savepath,safeW1=safeweight1,safeB1=safeweight2)
    return math.exp(p),np.min(maxss)


def compute_p(id,w_margin,savepath):
    [sW_0, sb_0, sW_1, sb_1]=GLOBAL_samples
    vW_0, vb_0, vW_1, vb_1 = [], [], [], []
    res=np.load(model_path)
    mW_0, mb_0, mW_1, mb_1, dW_0, db_0, dW_1, db_1=res['mW_0'],res['mb_0'],res['mW_1'],res['mb_1'],res['dW_0'],res['db_0'],res['dW_1'],res['db_1']
    for item in id:
        vW_0.append(sW_0[item])
        vb_0.append(sb_0[item])
        vW_1.append(sW_1[item])
        vb_1.append(sb_1[item])
    pW_0,maxs1,safeweight1 = compute_interval_probs_weight(np.asarray(vW_0), marg=w_margin, mean=mW_0, std=dW_0)
    pb_0 ,maxs2,safeweight2 = compute_interval_probs_bias(np.asarray(vb_0), marg=w_margin, mean=mb_0, std=db_0)
    pW_1 ,maxs3,safeweight3 = compute_interval_probs_weight(np.asarray(vW_1), marg=w_margin, mean=mW_1, std=dW_1)
    pb_1 ,maxs4,safeweight4= compute_interval_probs_bias(np.asarray(vb_1), marg=w_margin, mean=mb_1, std=db_1)
    maxss=np.array([maxs1,maxs2,maxs3,maxs4])
    p = 0.0
    for i in pW_0.flatten():
        p+=math.log(i)
    for i in pb_0.flatten():
        p+=math.log(i)
    for i in pW_1.flatten():
        p+=math.log(i)
    for i in pb_1.flatten():
        p+=math.log(i)
    #print( math.exp(p))
    # np.savez_compressed(savepath,safeW0=safeweight1,safeB0=safeweight2,safeW1=safeweight3,safeB1=safeweight4)
    return math.exp(p),np.min(maxss)

GLOBAL_samples = "lollol"
def gen_samples(first_layer,num):
    global GLOBAL_samples
    res=np.load(model_path)
    if not first_layer:
        mW_0, mb_0, mW_1, mb_1,dW_1, db_1=res['mW_0'],res['mb_0'],res['mW_1'],res['mb_1'],res['dW_1'],res['db_1']
        sW_1 = np.random.normal(mW_1, dW_1, (num, mW_1.shape[0], mW_1.shape[1]))
        sb_1 = np.random.normal(mb_1, db_1, (num, mb_1.shape[0]))
        GLOBAL_samples = [sW_1, sb_1]
    else:
        mW_0, mb_0, mW_1, mb_1, dW_0, db_0, dW_1, db_1=res['mW_0'],res['mb_0'],res['mW_1'],res['mb_1'],res['dW_0'],res['db_0'],res['dW_1'],res['db_1']
        sW_0 = np.random.normal(mW_0, dW_0, (num, mW_0.shape[0], mW_0.shape[1]))
        sb_0 = np.random.normal(mb_0, db_0, (num, mb_0.shape[0]))
        sW_1 = np.random.normal(mW_1, dW_1, (num, mW_1.shape[0], mW_1.shape[1]))
        sb_1 = np.random.normal(mb_1, db_1, (num, mb_1.shape[0]))
        GLOBAL_samples = [sW_0, sb_0, sW_1, sb_1]
def relu(x):
    return (x > 0) * x


def get_R(A,x,b,eps):#A:(Alow,Aup),x,b:(blow,bup)
    Alow,Aup=A
    xlow,xup=x-eps,x+eps
    blow,bup=b
    Ax_range=np.hstack([Alow@xlow,Alow@xup,Aup@xlow,Aup@xup])
    Axlow=np.min(Ax_range,axis=1)+blow
    Axup=np.max(Ax_range,axis=1)+bup

    return np.diag(np.float32(Axup>0))
def get_RAD(R,A,D):
    Alow, Aup = A
    #print(R.shape,Alow.shape,D.shape)
    res=np.array([R@Alow@D,R@Aup@D])
    res=np.max(np.abs(res),axis=0)
    return np.sqrt(sum(sum(res*res)))
def get_L_all(data,x,eps):
    [lW_0,uW_0,lB_0,uB_0,lW_1,uW_1] = data
    L = 1
    A = (lW_0.T, uW_0.T)
    b = (lB_0,uB_0)
    x=x.T

    R = get_R(A, x, b, eps)
    D = np.eye(A[0].shape[1])
    L *= get_RAD(R, A, D)


    A = (lW_1.T, uW_1.T)
    # b=(data['lB_1'],data['uB_1'])
    res = np.max(np.abs(np.array([A[0], A[1]])), axis=0)
    L *= np.sqrt(sum(sum(res * res)))
    return L
def get_L_one(data):
    [lW_0,uW_0] = data
    L = 1
    A = (lW_0.T, uW_0.T)
    # b=(data['lB_1'],data['uB_1'])
    res = np.max(np.abs(np.array([A[0], A[1]])), axis=0)
    L *= np.sqrt(sum(sum(res * res)))
    return L
def Poly_L(y, x0, epsilon):
    # x0=list(x0)
    x0=x0.reshape(-1)
    n=x0.shape[0]
    x=sympy.symbols(['x{}'.format(i)for i in range(n)])
    y = sympy.sympify(y)
    op=0
    for i in range(n):
        op+=(sympy.diff(y,x[i]))**2
    f=sympy.lambdify(x,op)
    bounds = [(x0[i]-epsilon,x0[i]+epsilon)for i in range(n)]
    res = opt.minimize(fun=lambda x: -f(*x), x0=np.array(x0), bounds=bounds)
    return np.sqrt(-res['fun'])
def get_AL(x,eps,out_poly,out_eps,w_margin,id,poly):
    res=np.load(model_path)
    mW_0, mb_0, mW_1, mb_1, dW_0, db_0, dW_1, db_1=res['mW_0'],res['mb_0'],res['mW_1'],res['mb_1'],res['dW_0'],res['db_0'],res['dW_1'],res['db_1']
    [sW_0, sb_0, sW_1, sb_1] = GLOBAL_samples
    lW_0,uW_0 = sW_0[id]-w_margin*dW_0,sW_0[id]+w_margin*dW_0
    lB_0,uB_0 = sb_0[id]-w_margin*db_0,sb_0[id]+w_margin*db_0
    lW_1,uW_1 = sW_1[id]-w_margin*dW_1,sW_1[id]+w_margin*dW_1

    y = (np.matmul(relu(np.matmul(x, sW_0[id]) + sb_0[id]), sW_1[id]) + sb_1[id])

    h_l, h_u = propagate_interval(sW_0[id], dW_0, sb_0[id], db_0, x[0,:], x[0,:], w_margin)
    h_l, h_u = my_relu(h_l), my_relu(h_u)
    y_pred_l, y_pred_u = propagate_interval(sW_1[id], dW_1, sb_1[id], db_1, h_l, h_u, w_margin)


    L = 1
    A = (lW_0.T, uW_0.T)
    b = (lB_0, uB_0)

    R = get_R(A, x.reshape(-1,1), b, eps)
    D = np.eye(A[0].shape[1])
    L *= get_RAD(R, A, D)


    A = (lW_1.T, uW_1.T)

    res = np.max(np.abs(np.array([A[0], A[1]])), axis=0)
    L *= np.sqrt(sum(sum(res * res))) 
    L_poly = Poly_L(poly,x,eps)

    diff =np.array([y_pred_l-out_poly,y_pred_u-out_poly])

    diff = np.abs(diff)
    final = np.max(diff)+(L_poly+L)*eps*np.sqrt(x.shape[1])
    
    isSafe = False
    if final <=out_eps:
        isSafe = True
    return isSafe,np.max(diff),L_poly,L,final
def get_L(x,eps,out_reg,w_margin,id):
    y_l, y_u = out_reg[0,:], out_reg[1,:]
    res=np.load(model_path)
    mW_0, mb_0, mW_1, mb_1, dW_0, db_0, dW_1, db_1=res['mW_0'],res['mb_0'],res['mW_1'],res['mb_1'],res['dW_0'],res['db_0'],res['dW_1'],res['db_1']
    [sW_0, sb_0, sW_1, sb_1] = GLOBAL_samples
    lW_0,uW_0 = sW_0[id]-w_margin*dW_0,sW_0[id]+w_margin*dW_0
    lB_0,uB_0 = sb_0[id]-w_margin*db_0,sb_0[id]+w_margin*db_0
    lW_1,uW_1 = sW_1[id]-w_margin*dW_1,sW_1[id]+w_margin*dW_1
    y = (np.matmul(relu(np.matmul(x, sW_0[id]) + sb_0[id]), sW_1[id]) + sb_1[id])

    

    L = 1
    A = (lW_0.T, uW_0.T)
    b = (lB_0, uB_0)

    R = get_R(A, x.reshape(-1,1), b, eps)
    D = np.eye(A[0].shape[1])
    L *= get_RAD(R, A, D)

    A = (lW_1.T, uW_1.T)

    res = np.max(np.abs(np.array([A[0], A[1]])), axis=0)
    L *= np.sqrt(sum(sum(res * res)))
    y_pred_l,y_pred_u=y-L*eps,y+L*eps 
    isSafe = False
    if (y_pred_l >= y_l and y_pred_u <= y_u):
        isSafe = True
    return isSafe

def interval_2layer_L(x,eps,out_poly,out_eps,
                      w_margin,id,poly):
    x = np.asarray(x); x = x.astype('float64')
    res=np.load(model_path)
    mW_0, mb_0, mW_1, mb_1,dW_1, db_1=res['mW_0'],res['mb_0'],res['mW_1'],res['mb_1'],res['dW_1'],res['db_1']

    isPass = False
    [sW_1, sb_1] = GLOBAL_samples
    h_l, h_u = propagate_interval(mW_0, 0, mb_0, 0, x[0,:], x[0,:], 0)
    h_l, h_u = my_relu(h_l), my_relu(h_u)

    y_pred_l, y_pred_u = propagate_interval(sW_1[id], dW_1, sb_1[id], db_1, h_l, h_u, w_margin)

    lx = get_L_one([sW_1[id]-w_margin*dW_1,sb_1[id]+w_margin*dW_1])
    px = Poly_L(poly,x,eps)
    diff =np.array([y_pred_l-out_poly,y_pred_u-out_poly])
    diff = np.abs(diff)
    final = np.max(diff)+(lx+px)*eps*np.sqrt(x.shape[1])
        # print(lx,px)
            # print("two layer:",[y_pred_l,y_pred_u])
    if final<=out_eps:
        isPass=True

    return isPass,final

def interval_all_L(x, eps,out_poly,out_eps,
                      w_margin,id,poly):
    x = np.asarray(x); x = x.astype('float64')
    res=np.load(model_path)
    mW,mb,mW_0, mb_0, mW_1, mb_1, dW_0, db_0, dW_1, db_1=res['mW'],res['mb'],res['mW_0'],res['mb_0'],res['mW_1'],res['mb_1'],res['dW_0'],res['db_0'],res['dW_1'],res['db_1']

    isPass = False

    layer1 = relu(np.matmul(x, mW) + mb)
    layer2 = relu(np.matmul(layer1, mW_0) + mb_0)
    y = (np.matmul(layer2, mW_1) + mb_1)
    [sW_0, sb_0, sW_1, sb_1] = GLOBAL_samples

    h_l, h_u = propagate_interval(mW, 0, mb, 0, x[0,:], x[0,:], 0)
    h_l, h_u = my_relu(h_l), my_relu(h_u)

    h_l1, h_u1 = propagate_interval(sW_0[id], dW_0, sb_0[id], db_0, h_l, h_u, w_margin)
    h_l1, h_u1 = my_relu(h_l1), my_relu(h_u1)

    y_pred_l, y_pred_u = propagate_interval(sW_1[id], dW_1, sb_1[id], db_1, h_l1, h_u1, w_margin)

    e=w_margin
    lx = get_L_all([sW_0[id]-e*dW_0,sW_0[id]+e*dW_0,sb_0[id]-e*db_0,sb_0[id]+e*db_0,sW_1[id]-e*dW_1,sW_1[id]+e*dW_1],layer1,eps)
    px = Poly_L(poly,x,eps)

    diff =np.array([y_pred_l-out_poly,y_pred_u-out_poly])
    diff = np.abs(diff)
    final = np.max(diff)+(lx+px)*eps*np.sqrt(x.shape[1])

    if final<=out_eps:   
        isPass = True

    return isPass,final

#code for propagate_interval and compute probability
#taken for here https://github.com/matthewwicker/ProbabilisticSafetyforBNNs
def propagate_interval(W, W_std, b, b_std, x_l, x_u, eps):
    W_l, W_u = W-(eps*W_std), W+(eps*W_std)   
    b_l, b_u = b-(eps*b_std), b+(eps*b_std)  
    h_max = np.zeros(len(W[0]))
    h_min = np.zeros(len(W[0])) 
    for i in range(len(W)): 
        for j in range(len(W[0])): 
            out_arr = [W_l[i][j]*x_l[i], W_l[i][j]*x_u[i],
                       W_u[i][j]*x_l[i], W_u[i][j]*x_u[i]]
            h_min[j] += min(out_arr)
            h_max[j] += max(out_arr)
    h_min = h_min + b_l
    h_max = h_max + b_u
    return h_min, h_max         

def merge_intervals(intervals):
    sorted_intervals = sorted(intervals)
    interval_index = 0
    intervals = np.asarray(intervals)
    for  i in sorted_intervals:
        if i[0] > sorted_intervals[interval_index][1]:
            interval_index += 1
            sorted_intervals[interval_index] = i
        else:
            sorted_intervals[interval_index] = [sorted_intervals[interval_index][0], i[1]]
    return sorted_intervals[:interval_index+1] 

import math
from scipy.special import erf
def compute_erf_prob(intervals, mean, stddev):
    prob = 0.0
    for interval in intervals:
        val1 = erf((mean-interval[0])/(math.sqrt(2)*(stddev)))
        val2 = erf((mean-interval[1])/(math.sqrt(2)*(stddev)))
        prob += 0.5*(val1-val2)
    return prob

def compute_interval_probs_weight(vector_intervals, marg, mean, std):
    means = mean; stds = std
    maxSigma=[]
    m,n = vector_intervals[0].shape
    safeweight =[[[] for i in range(n)] for j in range(m)]
    prob_vec = np.zeros(vector_intervals[0].shape)
    for i in range(len(vector_intervals[0])):
        for j in range(len(vector_intervals[0][0])):
            intervals = []
            for num_found in range(len(vector_intervals)):
                interval = [vector_intervals[num_found][i][j]-(stds[i][j]*marg), vector_intervals[num_found][i][j]+(stds[i][j]*marg)]
                intervals.append(interval)
            newInterval = merge_intervals(intervals)
            newInter= np.array(newInterval)
            for k in range(newInter.shape[0]):
                if newInter[k,0]<=means[i][j] and newInter[k,1]>=means[i][j]:
                    maxSigma.append(np.abs(newInter[k,:]-means[i][j]).min()/(stds[i][j]))
                    break
            p = compute_erf_prob(newInterval, means[i][j], stds[i][j])
            prob_vec[i][j] = p
            safeweight[i][j].extend(newInterval)
    if len(maxSigma)==means.shape[0]*means.shape[1]:
        maxSigma = np.array(maxSigma)
        return np.asarray(prob_vec),np.min(maxSigma),safeweight
    else:
        return np.asarray(prob_vec),0,safeweight

def compute_interval_probs_bias(vector_intervals, marg, mean, std):
    means = mean; stds = std
    maxSigma=[]
    prob_vec = np.zeros(vector_intervals[0].shape)
    m= vector_intervals[0].shape[0]
    safeweight =[[] for i in range(m)]
    for i in range(len(vector_intervals[0])):
        intervals = []
        for num_found in range(len(vector_intervals)):
            interval = [vector_intervals[num_found][i]-(stds[i]*marg), vector_intervals[num_found][i]+(std[i]*marg)]
            intervals.append(interval)
        newInterval = merge_intervals(intervals)
        newInter= np.array(newInterval)
        for k in range(newInter.shape[0]):
                if newInter[k,0]<=means[i] and newInter[k,1]>=means[i]:
                    maxSigma.append(np.abs(newInter[k,:]-means[i]).min()/(stds[i]))
                    break
        p = compute_erf_prob(newInterval, means[i], stds[i])
        prob_vec[i] = p
        safeweight[i].extend(newInterval)
    if len(maxSigma)==len(vector_intervals[0]):
        maxSigma = np.array(maxSigma)
        return np.asarray(prob_vec),np.min(maxSigma),safeweight
    else:
        return np.asarray(prob_vec),0,safeweight

import pickle

def my_relu(arr):
    arr = arr * (arr > 0)
    return arr


def relu(arr):
    return arr * (arr > 0)