import numpy as np

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

def get_L(data,x,eps):
    L = 1
    A = (data['lW_0'].T, data['uW_0'].T)
    b = (data['lB_0'], data['uB_0'])

    R = get_R(A, x, b, eps)
    D = np.eye(A[0].shape[1])
    L *= get_RAD(R, A, D)

    # x=relu((A[0]+A[1])/2*x+(b[0]+b[1])/2) #下一层的x
    #eps *= L  # 下一层的eps
    # D=R
    A = (data['lW_1'].T, data['uW_1'].T)
    # b=(data['lB_1'],data['uB_1'])
    res = np.max(np.abs(np.array([A[0], A[1]])), axis=0)
    L *= np.sqrt(sum(sum(res * res)))
    return L
if __name__ == '__main__':
    #data=np.load('weight_02_ranges_3std.npz')
    data = np.load('weight_3td_min2.npz')
    x = np.ones(shape=(data['lW_0'].shape[0], 1))*0.5
    eps=0.1
    print(get_L(data,x,eps))



