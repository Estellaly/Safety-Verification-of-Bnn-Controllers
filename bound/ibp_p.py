from matplotlib.pyplot import flag
import numpy as np
import joblib
from torch import float64
class IntervalNumber:
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def __str__(self):
        return '[{0},{1}]'.format(self.a, self.b)

    def __add__(self, other):
        return _add(self, other)

    def __sub__(self, other):
        return _sub(self, other)

    def __mul__(self, other):
        return _mul(self, other)

    def __pow__(self, power, modulo=None):
        return _pow(self,power)


def _add(I, J):
    if type(I) in [int ,float,np.float64]:
        I=IntervalNumber(I,I)
    if type(J) in [int ,float,np.float64]:
        J=IntervalNumber(J,J)

    result = IntervalNumber(I.a + J.a, I.b + J.b)
    return result

def _sub(I, J):
    if type(I) in [int ,float,np.float64]:
        I=IntervalNumber(I,I)
    if type(J) in [int ,float,np.float64]:
        J=IntervalNumber(J,J)
    result = IntervalNumber(I.a - J.b, I.b - J.a)
    return result


def _mul(I, J):
    if type(I) in [int ,float,np.float64]:
        I=IntervalNumber(I,I)
    if type(J) in [int ,float,np.float64]:
        J=IntervalNumber(J,J)
    V=[I.a*J.a,I.a*J.b,I.b*J.a,I.b*J.b]
    result = IntervalNumber(min(V), max(V))
    return result
def _pow(I,p):
    if type(I) in [int ,float,np.float64]:
        I=IntervalNumber(I,I)
    low=I.a**p
    up=I.b**p

    if I.a<=0<=I.b:#入过区间包含原点
        l=min(min(low,up),0)
    else:
        l=min(low,up)
    result = IntervalNumber(l,max(low,up))
    return result

def ibp_poly(x_range,inter,coef):
    x1=IntervalNumber(x_range[0,0],x_range[1,0])
    x2=IntervalNumber(x_range[0,1],x_range[1,1])
    coef = np.array(coef,dtype=float)
    coef = list(coef.reshape(-1))
    inter = inter+coef[0]
    if  x_range.shape[1]==3:
        x3=IntervalNumber(x_range[0,2],x_range[1,2])
        # xs = np.array([1,x1,x2,x3,x1**2,x1*x2,x1*x3,x2*x2,x2*x3,x3**2,x1**3,x1**2*x2,x1**2*x3,x1*x2**2,x1*x2*x3,x1*x3**2,x2**3,x2**2*x3,x2*x3**2,x3**3])
        res= x1*coef[1]+ x2*coef[2]+x3*coef[3]+ x1*x1*coef[4]+ x1*x2*coef[5] +x1*x3*coef[6]+x2*x2*coef[7]+x2*x3*coef[8]+x3*x3*coef[9]
    else:
        res=x1*coef[1]+ x2*coef[2]+ x1**2*coef[3]+ x1*x2*coef[4]+ x2**2*coef[5]
    return res.a+inter,res.b+inter


if __name__ == '__main__':
    x1=IntervalNumber(-1,1)
    x2=IntervalNumber(-2,1)    
    # 使用例子
    def fun(x1,x2):
        return x1*0.2-x2**2+x1*x2
    print(fun(x1,x2))
    id=10
    poly_path=f"/home/jasminezli/home/code/AIBNN/BNN_controller/data/{id}/{id}_poly_2.pkl"
    lin_reg_2=joblib.load(poly_path)
    rr=ibp_poly(np.array([[-1,-2],[-0.9,-1.9]]),lin_reg_2.intercept_,lin_reg_2.coef_)
    print(rr)