import scipy.optimize as opt
import numpy as np
import sympy

def Poly_L(y, x0, epsilon):## y:字符串表达式,x0：求解位置，epsilion：求解范围
    x0=list(x0)
    n=len(x0)
    print(n)
    x=sympy.symbols(['x{}'.format(i)for i in range(n)])
    y = sympy.sympify(y)#将字符串转化为 sympy 表达式
    op=0
    for i in range(n):
        op+=(sympy.diff(y,x[i]))**2
    f=sympy.lambdify(x,op)#将sympy表达式转化为python 可执行函数
    bounds = [(x0[i]-epsilon,x0[i]+epsilon)for i in range(n)]  # 定义域
    res = opt.minimize(fun=lambda x: -f(*x), x0=np.array(x0), bounds=bounds)
    return np.sqrt(-res['fun'])

if __name__ == '__main__':
    y='-0.00657296681332328*x0**2 + 0.000127946182030519*x0*x1 + 7.97947335613378e-5*x0 - 0.00811991115249655*x1**2 + 0.00012049188211806*x1 + 0.00223922056122807'

    print(Poly_L(y,[1,2],0.1))