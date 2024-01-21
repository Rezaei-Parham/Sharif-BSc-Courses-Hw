import numpy as np
from math import sin
import matplotlib.pyplot as plt

def estimate(x,y):
    n = len(x)
    vm = np.vander(x,6, increasing=True)
    vmv = vm.T@vm

    vmv_inv = np.linalg.inv(vmv)

    c = vmv_inv@vm.T @ y
    
    return c

def cal_x(c,x):
        y = 0
        for i in reversed(c):
            y = y * x + i
        return y

f = lambda x: x**6+3*x**2

x = np.linspace(0,1000,80)
y = [f(i) for i in x]
cof = estimate(x,y)
y_generated = np.polyval(np.flip(cof), x)
plt.plot(x,y)
plt.scatter(x, y_generated, label='Sampled points', color='red')
plt.show()
f = lambda x: np.log(x)

x = np.linspace(1,1000,80)
y = [f(i) for i in x]
cof = estimate(x,y)
y_generated = np.polyval(np.flip(cof), x)
plt.plot(x,y)
plt.scatter(x, y_generated, label='Sampled points', color='red')
plt.show()


print(f"log error: {np.mean(np.square(y-y_generated))}")