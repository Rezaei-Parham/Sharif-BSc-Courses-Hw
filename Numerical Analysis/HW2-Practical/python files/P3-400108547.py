
from math import sin,cos,factorial,sqrt
import numpy as np

def sin_derivative(x, i):
    sd = 0
    if i%4 == 0:
        sd = sin(x)
    elif i%4 == 1:
        sd = cos(x)
    elif i%4 == 2:
        sd = -sin(x)
    else:
        sd = -cos(x)
    xd = 0
    sign = 1
    if i>=1:
        sign = (-1)**(i-1)
    exp = -(2*i-1)/2
    coeff = 5
    for t in range(2*i-1):
        if t%2 == 1:
            coeff *= t
    coeff /= 2**i
    xd = sign*coeff*x**exp
    if i==0:
        return xd+sd-10
    return xd+sd

def e_derivative(x, i):
    coeff5 = (1/5)**i
    coeff1 = (-1)**i
    return coeff5*np.exp(x/5)+coeff1*np.exp(-x)

def taylor(derv,x, n, base):
    sum = 0
    for i in range(n+1):
        sum += derv(base, i)/factorial(i)*(x-base)**i
    return sum

# PART 1
def p1calculator(i):
    mine = taylor(sin_derivative,10,i,7)
    real = sin(10)+5*sqrt(10)-10
    print(f"PART1 iters: 10, mine: {mine}, real: {real}, error: {abs(mine-real)} which is less than 0.01")
p1calculator(10)

# PART 2
def p2calculator(i):
    mine = taylor(e_derivative,3,i,0)
    real = np.exp(3/5)+np.exp(-3)
    print(f"PART2 iters: 10, mine: {mine}, real: {real}, error: {abs(mine-real)} which is less than 0.01")
p2calculator(10)

import matplotlib.pyplot as plt
import numpy as np
from sympy import *
# taylor using sympy
x = symbols('x')
p1func = sin(x)+5*sqrt(x)-10
p2func = exp(x/5)+exp(-x)
def taylor(func,x,xeval,base,n):
    sum = 0
    for i in range(n+1):
        sum += diff(func, x, i)/factorial(i)*(xeval-base)**i
    return sum.evalf(subs={x:base})
print(f"Part 1: 10 iteration, real value: {p1func.evalf(subs={x:10})}, estimated value: {taylor(p1func,x,10,7,10)},\
 error : {abs(p1func.evalf(subs={x:10})-taylor(p1func,x,10,7,10))} which is less than 0.01")
print(f"Part 2: 10 iteration, real value: {p2func.evalf(subs={x:3})}, estimated value: {taylor(p2func,x,3,0,10)},\
 error : {abs(p2func.evalf(subs={x:3})-taylor(p2func,x,3,0,10))} which is less than 0.01")

# plots
r = np.linspace(2,10,100)
def fpart1(x):
  return np.sin(x)+5*np.sqrt(x)-10
def fpart2(x):
  return np.exp(x/5)+np.exp(-x)

print("***********Part 1***********\n")
y = [fpart1(i) for i in r]
plt.plot(r,y)
plt.title("function plot")
plt.show()
y = [taylor(p1func,x,i,7,10) for i in r]
plt.plot(r,y)
plt.title("taylor plot")
plt.show()

print("***********Part 2***********\n")
y = [fpart2(i) for i in r]
plt.plot(r,y)
plt.title("function plot")
plt.show()
y = [taylor(p2func,x,i,0,10) for i in r]
plt.plot(r,y)
plt.title("taylor plot")
plt.show()

