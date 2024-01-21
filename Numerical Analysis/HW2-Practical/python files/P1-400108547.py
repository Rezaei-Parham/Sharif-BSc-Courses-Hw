
from math import sin,sqrt
def lagrange_interpolation(x, y, x0):
    y0 = 0
    for i in range(len(x)):
        p = 1
        for j in range(len(x)):
            if i != j:
                p *= (x0 - x[j]) / (x[i] - x[j])
        y0 += p * y[i]
    return y0

def newton_difference_calculator(x,y):
    n = len(x)
    d0 = x
    d1 = y
    d2 = []
    for i in range(n-1):
        d2.append((d1[i+1]-d1[i])/(d0[i+1]-d0[i]))
    d3 = []
    for i in range(n-2):
        d3.append((d2[i+1]-d2[i])/(d0[i+2]-d0[i]))
    d4 = []
    for i in range(n-3):
        d4.append((d3[i+1]-d3[i])/(d0[i+3]-d0[i]))
    d5 = []
    for i in range(n-4):
        d5.append((d4[i+1]-d4[i])/(d0[i+4]-d0[i]))

    return [d0,d1,d2,d3,d4,d5]

def newton_interpolation(x,y,x0):
    n = len(x)
    diffs = newton_difference_calculator(x,y)
    res = 0
    for i in range(n):
        p = 1
        for j in range(i):
            p *= (x0 - x[j])
        res += p * diffs[i+1][0]
    return res


x = [2,3,5,8,10]
y = [sin(i) + sqrt(1+i) for i in x]
print(f"Part1 \nx = {x}\ny = 1/(1-x) = {y}\nreal value at 6.5: {sin(6.5)+sqrt(1+6.5)}\nlagrange_interpolation(x,y,8) = {lagrange_interpolation(x,y,6.5)}\nnewton_interpolation(x,y,8) = {newton_interpolation(x,y,6.5)}")
print("==================================")
x = [5,7,9,11]
y = [1/(1-i) for i in x]
print(f"Part2 \nx = {x}\ny = 1/(1-x) = {y}\nreal value at 8: {1/(1-8)}\nlagrange_interpolation(x,y,8) = {lagrange_interpolation(x,y,8)}\nnewton_interpolation(x,y,8) = {newton_interpolation(x,y,8)}")
print("==================================")