
from math import sin,cos,factorial
df =[lambda x: cos(x)+4*x-2,lambda x: -sin(x)+4, lambda x: -cos(x), lambda x: sin(x), lambda x: cos(x), lambda x: -sin(x), lambda x: -cos(x), lambda x: sin(x), lambda x: cos(x), lambda x: -sin(x), lambda x: -cos(x)]

def newton(df,x0,tol=1e-5,maxiter=1000):
    x = x0
    for _ in range(maxiter):
        x = x - df[0](x)/df[1](x)
        if abs(df[0](x)) < tol:
            return x
    return x

def taylor_coeffs(x,base,df):
    func = 0
    for i in range(11):
        func += df[i](base)*((x-base)**i)/factorial(i)
    dfunc = 0
    for i in range(10):
        dfunc += df[i+1](base)*((x-base)**i)/factorial(i)

    return func,dfunc

def newton_on_taylor(base,df,x0):
    d = [lambda x: taylor_coeffs(x,base,df)[0], lambda x: taylor_coeffs(x,base,df)[1]]
    return newton(d,x0)

print(f"Only newton: {newton(df,0)}")
print(f"Using taylor on -3: {newton_on_taylor(-3,df,0)}")
print(f"Using taylor on 4: {newton_on_taylor(4,df,0)}")
print(f"Using taylor on 7: {newton_on_taylor(7,df,0)}")
