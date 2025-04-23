import numpy as np
import matplotlib.pylab as plt

#it can input a np array
def step_function(x):
    y=x>0
    return y.astype(np.int64) #dtype must be assigned as int32 or int64, instead of int
def sigmoid (x):  #when a value calculate with a vector, the value will be broadcast
    return 1/(1+np.exp(-x))
def relu(x):
    return np.maximum(0,x)
def softmax(x):
    x=x-np.max(x)
    e=np.exp(x)
    s=sum(e)
    return e/s

x=np.arange(-5,5,0.1)
y0=step_function(x)
y1=sigmoid(x)
y2=relu(x)
plt.plot(x,y0,x,y1,x,y2)
#plt.show()
xx=np.array([0.2,0.2,0.4,0.2])
#print(np.sum(softmax(xx)))