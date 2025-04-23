import pickle

import mpmath

from basic import sigmoid,softmax,step_function
import numpy as np
import matplotlib
import sys,os
sys.path.append(os.pardir)
from book.dataset.mnist import load_mnist
from PIL import Image
def igshow(img):
    pil_img=Image.fromarray(np.uint8(img))
    pil_img.show()
#mnist数据集为28*28像素，每个像素取值在0-255
#(x_tri, t_tri), (x_tst, t_tst) = load_mnist(flatten=True, normalize=False)
#normalize是正规化，将数据映射到（0，1）
#   实现方法仅仅是将每个像素除以255，由于本身数据unsigned，故下界为0
#flatten表示是否将数组转变为一维数组，若为false，则输入为1*28*28
#第三个参数是one_hot_label，表示仅正解标签设置为1，其余为0
#x表示自变量，t为因变量
def getdata():
    (x_tri,t_tri),(x_tst,t_tst)=\
    load_mnist(normalize=True,flatten=True,one_hot_label=False)
    return x_tst,t_tst
def init_network():
    with open("sample_weight.pkl",'rb')as f:
        network=pickle.load(f)
    return network
def predict(network,x):
    w1,w2,w3=network['W1'],network['W2'],network['W3']
    b1,b2,b3=network['b1'],network['b2'],network['b3']
    a1=np.dot(x,w1)+b1
    z1=sigmoid(a1)
    a2=np.dot(z1,w2)+b2
    z2=sigmoid(a2)
    a3=np.dot(z2,w3)+b3
    y=softmax(a3)
    return y
x,t=getdata()
network=init_network()
acc=0
for i in range(len(x)):
    y=predict(network,x[i])
    p=np.argmax(y)
    if p==t[i]:
        acc+=1
print(float(acc)/len(x))
