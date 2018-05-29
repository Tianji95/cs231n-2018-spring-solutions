##course note

1. use a good active function //computation cheap, will not die, will not saturate
2. preprocessing data, zero mean, subtract the mean of all the data
3. initial weight//small random numbers,但是在深度学习中weight初始化的时候，如果w比较小，而且数据饱和时，数据就会越乘越小，很快就失去数据了

如果我们将：

    W = np.random.rand(fan_in, fan_out)*0.01;换成W = np.random.rand(fan_in, fan_out)*1，让w大一些，则会只有1和0.

如果换成下面这样，就有比较好的分布了

    W = np.random.rand(fan_in, fan_out)/np.sqrt(fan_in);

4. ReLu的问题：每次都会失去一半的数据
5. 数据的正则化batch normalization：从而强制让我们中间的数据仍然保持正态分布， 而不会在中间因为一些网络的操作导致数据出现偏差
6. cost 出现nan或者inf，考虑是不是learning rate太高了
7. 超参数（hyperparameters):指的是学习速率等这种不是通过学习得来的，人工设置的参数，超参数不同的模型不同
8. momentum + SGD：添加一个速度向量:

普通的SGD：

    while True:
        dx = compute_gradient(x)
        x += learning_rate * dx

添加了momentum的SGD：

    vx = 0
    while True:
        dx = compute_gradient(x)
        vx = rho * vs + dx
        x += learning_rate * vx

Nesterov Momentum:
    dx    = compute_gradient(x)
    old_v = v
    v     = rho * v -learning_rate * dx
    x    += -rho * old_v + (1 + rho) * v

上面两个momentum的缺点在于可能会错过一些比较sharp的低点，但是通常这些比较sharp的低点并不是我们想要的local minimum，我们想要比较平缓的local minium，这样程序的鲁棒性才比较强

adaGrad：//用的不多，更多的时候用RMSProp
    
    grad_squared = 0
    while True:
        dx = compute_gradient(x)
        grad_squared += dx * dx
        x -= learning_rate * dx / (np.sqrt(grad_squared) + 1e-7)

RMSProp://不像momentum那样容易超过原来的轨道然后再回来，RMSProp更像是慢慢的接近local minimum，但是比传统SGD快很多

    grad_squared = 0
    while True:
        dx = compute_gradient(x)
        grad_squared = decay_rate * grad_squared + (1 - decay_rate) * dx * dx
        x -= learning_rate * dx / (np.sqrt(grad_squared) + 1e-7)

Adam：

    first_moment  = 0
    second_moment = 0
    beta1 = beta2 = 0.99或者其他接近于1的数字
    while True:
        dx = compute_gradient(x)
        first_moment = beta1 * first_moment + (1 - beta1) * dx //momentum
        second_moment = beta2 * second_moment + (1 - beta2) * dx * dx       //adaGrad /RMSProp
        x -= learning_rate * first_moment / (np.sqrt(second_moment) + 1e-7)


Adam(full form)：

    first_moment  = 0
    second_moment = 0
    beta1 = beta2 = 0.99或者其他接近于1的数字
    while True:
        dx = compute_gradient(x)
        first_moment  = beta1 * first_moment + (1 - beta1) * dx //momentum
        second_moment = beta2 * second_moment + (1 - beta2) * dx * dx       //adaGrad /RMSProp
        first_unbias  = first_moment / (1 - beta1 ** t)
        second_unbias = second_moment / (1 - beta2 **t) //Bias correction, 
        x -= learning_rate * first_moment / (np.sqrt(second_moment) + 1e-7)

9. dropout : 防止过拟合，将网络中的一些节点随机拿掉，类似的还有：

Batch Normalization,
Data Augmentation,
DropConnect,
Fractional Max Pooling,
Stochastic Depth,


**写代码过程中的笔记**

## assignment1 note

**配置环境**

首先是配置环境（我是windows系统），下载完代码以后windows系统需要执行下面的命令（没有安装python环境的需要安装python环境，建议安装python3.6）：
    
    pip install ipython
    pip install jupyter
    pip install numpy
    pip install matplotlib

之后建议将下载下来的代码文件放在桌面上面，因为ipython不太好访问其他盘符，放在桌面是最好找的。环境配置好以后按win+R，输入

    ipython3 notebook # 因为是python3.6版本，所以这里要用ipython3而不是ipython notebook

进入到浏览器界面，打开knn.ipynb文件，依次执行代码（中间遇到报错的需要自己依次解决）

比如我自己下载的CIFAR-10文件路径不对，这里就需要根据打开的 cs231n\datasets\get_datasets.sh 文件中具体的路径, 我这里是[http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz](http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)下载解压，放到dataset文件夹下面

执行到第6的代码块的时候，需要自己实现KNearestNeighbor


**KNN**

[这里](https://github.com/Tianji95/CS231n-Assignment-Solutions-Spring-2018/blob/master/assignment1/cs231n/classifiers/k_nearest_neighbor.py)和
[这里](https://github.com/Tianji95/CS231n-Assignment-Solutions-Spring-2018/blob/master/assignment1/knn.py)是我的KNN代码

另外ipython文件最好用ipython notebook打开，上面的knn.py只是我导出的一个py文件

1. data_utils里面可以看出他们对源数据做了Normalize，具体在get_CIFAR10_data里面实现
2. KNN的two就不说了，one loop需要注意sum的时候axis=1，要不然加起来矩阵的方向是反的，no loop需要稍微思考一下矩阵运算，这里主要是要会用numpy的broadcast sum
+ [这里是broadcast sum的官方文档](https://docs.scipy.org/doc/numpy-1.13.0/user/basics.broadcasting.html)
+ [以及一个Stack Overflow](https://stackoverflow.com/questions/27948363/numpy-broadcast-to-perform-euclidean-distance-vectorized?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa)
3. predict_labels里面要注意bincount和argmax的使用
4. 代码截图见assignment1/screenshot目录
5. validation：要让每个folder都有机会成为validation，所以循环中要拼接所有不是validation的folder，让他们成为train set 


**linear SVM**

1. 因为是多分类的svm，所以具体的做法需要回溯到lecture 3，里面讲了如何确定w矩阵,具体主要是难在如何确定dw，对svm公式求偏导， 参见[官方笔记](http://cs231n.github.io/optimization-1/)，即当出现偏差的时候偏导数为xi，没有偏差的时候偏导数为0，而当i和j相等的时候有偏差为所有不等于i的-xi的和

2. 向量化的svm计算方式和前面相同，只不过需要把需要加上的偏导数换成矩阵运算

**SoftMax**

1. 和SVM一样，主要难点在于如何确定dw，以及多了一点就是确保稳定性，参见[官方笔记](http://cs231n.github.io/linear-classify/#softmax)

**Neural  Network**

1. 这里真的需要自己推一边偏导数了，grad的计算一直都是比较难的，自己推一遍 softmax(Relu(W1 * X + b1) * W2 + b2)对w和b的偏导数，计算起来就会顺手很多