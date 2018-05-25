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

10. 

##assignment1 note


