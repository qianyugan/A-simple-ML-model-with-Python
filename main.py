from numpy import exp, random, dot, array

#正向推导：根据输入和权重，算出结果
def fp(input):
    l1 = 1 / (1 + exp(-dot(input, w0)))
    l2 = 1 / (1 + exp(-dot(l1, w1)))
    return l1, l2

#反向传播：用计算结果和实际结果的误差，反向推算权重的调整量
def bp(l1, l2, y):
    #看看计算出来的和实际的有多大误差
    error = y - l2#误差
    slope = l2 * (1 - l2)#也就是sigmoid函数求导得出
    l1_delta = error * slope
    #重复
    l0_slope = l1 * (1 - l1)
    l0_error = l1_delta.dot(w1.T)#dot为矩阵点积, .T是旋转矩阵
    l0_delta = l0_slope * l0_error
    #计算增量
    return l0_delta, l1_delta

#准备数据
X = array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]])
y = array([[0,1,1,0]]).T

#设置随机权重，随机值 * 2 - 1 是为了让随机值在(-1, 1)之间
random.seed(1)
w0 = random.random((3, 4)) * 2 - 1#3, 4是生成三行四列矩阵，隐藏层为四个神经元，输入层为三个神经元
w1 = random.random((4, 1)) * 2 - 1#隐藏层为四个神经元输出层为一个神经元

for i in range(10000):
    #正向推导
    l0 = X
    l1, l2 = fp(l0)
    
    #反向传播
    l0_delta, l1_delta = bp(l1, l2, y)
    
    #更新权重
    w1 = w1 + dot(l1.T, l1_delta)
    w0 = w0 + dot(l0.T, l0_delta)
    
print(fp([[1,0,1]])[1])
