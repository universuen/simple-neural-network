import math


# 一维向量点乘
def dot(a: list, b: list) -> float:
    assert len(a) == len(b)
    result = 0
    for i, j in zip(a, b):
        result += i * j
    return result


# 转置矩阵与向量相乘
def mt_v_mul(m: list, v: list) -> list:
    assert len(m) == len(v)
    # 有点困了，先用一个很笨的方法得到转置矩阵 =_=
    mt = [[0 for j in range(len(m))] for i in range(len(m[0]))]
    for i in range(len(m)):
        for j in range(len(m[0])):
            mt[j][i] = m[i][j]
    # 然后用一个很笨的方法算乘法
    result = []
    for i in range(len(mt)):
        temp = 0
        for j in range(len(mt[0])):
            temp = temp + mt[i][j] * v[j]
        result.append(temp)
    return result


# 激活函数
def sigmoid(x: float) -> float:
    return 1/(1 + math.exp(-x))


# 激活函数的导函数
def d_sigmoid(x: float) -> float:
    return math.exp(-x)/((1 + math.exp(-x))**2)


# 代价函数
def cost(y: list, a: list) -> float:
    assert len(y) == len(a)
    result = 0
    for i, j in zip(y, a):
        result += (i - j)**2
    return result/(2 * len(a))


# 代价函数的导函数
def d_cost(y: list, a: list) -> list:
    assert len(y) == len(a)
    result = []
    for i, j in zip(y, a):
        result.append(j - i)
    return result


# 神经元
class Neu:
    def __init__(self, inputs_number: int, b=0, *w: float):
        self.n = inputs_number  # 输入值个数
        self.b = b  # 偏置值
        self.w = list(w)  # 权重值
        self.z = -1;  # 存储最近一次的输入值
        self.store = -1  # 存储最近一次的激活值
        assert len(self.w) <= inputs_number
        if len(self.w) < inputs_number:
            for i in range(inputs_number - len(self.w)):
                self.w.append(0)

# 激活一次神经元
    def activate(self, x: list) -> float:
        assert self.n == len(x)
        self.z = dot(x, self.w) + self.b
        self.store = sigmoid(self.z)
        return self.store  # 返回神经元激活值


# 神经网络
class NetWork:
    def __init__(self, number: list, rate: float, ):
        number = list(number)
        self.rate = rate  # 设置学习步长
        self.layer_number = len(number)  # 神经网络层数
        self.neus = [[Neu(1) if j == 0 else Neu(number[j - 1]) for i in range(number[j])] for j in range(self.layer_number)]  # 创建神经网络

# 激活一次神经网络
    def activate(self, original_data: list) -> list:
        for i, j in zip(self.neus[0], original_data):  # 激活第一层神经元
            i.activate([j])
        for i in range(1, self.layer_number):  # 迭代激活各层神经元
            for j in self.neus[i]:
                j.activate([self.neus[i-1][k].store for k in range(len(self.neus[i - 1]))])
        return [self.neus[self.layer_number - 1][i].store for i in range(len(self.neus[self.layer_number - 1]))] # 返回包含网络输出值的列表

# 反向传播
    def backward(self, answers: list, original_data: list):
        assert len(answers) == len(self.neus[self.layer_number - 1])
        temp = [d_sigmoid(i.z) for i in self.neus[self.layer_number - 1]]  # sigmoid函数无法直接对向量运算， 因此这里使用temp作为过渡
        delta = [i * j for i, j in zip(d_cost(answers, self.activate(original_data)), temp)]  # 激活一次神经网络并计算输出层误差
        # 调整输出层的b, w
        for i, j in zip(delta, self.neus[self.layer_number - 1]):
            j.b = j.b - self.rate * i
            for k in range(j.n):
                j.w[k] = j.w[k] - self.rate * i * self.neus[self.layer_number - 2][k].store
        #  反向传播，迭代调整每一层的b, 输入层单独处理
        for i in range(self.layer_number - 2, 0, -1):
            # 由上一层误差计算本层误差
            # 首先获取上一层的w矩阵
            w_m = [j.w for j in self.neus[i + 1]]
            # 然后算w_m的转置矩阵与上一层误差向量的乘积
            delta = mt_v_mul(w_m, delta)
            # 最后deltal与本层输入向量作Hardamard积得到本层误差
            temp = [d_sigmoid(k.z) for k in self.neus[i]]
            delta = [i1 * j1 for i1, j1 in zip(delta, temp)]
            # 调整本层的b, w
            for i1, j1 in zip(delta, self.neus[i]):
                j1.b = j1.b - self.rate * i1
                for k1 in range(j1.n):
                    j1.w[k1] = j1.w[k1] - self.rate * i1 * self.neus[i - 1][k1].z
        # 调整输入层的b, w
        w_m = [j.w for j in self.neus[1]]
        delta = mt_v_mul(w_m, delta)
        temp = [d_sigmoid(k.z) for k in self.neus[0]]
        delta = [i * j for i, j in zip(delta, temp)]
        for i, j in zip(range(len(delta)), self.neus[0]):
            j.b = j.b - self.rate * delta[i]
            for k in range(j.n):
                j.w[k] = j.w[k] - self.rate * delta[i] * original_data[i]
