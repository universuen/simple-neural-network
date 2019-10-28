import math
# 一维向量点乘
def dot(a, b):
    assert len(a) == len(b)
    result = 0
    for i, j in zip(a, b):
        result += i * j
    return result
# 代价函数
def cost(a, b):
    assert len(a) == len(b)
    result = 0;
    for i, j in zip(a, b):
        result += (i - j)**2
    return result/len(a)

# 神经元
class Neu:
    def __init__(self, inputs_number, b = 0, *w ):
        self.n = inputs_number
        self.b = b
        self.w = list(w)
        self.store = -1
        assert len(self.w) <= inputs_number
        if len(self.w) < inputs_number:
            for i in range(inputs_number - len(self.w)):
                self.w.append(0)
# 激活一次神经元
    def activate(self, x):
        assert self.n == len(x)
        self.store = dot(x, self.w) + self.b
        return 1/(1 + math.exp(self.store))

# 神经网络
class NetWork:
    def __init__(self, *number):
        self.layer_number = len(number)
        self.neus = [[Neu(1) if j == 0 else Neu(number[j - 1]) for i in range(number[j])] for j in range(self.layer_number)]
# 激活一次神经网络
    def activate(self, *original_data):
        original_data = list(original_data)
        assert len(original_data) == len(self.neus[0])
        for i, j in zip(self.neus[0], original_data):
            i.activate([j])
        for i in range(1, self.layer_number):
            for j in self.neus[i]:
                j.activate([self.neus[i-1][k].store for k in range(len(self.neus[i - 1]))])
        return [self.neus[self.layer_number - 1][i].store for i in range(len(self.neus[self.layer_number - 1]))]

# 反向传播
    def backward(self, *answers):
        answers = list(answers)
        assert len(answers) == len(self.neus[self.layer_number - 1])

