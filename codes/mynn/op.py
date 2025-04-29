from abc import abstractmethod
import numpy as np
import random

class Layer():
    def __init__(self) -> None:
        self.optimizable = True
    
    @abstractmethod
    def forward(self, X):
        pass

    @abstractmethod
    def backward(self, grads):
        pass


class Linear(Layer):
    """
    The linear layer for a neural network. You need to implement the forward function and the backward function.
    """
    def __init__(self, in_dim, out_dim, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        self.W = initialize_method(loc=0.0, size=(in_dim, out_dim))  # , scale=np.sqrt(2.0 / in_dim)
        self.b = initialize_method(size=(1, out_dim))
        self.in_dim = in_dim
        self.out_dim = out_dim 
        self.grads = {'W' : None, 'b' : None}
        self.input = None  # Record the input for backward process.

        self.params = {'W' : self.W, 'b' : self.b}

        self.weight_decay = weight_decay  # whether using weight decay
        self.weight_decay_lambda = weight_decay_lambda  # control the intensity of weight decay

    def __call__(self, X) -> np.ndarray:
        return self.forward(X)

    def forward(self, X):
        """
        input: [batch_size, in_dim]
        out: [batch_size, out_dim]
        """
        # print(X.mean())
        # if abs(self.W.mean()) > 100:
        #     print()
        #     self.W /= abs(self.W.mean())
        # if abs(self.b.mean()) > 100:
        #     self.b /= abs(self.b.mean())
        # print(X.shape, self.W.shape, self.b.shape, self.W.mean(), self.b.mean())
        self.input = X.copy()
        batch_size = X.shape[0]
        return X @ self.W + np.ones((batch_size, 1)) @ self.b

    def backward(self, grad : np.ndarray):
        """
        input: [batch_size, out_dim] the grad passed by the next layer.
        output: [batch_size, in_dim] the grad to be passed to the previous layer.
        This function also calculates the grads for W and b.
        """
        self.grads['W'] = self.input.T @ grad
        if self.weight_decay:
            self.grads['W'] += self.weight_decay_lambda * self.W
        
        self.grads['b'] = np.sum(grad, axis=0)
        
        return grad @ self.W.T
    
    def clear_grad(self):
        self.grads = {'W': None, 'b': None}

class conv2D(Layer):
    """
    The 2D convolutional layer. Try to implement it on your own.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()

        # 将kernel_size转为元组（如果是整数）
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        # 将stride转为元组（如果是整数）
        if isinstance(stride, int):
            stride = (stride, stride)

        # 将padding转为元组（如果是整数）
        if isinstance(padding, int):
            padding = (padding, padding)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k = kernel_size
        self.stride = stride
        self.padding = padding
        self.grads = None
        self.input = None

        self.weight_decay = weight_decay
        self.weight_decay_lambda = weight_decay_lambda

        if initialize_method == np.random.normal:
            scale = np.sqrt(2.0 / (in_channels * kernel_size[0] * kernel_size[1]))
            self.W = initialize_method(
                loc=0.0,
                scale=scale,
                size=(out_channels, in_channels, *kernel_size)
            )

        self.b = np.zeros(out_channels)

        self.grads = {'W': None, 'b': None}
        self.params = {'W': self.W, 'b': self.b}

        self.optimizable = True

    def __call__(self, X) -> np.ndarray:
        return self.forward(X)
    
    def forward(self, X):
        """
        input X: [batch, channels, H, W]
        W : [1, out, in, k, k] ?
        W : [out, in, k, k] now
        no padding
        """
        self.input = X

        batch_size, _, H_in, W_in = X.shape
        # 计算输出尺寸（兼容padding）
        H_out = (H_in - self.k[0] + self.padding[0]) // self.stride[0] + 1
        W_out = (W_in - self.k[1] + self.padding[1]) // self.stride[1] + 1

        # 初始化输出张量
        output = np.zeros((batch_size, self.out_channels, H_out, W_out))

        for batch_i in range(batch_size):  # 遍历每个样本
            for c_out in range(self.out_channels):  # 遍历每个输出通道
                for h_out in range(H_out):  # 遍历输出高度
                    for w_out in range(W_out):  # 遍历输出宽度
                        # 计算输入窗口位置
                        h_start = h_out * self.stride[0]
                        h_end = h_start + self.k[0]
                        w_start = w_out * self.stride[1]
                        w_end = w_start + self.k[1]

                        # 计算点积并加偏置
                        output[batch_i, c_out, h_out, w_out] = np.sum(
                            X[batch_i, :, h_start:h_end, w_start:w_end] * self.W[c_out]
                        ) + self.b[c_out]

        # print(H_in, self.k[0])
        # print(X.shape, output.shape)
        return output

    def backward(self, grads):
        """
        grads : [batch_size, out_channel, new_H, new_W]
        """
        batch_size, _, H_in, W_in = self.input.shape
        _, __, H_out, W_out = grads.shape

        # 初始化梯度
        new_grads = np.zeros((batch_size, self.in_channels, H_in, W_in))
        self.grads['W'] = np.zeros_like(self.W)
        self.grads['b'] = np.zeros_like(self.b)

        # 计算梯度（四重循环实现）
        for b in range(batch_size):
            for c_out in range(self.out_channels):
                for h_out in range(H_out):
                    for w_out in range(W_out):
                        # 定位输入感受野位置
                        h_start = h_out * self.stride[0]
                        h_end = h_start + self.k[0]
                        w_start = w_out * self.stride[1]
                        w_end = w_start + self.k[1]

                        # 1. 计算输入梯度（累加到对应位置）
                        new_grads[b, :, h_start:h_end, w_start:w_end] += self.W[c_out] * grads[b, c_out, h_out, w_out]

                        # 2. 计算权重梯度（累加所有样本和位置的贡献）
                        self.grads['W'][c_out] += \
                            self.input[b, :, h_start:h_end, w_start:w_end] * grads[b, c_out, h_out, w_out]

                # 3. 计算偏置梯度（对输出通道所有位置求和）
                self.grads['b'][c_out] += np.sum(grads[b, c_out])

        # 添加权重衰减（L2正则化）的梯度
        # if self.weight_decay:
        #     self.weight_grad += self.weight_decay_lambda * self.weights

        return new_grads

    
    def clear_grad(self):
        self.grads = {'W': None, 'b': None}
        
class MaxPooling(Layer):
    """
    最大池化层（仅支持2D池化）
    """
    def __init__(self, pool_size=2, stride=None) -> None:
        """
        :param pool_size: 池化窗口大小（正方形）
        :param stride: 滑动步长（默认等于pool_size）
        """
        super().__init__()
        self.pool_size = pool_size
        self.stride = stride if stride is not None else pool_size
        self.input = None
        self.optimizable = False  # 池化层无可训练参数
        self.max_indices = None  # 记录最大值位置

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        """
        输入: [batch_size, channels, height, width]
        输出: [batch_size, channels, new_h, new_w]
        """
        self.input = X.copy()
        batch_size, channels, h, w = X.shape
        new_h = (h - self.pool_size) // self.stride + 1
        new_w = (w - self.pool_size) // self.stride + 1

        # 初始化输出和最大值索引
        output = np.zeros((batch_size, channels, new_h, new_w))
        self.max_indices = np.zeros((batch_size, channels, new_h, new_w, 2), dtype=int)

        # 执行池化
        for b in range(batch_size):
            for c in range(channels):
                for i in range(new_h):
                    for j in range(new_w):
                        h_start = i * self.stride
                        h_end = h_start + self.pool_size
                        w_start = j * self.stride
                        w_end = w_start + self.pool_size

                        window = X[b, c, h_start:h_end, w_start:w_end]
                        output[b, c, i, j] = np.max(window)
                        # 记录最大值位置（相对于窗口的偏移）
                        max_pos = np.unravel_index(np.argmax(window), window.shape)
                        self.max_indices[b, c, i, j] = [h_start + max_pos[0], w_start + max_pos[1]]

        return output

    def backward(self, grad):
        """
        输入 grad: [batch_size, channels, new_h, new_w]
        输出: [batch_size, channels, h, w]
        """
        batch_size, channels, h, w = self.input.shape
        _, _, new_h, new_w = grad.shape
        output_grad = np.zeros_like(self.input)

        # 将梯度传回最大值位置
        for b in range(batch_size):
            for c in range(channels):
                for i in range(new_h):
                    for j in range(new_w):
                        h_max, w_max = self.max_indices[b, c, i, j]
                        output_grad[b, c, h_max, w_max] += grad[b, c, i, j]

        return output_grad

    def clear_grad(self):
        pass  # 池化层无需梯度清除

class ReLU(Layer):
    """
    An activation layer.
    """
    def __init__(self) -> None:
        super().__init__()
        self.input = None

        self.optimizable = False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        """
        正向传播计算 ReLU(X)
        输入 X: 任意形状的numpy数组
        输出: 与X形状相同的ReLU激活结果
        """
        self.input = X
        output = np.where(X<0, 0, X)
        return output
    
    def backward(self, grads):
        """
        反向传播计算梯度
        输入 grads: 来自上一层的梯度，形状与forward输入相同
        输出: 如果输入为负，本层梯度 = 0；否则不变
        """
        assert self.input.shape == grads.shape
        output = np.where(self.input < 0, 0, grads)
        return output

class LeakyReLU(Layer):
    """
    LeakyReLU激活层
    数学表达式：
        f(x) = x   if x >= 0
             = αx  if x < 0
    其中α是负半区的固定斜率（默认0.01）
    """
    def __init__(self, alpha=0.01) -> None:
        """
        :param alpha: 负半区的斜率系数（默认0.01）
        """
        super().__init__()
        self.input = None  # 保存输入用于反向传播
        self.alpha = alpha
        self.optimizable = False  # 无训练参数

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        """
        正向传播计算 LeakyReLU(X)
        输入 X: 任意形状的numpy数组
        输出: 与X形状相同的激活结果
        """
        self.input = X  # 保存输入用于反向传播
        return np.where(X >= 0, X, self.alpha * X)

    def backward(self, grads):
        """
        反向传播计算梯度
        输入 grads: 来自上一层的梯度，形状与forward输入相同
        输出: 本层梯度 = grads * (输入>=0 ? 1 : α)
        """
        assert self.input.shape == grads.shape, "输入梯度形状不匹配"
        return np.where(self.input >= 0, grads, self.alpha * grads)

class Logistic(Layer):
    """
    Logistic (Sigmoid) activation layer.
    Sigmoid(x) = 1 / (1 + exp(-x))
    """
    def __init__(self) -> None:
        super().__init__()
        self.output = None 
        self.optimizable = False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        """
        正向传播计算 Sigmoid(X)
        输入 X: 任意形状的numpy数组
        输出: 与X形状相同的Sigmoid激活结果
        """
        # 数值稳定实现：对负值采用exp(x)/(1 + exp(x))形式
        neg_mask = (X < 0)
        pos_mask = ~neg_mask
        
        # 分区域计算避免数值溢出
        self.output = np.zeros_like(X)
        self.output[neg_mask] = np.exp(X[neg_mask]) / (1. + np.exp(X[neg_mask]))
        self.output[pos_mask] = 1. / (1. + np.exp(-X[pos_mask]))
        
        return self.output
    
    def backward(self, grads):
        """
        反向传播计算梯度
        输入 grads: 来自上一层的梯度，形状与forward输出相同
        输出: 本层梯度 = grads * sigmoid'(x) = grads * (sigmoid(x)*(1-sigmoid(x)))
        """
        assert self.output is not None, "Must call forward before backward"
        assert self.output.shape == grads.shape
        
        # sigmoid的导数: sigmoid(x)*(1-sigmoid(x))
        sigmoid_derivative = self.output * (1. - self.output)
        return grads * sigmoid_derivative

class MultiCrossEntropyLoss(Layer):
    """
    A multi-cross-entropy loss layer, with Softmax layer in it, which could be cancelled by method cancel_softmax
    """
    def __init__(self, model=None, max_classes=10) -> None:
        super().__init__()
        self.max_classes = max_classes
        self.model = model
        self.optimizable = False
        self.grads = None
        self.has_softmax = True
        self.last_predicts = None
        self.last_labels = None

    def __call__(self, predicts, labels):
        return self.forward(predicts, labels)
    
    def forward(self, predicts, labels):
        """
        predicts: [batch_size, D]
        labels : [batch_size, ]
        This function generates the loss.
        """
        self.last_labels = labels.copy()
        batch_size = labels.shape[0]

        if self.has_softmax:
            predicts_softmax = softmax(predicts)
        else:
            predicts_softmax = predicts

        self.last_predicts = predicts_softmax.copy()
        loss = 0

        epsilon = 1e-12

        for i in range(labels.shape[0]):
            loss -= np.log(predicts_softmax[i][labels[i]] + epsilon)

        return loss / batch_size

    
    def backward(self):
        # first compute the grads from the loss to the input
        # Then send the grads to model for back propagation
        self.grads = self.last_predicts
        for i in range(self.last_labels.shape[0]):
            self.grads[i][self.last_labels[i]] -= 1

        self.model.backward(self.grads)

    def cancel_soft_max(self):
        self.has_softmax = False
        return self
    
class L2Regularization(Layer):
    """
    L2 Reg can act as weight decay that can be implemented in class Linear.
    """
    pass
       
def softmax(X):
    x_max = np.max(X, axis=1, keepdims=True)
    x_exp = np.exp(X - x_max)
    partition = np.sum(x_exp, axis=1, keepdims=True)
    return x_exp / partition