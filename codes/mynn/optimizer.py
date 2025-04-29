from abc import abstractmethod
import numpy as np


class Optimizer:
    def __init__(self, init_lr, model) -> None:
        self.init_lr = init_lr
        self.model = model

    @abstractmethod
    def step(self):
        pass


class SGD(Optimizer):
    def __init__(self, init_lr, model):
        super().__init__(init_lr, model)
    
    def step(self):
        for layer in self.model.layers:
            if layer.optimizable == True:
                for key in layer.params.keys():
                    if layer.weight_decay:
                        layer.params[key] *= (1 - self.init_lr * layer.weight_decay_lambda)
                    layer.params[key] -= self.init_lr * layer.grads[key]


class MomentGD(Optimizer):
    def __init__(self, init_lr, model, momentum=0.7):
        """
        带动量的梯度下降优化器
        
        参数:
            init_lr: 初始学习率
            model: 要优化的模型
            momentum: 动量系数（默认0.7）
        """
        super().__init__(init_lr, model)
        self.momentum = momentum
        # 初始化速度项
        self.velocities = {}
        for layer in self.model.layers:
            if layer.optimizable:
                self.velocities[layer] = {
                    key: np.zeros_like(layer.params[key]) 
                    for key in layer.params.keys()
                }
    
    def step(self):
        """
        执行一步参数更新，使用动量方法
        """
        for layer in self.model.layers:
            if layer.optimizable:
                for key in layer.params.keys():
                    # 1. 计算当前速度（动量项）
                    self.velocities[layer][key] = (
                        self.momentum * self.velocities[layer][key] 
                        - self.init_lr * layer.grads[key]
                    )
                    
                    # 2. 应用权重衰减（L2正则化）
                    if layer.weight_decay:
                        layer.params[key] *= (1 - self.init_lr * layer.weight_decay_lambda)
                    
                    # 3. 应用速度更新参数
                    layer.params[key] += self.velocities[layer][key]