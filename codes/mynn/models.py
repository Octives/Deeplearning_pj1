from mynn.op import *
import pickle
import os

class Model_MLP(Layer):
    """
    A model with linear layers. We provied you with this example about a structure of a model.
    """
    def __init__(self, size_list=None, act_func=None, lambda_list=None, batch_norm=False, bn_params=[0, 1, 0.9]):
        self.size_list = size_list
        self.act_func = act_func

        self.batch_norm = batch_norm
        self.bn_mean = bn_params[0]
        self.bn_var = bn_params[1]
        self.bn_momentum = bn_params[2]

        if size_list is not None and act_func is not None:
            self.layers = []
            for i in range(len(size_list) - 1):
                layer = Linear(in_dim=size_list[i], out_dim=size_list[i + 1])
                if lambda_list is not None:
                    layer.weight_decay = True
                    layer.weight_decay_lambda = lambda_list[i]
                if act_func == 'Logistic':
                    layer_f = Logistic()
                elif act_func == 'ReLU':
                    layer_f = ReLU()
                elif self.act_func == 'LeakyReLU':
                    layer_f = LeakyReLU()
                self.layers.append(layer)
                if i < len(size_list) - 2:
                    self.layers.append(layer_f)

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        assert self.size_list is not None and self.act_func is not None, \
            'Model has not initialized yet. Use model.load_model to load a model or create a new model with size_list and act_func offered.'
        outputs = X
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs

    def backward(self, loss_grad):
        grads = loss_grad
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
        return grads

    def load_model(self, param_list):
        with open(param_list, 'rb') as f:
            param_list = pickle.load(f)
        self.size_list = param_list[0]
        self.act_func = param_list[1]
        self.batch_norm = param_list[2]
        self.bn_mean, self.bn_var, self.bn_momentum = param_list[3]

        self.layers = []
        for i in range(len(self.size_list) - 1):
            layer = Linear(in_dim=self.size_list[i], out_dim=self.size_list[i + 1])
            layer.W = param_list[i + 4]['W']
            layer.b = param_list[i + 4]['b']
            layer.params['W'] = layer.W
            layer.params['b'] = layer.b
            layer.weight_decay = param_list[i + 4]['weight_decay']
            layer.weight_decay_lambda = param_list[i + 4]['lambda']
            if self.act_func == 'Logistic':
                layer_f = Logistic()
            elif self.act_func == 'ReLU':
                layer_f = ReLU()
            elif self.act_func == 'LeakyReLU':
                layer_f = LeakyReLU()
            self.layers.append(layer)
            if i < len(self.size_list) - 2:
                self.layers.append(layer_f)
        
    def save_model(self, save_path):
        bn_params = (self.bn_mean, self.bn_var, self.bn_momentum)
        param_list = [self.size_list, self.act_func, self.batch_norm, bn_params]
        for layer in self.layers:
            if layer.optimizable:
                param_list.append({'W' : layer.params['W'], 'b' : layer.params['b'], 'weight_decay' : layer.weight_decay, 'lambda' : layer.weight_decay_lambda})
        
        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)

        file_size = os.path.getsize(save_path)
        print(f"模型已保存到: {save_path}")
        print(f"文件大小: {self._format_size(file_size)}")

    def _format_size(self, size_bytes):
        """
        将字节大小转换为友好格式
        :param size_bytes: 文件大小（字节）
        :return: 格式化后的字符串
        """
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.2f} TB"
        

class Model_CNN(Layer):
    """
    A model with conv2D layers. Implement it using the operators you have written in op.py
    """
    def __init__(self, size_list=None, act_func=None, lambda_list=None, pool_list=None):
        self.size_list = size_list
        self.act_func = act_func
        self.pool_list = pool_list
        self.out_dim_0 = None
        self.out_dim_1 = None
        self.out_dim_2 = None

        if size_list is not None and act_func is not None:
            self.layers = []
            for i in range(len(size_list) - 1):
                if isinstance(size_list[i], tuple):
                    if isinstance(size_list[i + 1], tuple):
                        if pool_list is not None and i > 0:
                            kernel_size = size_list[i][1] // pool_list[i - 1] - size_list[i + 1][1] + 1
                        else:
                            kernel_size = size_list[i][1] - size_list[i + 1][1] + 1
                        layer = conv2D(in_channels=size_list[i][0], out_channels=size_list[i + 1][0], kernel_size=kernel_size)
                    else:
                        self.out_dim_0 = size_list[i][0]
                        if pool_list is not None:
                            self.out_dim_1 = size_list[i][1] // pool_list[i - 1]
                            self.out_dim_2 = size_list[i][1] // pool_list[i - 1]
                        else:
                            self.out_dim_1 = size_list[i][1]
                            self.out_dim_2 = size_list[i][1]

                        in_dim = size_list[i][0] * (self.out_dim_1 ** 2)
                        layer = Linear(in_dim=in_dim, out_dim=size_list[i + 1])
                else:    
                    layer = Linear(in_dim=size_list[i], out_dim=size_list[i + 1])

                if lambda_list is not None:
                    layer.weight_decay = True
                    layer.weight_decay_lambda = lambda_list[i]

                if act_func == 'Logistic':
                    layer_f = Logistic()
                elif act_func == 'ReLU':
                    layer_f = ReLU()
                elif self.act_func == 'LeakyReLU':
                    layer_f = LeakyReLU()

                self.layers.append(layer)
                if i < len(size_list) - 2:
                    self.layers.append(layer_f)

                if pool_list is not None and i < len(pool_list):
                    layer_p = MaxPooling(pool_size=pool_list[i])
                    self.layers.append(layer_p)

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        assert self.size_list is not None and self.act_func is not None, \
            'Model has not initialized yet. Use model.load_model to load a model or create a new model with size_list and act_func offered.'
        outputs = X
        linear = False
        for layer in self.layers:
            if not linear and isinstance(layer, Linear):
                linear = True
                size_0 = outputs.shape[0]
                outputs = outputs.reshape(size_0, layer.in_dim)
            outputs = layer(outputs)
        return outputs

    def backward(self, loss_grad):
        grads = loss_grad
        linear = True
        for layer in reversed(self.layers):
            if linear and (self.out_dim_0 * self.out_dim_1 * self.out_dim_2 == grads.shape[1] or isinstance(layer, MaxPooling) or isinstance(layer, conv2D)):
                linear = False
                size_0 = grads.shape[0]
                grads = grads.reshape(size_0, self.out_dim_0, self.out_dim_1, self.out_dim_2)
            grads = layer.backward(grads)
        return grads
    
    def load_model(self, path):
        with open(path, 'rb') as f:
            param_list = pickle.load(f)
        self.size_list = param_list[0]
        self.act_func = param_list[1]
        self.pool_list = param_list[2]

        self.layers = []
        for i in range(len(self.size_list) - 1):
            if isinstance(self.size_list[i], tuple):
                if isinstance(self.size_list[i + 1], tuple):
                    if self.pool_list is not None and i > 0:
                        kernel_size = self.size_list[i][1] // self.pool_list[i - 1] - self.size_list[i + 1][1] + 1
                    else:
                        kernel_size = self.size_list[i][1] - self.size_list[i + 1][1] + 1
                    layer = conv2D(in_channels=self.size_list[i][0], out_channels=self.size_list[i + 1][0], kernel_size=kernel_size)
                else:
                    self.out_dim_0 = self.size_list[i][0]
                    if self.pool_list is not None:
                        self.out_dim_1 = self.size_list[i][1] // self.pool_list[i - 1]
                        self.out_dim_2 = self.size_list[i][1] // self.pool_list[i - 1]
                    else:
                        self.out_dim_1 = self.size_list[i][1]
                        self.out_dim_2 = self.size_list[i][1]

                    in_dim = self.size_list[i][0] * (self.out_dim_1 ** 2)
                    layer = Linear(in_dim=in_dim, out_dim=self.size_list[i + 1])
            else:    
                layer = Linear(in_dim=self.size_list[i], out_dim=self.size_list[i + 1])

            layer.W = param_list[i + 3]['W']
            layer.b = param_list[i + 3]['b']
            layer.params['W'] = layer.W
            layer.params['b'] = layer.b
            layer.weight_decay = param_list[i + 3]['weight_decay']
            layer.weight_decay_lambda = param_list[i + 3]['lambda']

            if self.act_func == 'Logistic':
                layer_f = Logistic()
            elif self.act_func == 'ReLU':
                layer_f = ReLU()
            elif self.act_func == 'LeakyReLU':
                layer_f = LeakyReLU()

            self.layers.append(layer)
            if i < len(self.size_list) - 2:
                self.layers.append(layer_f)

            if i < len(self.pool_list):
                layer_p = MaxPooling(pool_size=self.pool_list[i])
                self.layers.append(layer_p)
        
    def save_model(self, save_path):
        param_list = [self.size_list, self.act_func, self.pool_list]
        for layer in self.layers:
            if layer.optimizable:
                param_list.append({'W' : layer.params['W'], 'b' : layer.params['b'], 'weight_decay' : layer.weight_decay, 'lambda' : layer.weight_decay_lambda})
        
        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)

        file_size = os.path.getsize(save_path)
        print(f"模型已保存到: {save_path}")
        print(f"文件大小: {self._format_size(file_size)}")

    def _format_size(self, size_bytes):
        """
        将字节大小转换为友好格式
        :param size_bytes: 文件大小（字节）
        :return: 格式化后的字符串
        """
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.2f} TB"