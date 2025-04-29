# An example of read in the data and train the model.
# The runner is implemented, while the model used for training need your implementation.
import mynn as nn
from draw_tools.plot import plot

import numpy as np
from struct import unpack
import gzip
import matplotlib.pyplot as plt
import pickle
import time

# fixed seed for experiment
np.random.seed(309)

train_images_path = r'./dataset/MNIST/train-images-idx3-ubyte.gz'
train_labels_path = r'./dataset/MNIST/train-labels-idx1-ubyte.gz'

with gzip.open(train_images_path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        train_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28)
    
with gzip.open(train_labels_path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        train_labs = np.frombuffer(f.read(), dtype=np.uint8)


# choose 10000 samples from train set as validation set.
idx = np.random.permutation(np.arange(num))
# save the index.
with open('idx.pickle', 'wb') as f:
        pickle.dump(idx, f)
train_imgs = train_imgs[idx]
train_labs = train_labs[idx]
valid_imgs = train_imgs[:10000]
valid_labs = train_labs[:10000]
train_imgs = train_imgs[10000:]
train_labs = train_labs[10000:]

# normalize from [0, 255] to [0, 1]
train_imgs = train_imgs / 255
valid_imgs = valid_imgs / 255

train_imgs = train_imgs.reshape(num - 10000, 1, 28, 28) # 使用MLP时删除这一部分
valid_imgs = valid_imgs.reshape(10000, 1, 28, 28) # 使用MLP时删除这一部分

linear_model = nn.models.Model_CNN([(1, 28, 28), (3, 24, 24), (5, 10, 10), 10], 'Logistic', pool_list=[2, 2])  #
optimizer = nn.optimizer.SGD(init_lr=0.06, model=linear_model)
scheduler = nn.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[800, 2400, 4000], gamma=0.5) 
loss_fn = nn.op.MultiCrossEntropyLoss(model=linear_model, max_classes=train_labs.max()+1)

runner = nn.runner.RunnerM(linear_model, optimizer, nn.metric.accuracy, loss_fn, 
                           scheduler=scheduler)

start_time = time.time()
runner.train([train_imgs, train_labs], [valid_imgs, valid_labs], num_epochs=5, log_iters=500, save_dir=r'./saved_models/_')
end_time = time.time()

print("训练耗时(s)：", end_time - start_time)

# 绘制训练过程中的损失与准确率曲线
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
# plt.tight_layout()
axes.reshape(-1)
plot(runner, axes)

# plt.show()
plt.savefig("plot.png")

# 计算测试集上的预测准确率
test_images_path = r'./dataset/MNIST/t10k-images-idx3-ubyte.gz'
test_labels_path = r'./dataset/MNIST/t10k-labels-idx1-ubyte.gz'

with gzip.open(test_images_path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        test_imgs=np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28)
    
with gzip.open(test_labels_path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        test_labs = np.frombuffer(f.read(), dtype=np.uint8)

test_imgs = test_imgs / 255

test_imgs = test_imgs.reshape(num, 1, 28, 28) # 使用MLP时删除这一部分

start_time = time.time()
logits = linear_model(test_imgs)
end_time = time.time()

print("\n预测耗时(s)：", end_time - start_time)
print('test accuracy:', nn.metric.accuracy(logits, test_labs))
