# codes to make visualization of your weights.
import mynn as nn
import numpy as np
from struct import unpack
import gzip
import matplotlib.pyplot as plt
import pickle

model = nn.models.Model_CNN()
model.load_model(r'./saved_models/CNN/best_model.pickle')

test_images_path = r'./dataset/MNIST/t10k-images-idx3-ubyte.gz'
test_labels_path = r'./dataset/MNIST/t10k-labels-idx1-ubyte.gz'

with gzip.open(test_images_path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        test_imgs=np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28)
    
with gzip.open(test_labels_path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        test_labs = np.frombuffer(f.read(), dtype=np.uint8)

test_imgs = test_imgs / test_imgs.max()

# logits = model(test_imgs)

mats = []
mats.append(model.layers[0].params['W'])
mats.append(model.layers[3].params['W'])
mats.append(model.layers[6].params['W'])

def visualize_conv_weights(weights, title):
    """
    可视化卷积层权重
    :param weights: 四维权重张量 [out_ch, in_ch, kh, kw]
    :param title: 子图标题
    """
    out_ch, in_ch, kh, kw = weights.shape
    
    # 计算展示布局 (每个输入通道的核展平为一行)
    rows = out_ch
    cols = in_ch
    figsize = (cols * 1.5, rows * 1.2)  # 动态调整图形大小
    
    # 创建子图
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    fig.suptitle(title, y=1.02, fontsize=12)
    
    # 绘制每个卷积核
    for i in range(out_ch):
        for j in range(in_ch):
            ax = axes[i, j] if (rows > 1 and cols > 1) else axes[j] if rows == 1 else axes[i]
            kernel = weights[i, j]
            im = ax.imshow(kernel, cmap='coolwarm', vmin=np.min(weights), vmax=np.max(weights))  # 共享颜色范围
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(f'In {j}', fontsize=8)
            if j == 0:
                ax.set_ylabel(f'Out {i}', fontsize=8)
    
    # 添加共享颜色条
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6)
    cbar.ax.tick_params(labelsize=8)
    
#     plt.tight_layout()
    return fig
 
def visualize_linear_weights(weights, title):
    """
    可视化线性层权重
    :param weights: 二维权重张量 [out_features, in_features]
    :param title: 子图标题
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(weights, cmap='coolwarm', aspect='auto')
    ax.set_title(title, fontsize=12)
    ax.set_xlabel('Input Features', fontsize=10)
    ax.set_ylabel('Output Features', fontsize=10)
    
    # 添加颜色条
    cbar = fig.colorbar(im, ax=ax, shrink=0.6)
    cbar.ax.tick_params(labelsize=8)
    
    plt.tight_layout()
    return fig
 
# 假设 model.layers[0].params['W'] 和 model.layers[3].params['W'] 是卷积层权重
conv1_weights = model.layers[0].params['W']  # 形状: [out_ch1, in_ch1, kh1, kw1]
conv2_weights = model.layers[3].params['W']  # 形状: [out_ch2, in_ch2, kh2, kw2]
 
# 可视化第一个卷积层
fig1 = visualize_conv_weights(conv1_weights, 
                            f"Conv Layer 0 Weights\nShape: {conv1_weights.shape}")
plt.savefig("conv_layer_0_weights.png", bbox_inches='tight', dpi=200)
plt.close(fig1)  # 关闭图形避免内存泄漏
 
# 可视化第二个卷积层
fig2 = visualize_conv_weights(conv2_weights, 
                            f"Conv Layer 3 Weights\nShape: {conv2_weights.shape}")
plt.savefig("conv_layer_3_weights.png", bbox_inches='tight', dpi=200)
plt.close(fig2)
 
# 可视化线性层（假设 model.layers[6].params['W'] 是线性层权重）
linear_weights = model.layers[6].params['W']  # 形状: [out_features, in_features]
fig3 = visualize_linear_weights(linear_weights, 
                             f"Linear Layer 6 Weights\nShape: {linear_weights.shape}")
plt.savefig("linear_layer_6_weights.png", bbox_inches='tight', dpi=200)
plt.close(fig3)
 
print("可视化已保存为 conv_layer_*.png 和 linear_layer_6_weights.png")