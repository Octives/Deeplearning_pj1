MLP :
linear_model = nn.models.Model_MLP([train_imgs.shape[-1], 600, 10], 'ReLU')  # , [1e-4, 1e-4] 
optimizer = nn.optimizer.MomentGD(init_lr=0.006, model=linear_model)
scheduler = nn.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[800, 2400, 4000], gamma=0.6) 
loss_fn = nn.op.MultiCrossEntropyLoss(model=linear_model, max_classes=train_labs.max()+1)

runner = nn.runner.RunnerM(linear_model, optimizer, nn.metric.accuracy, loss_fn, 
                           scheduler=scheduler)

self.W = initialize_method(loc=0.0, scale=np.sqrt(2.0 / in_dim), size=(in_dim, out_dim))  # linear layer

def __init__(self, init_lr, model, momentum=0.7):
acc = 0.9812 with epoch = 15 & 0.9797 with epoch = 5
