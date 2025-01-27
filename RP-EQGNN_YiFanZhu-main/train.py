import time
import datetime
import numpy as np
import torch
import torch.nn.functional as F
from graph_regression_qm9.process.sample_graph import SampleGraph
import sys
import os
from graph_regression_qm9.process.get_dataloader import get_dataloader
from utils.callbacks import LossHistory
from tqdm import tqdm
from graph_regression_qm9.model.qgnn import QGNN
from torchmetrics import MeanAbsoluteError


# 准备训练
epochs = 10
cuda = torch.cuda.is_available()

molecular_property = 'zpve'

# # 设置多少个epoch保存一次权值
# save_period = 500
# # 权值与日志文件保存的文件夹
# save_dir = 'logs'
# time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
# log_dir = os.path.join(save_dir, "loss_" + str(time_str))
# loss_history = LossHistory(log_dir)

model = QGNN()

device = torch.device('cuda' if cuda else 'cpu')


model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-16)
# 调整学习率, 周期性变化
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, verbose=False)

print('> start training')

# 测试 -> model
train_loader, val_loader, test_loader, _ = get_dataloader(num_workers=0)
tr_ys = train_loader.dataset.data[molecular_property]

train_loss = []
val_loss = []
test_loss = []

for epoch in range(epochs):
    start = time.time()

    graph_train_loss = []
    graph_val_loss = []

    model.train()

    loop_train = tqdm(train_loader, file=sys.stdout, desc=f'Epoch [{epoch + 1}/{epochs}]' + 'train')

    for graph in loop_train:
        graph = SampleGraph(graph, molecular_property, cuda)
        # print(graph)
        print(f"graph.edges type: {type(graph.edges)}, device: {graph.edges.device}")
        edges = graph.edges.cpu().numpy().reshape(-1)
        model_instance = QGNN(graph.num[-1].cpu().numpy(), edges).to(device)
        graph.h=graph.h.to(device)
        graph.x = graph.x.to(device)
        graph.y = graph.y.to(device)
        graph.edges = graph.edges.to(device)
        out = model_instance(graph).reshape(-1)
        loss = MeanAbsoluteError()(out, graph.y)

        # 清零
        optimizer.zero_grad()
        # 反向传播
        loss.backward()
        # 更新
        optimizer.step()
        # 每一个graph的loss序列
        graph_train_loss += [float(loss.data.cpu().numpy())]

        # 更新训练信息, 最新的loss值
        loop_train.set_postfix(loss='{:.3f}'.format(graph_train_loss[-1]))

    #  整体求平均 / 0.001（增大幅度）
    # train_loss += [np.mean(graph_train_loss) / 0.001] train_loss = []

    # 一个epoch中所有graph的loss的平均值列表
    # train_loss += [np.mean(graph_train_loss)]
    train_loss += [np.mean(graph_train_loss)]
    loop_train.close()

    # print(train_loss, train_loss[-1], type(train_loss[-1]))

    # 存储列表中的最新值
    # loss_history.append_loss(epoch + 1, train_loss[-1])

    with torch.no_grad():
        model.eval()

        loop_val = tqdm(val_loader, file=sys.stdout, desc=f'Epoch [{epoch + 1}/{epochs}]' + 'val')
        for graph in loop_val:
            graph = SampleGraph(graph, molecular_property, cuda)
            graph.to(device)
            # edges = np.array(graph.edges).reshape(-1)
            # model_instance = QGNN(graph.num[-1].numpy(), edges).to(device)
            # out = model_instance(graph).reshape(-1)
            # loss = F.l1_loss(out, graph.y)
            print("成功")
            edges = graph.edges.cpu().numpy().reshape(-1)
            model_instance = QGNN(graph.num[-1].cpu().numpy(), edges).to(device)
            out = model_instance(graph).reshape(-1)
            loss = F.l1_loss(out, graph.y.to(device))
            graph_val_loss += [float(loss.data.cpu().numpy())]

            loop_val.set_postfix(loss='{:.3f}'.format(graph_val_loss[-1]))

        #  / 0.001
        val_loss += [np.mean(graph_val_loss)]

        loop_val.close()

        tqdm.write('\n第' + str(epoch + 1) + '个epoch: ', end=' ')
        tqdm.write('train %.3f' % train_loss[-1], end=' ')
        tqdm.write('val %.3f' % val_loss[-1], end=' ')

        # end = time.time()
        # print('epoch耗时: (%.1f sec) \n' % (end - start))

        # loss_history.append_loss(epoch + 1, train_loss[-1], val_loss[-1])
        # print(' ')
        # #  保存权值（每n次）
        # if (epoch + 1) % save_period == 0:
        #     torch.save(model.state_dict(), os.path.join(save_dir, 'ep%03d-loss%.3f-val_loss%.3f.pth' % (
        #         (epoch + 1), train_loss[-1], val_loss[-1])))
        #
        # if len(loss_history.val_loss) <= 1 or val_loss[-1] <= min(loss_history.val_loss):
        #     # print('Save best model to best_epoch_weights.pth')
        #     torch.save(model.state_dict(), os.path.join(save_dir, "best_epoch_weights.pth"))
        #
        # torch.save(model.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))
        #
        # lr_scheduler.step()


with torch.no_grad():
    model.eval()
    # 测试

    graph_test_loss = []

    loop_test = tqdm(test_loader, file=sys.stdout)
    for graph in loop_test:
        graph = SampleGraph(graph, molecular_property, cuda)
        edges = np.array(graph.edges).reshape(-1)
        model = QGNN(graph.num[-1].numpy(), edges)
        out = model(graph).reshape(-1)
        loss = F.l1_loss(out, graph.y)
        graph_test_loss += [float(loss.data.cpu().numpy())]

        # 更新最新的loss值
        loop_test.set_postfix(loss='{:.3f}'.format(graph_test_loss[-1]))

    #  / 0.001
    test_loss += [np.mean(graph_test_loss)]

    loop_test.close()

    tqdm.write('测试集MAE: ', end=' ')
    tqdm.write('test %.5f' % (test_loss[-1]))
