import time
import datetime
import numpy as np
import torch
import torch.nn.functional as F
from process.sample_graph import SampleGraph
import sys
import os
from process.get_dataloader import get_dataloader
from utils.callbacks import LossHistory
from tqdm import tqdm
from model.qgnn import QGNN
from torchmetrics import MeanAbsoluteError


# 准备训练
epochs = 10
cuda = torch.cuda.is_available()

molecular_property = 'zpve'

# Set how many epochs to save the weights once
save_period = 500
# Folders where permissions and log files are kept
save_dir = 'logs'
time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
log_dir = os.path.join(save_dir, "loss_" + str(time_str))
loss_history = LossHistory(log_dir)

model = QGNN()

device = torch.device('cuda' if cuda else 'cpu')


model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-16)
# Adjustment of learning rate, cyclical variation
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, verbose=False)

print('> start training')

train_loader, val_loader, test_loader, _ = get_dataloader(num_workers=0)
tr_ys = train_loader.dataset.data[molecular_property]

train_loss = []
val_loss = []
test_loss = []

# Save directory for model checkpoints
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

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

        # reset
        optimizer.zero_grad()
        # backward propagation
        loss.backward()
        # update
        optimizer.step()
        # Sequence of losses for each graph
        graph_train_loss += [float(loss.data.cpu().numpy())]

        # Update training information, latest loss value
        loop_train.set_postfix(loss='{:.3f}'.format(graph_train_loss[-1]))

    train_loss += [np.mean(graph_train_loss)]
    loop_train.close()

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

    # Save the model weights (every `save_period` epochs)
    if (epoch + 1) % save_period == 0:
        torch.save(model.state_dict(),
                   os.path.join(save_dir, f'ep{epoch + 1:03d}-loss{train_loss[-1]:.3f}-val_loss{val_loss[-1]:.3f}.pth'))

    # Save the best model based on validation loss
    with torch.no_grad():
        model.eval()

        loop_val = tqdm(val_loader, file=sys.stdout, desc=f'Epoch [{epoch + 1}/{epochs}]' + 'val')
        for graph in loop_val:
            graph = SampleGraph(graph, molecular_property, cuda)
            graph.to(device)
            edges = graph.edges.cpu().numpy().reshape(-1)
            model_instance = QGNN(graph.num[-1].cpu().numpy(), edges).to(device)
            out = model_instance(graph).reshape(-1)
            loss = F.l1_loss(out, graph.y.to(device))
            graph_val_loss += [float(loss.data.cpu().numpy())]

            loop_val.set_postfix(loss='{:.3f}'.format(graph_val_loss[-1]))

        # Store validation loss
        val_loss += [np.mean(graph_val_loss)]

        loop_val.close()

        # Log the results
        tqdm.write(f'Epoch {epoch + 1}: train_loss = {train_loss[-1]:.3f}, val_loss = {val_loss[-1]:.3f}')

        # Save the best model based on validation loss
        if len(loss_history.val_loss) <= 1 or val_loss[-1] <= min(loss_history.val_loss):
            torch.save(model.state_dict(), os.path.join(save_dir, "best_epoch_weights.pth"))

    # Apply the learning rate scheduler
    lr_scheduler.step()

# Save final model after training ends
torch.save(model.state_dict(), os.path.join(save_dir, "final_model.pth"))


with torch.no_grad():
    model.eval()

    graph_test_loss = []

    loop_test = tqdm(test_loader, file=sys.stdout)
    for graph in loop_test:
        graph = SampleGraph(graph, molecular_property, cuda)
        edges = np.array(graph.edges).reshape(-1)
        model = QGNN(graph.num[-1].numpy(), edges)
        out = model(graph).reshape(-1)
        loss = F.l1_loss(out, graph.y)
        graph_test_loss += [float(loss.data.cpu().numpy())]

        # Updated with the latest loss values
        loop_test.set_postfix(loss='{:.3f}'.format(graph_test_loss[-1]))

    #  / 0.001
    test_loss += [np.mean(graph_test_loss)]

    loop_test.close()

    tqdm.write('TEST_MAE: ', end=' ')
    tqdm.write('test %.5f' % (test_loss[-1]))
