import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
import time
from scipy import io

class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden1, n_hidden2, n_output):
        super(Net, self).__init__()
        self.hidden1 = torch.nn.Linear(n_feature, n_hidden1)
        self.hidden2 = torch.nn.Linear(n_hidden1, n_hidden2)
        self.out = torch.nn.Linear(n_hidden2, n_output)

    def forward(self, x):
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = torch.sigmoid(x)
        x = self.out(x)
        return x

def get_kfold_data(k, i, X, y):
    #split the data
    #get the training set and validaion set of each fold

    fold_size = X.shape[0] // k
    val_start = i * fold_size
    if i != k - 1:
        val_end = (i + 1) * fold_size
        X_valid, y_valid = X[val_start:val_end], y[val_start:val_end]
        X_train = torch.cat((X[0:val_start], X[val_end:]), dim=0)
        y_train = torch.cat((y[0:val_start], y[val_end:]), dim=0)
    else:
        #put more data in the last fold if not divisible
        X_valid, y_valid = X[val_start:], y[val_start:]
        X_train = X[0:val_start]
        y_train = y[0:val_start]

    return X_train, y_train, X_valid, y_valid


def train_fold(model, X_train, y_train, X_val, y_val, BATCH_SIZE, learning_rate, EPOCHS):

    #package the data
    train_loader = DataLoader(TensorDataset(X_train, y_train), BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), BATCH_SIZE, shuffle=True)
    #loss function
    criterion = nn.CrossEntropyLoss()
    #optimizer
    optimizer = torch.optim.SGD(params=model.parameters(), lr=learning_rate)

    losses = []
    val_losses = []
    train_acc = []
    val_acc = []

    for epoch in range(EPOCHS):
        correct = 0
        for i, (features, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(features)

            #calculate the loss function
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

            #calculate the accuracy of each epoch in the training set
            y_pred = model(features)
            pred = torch.max(y_pred, 1)[1]
            correct += pred.eq(labels.view_as(pred)).sum().item()
        accuracy = 100. * correct / len(X_train)
        train_acc.append(accuracy)

        #calculate the accuracy of each epoch in the validation set
        val_loss = 0
        correct = 0

        with torch.no_grad():
            for i, (features, labels) in enumerate(val_loader):
                optimizer.zero_grad()
                y_pred = model(features)
                #calculate batch average loss
                loss = criterion(y_pred, labels).item()
                #sum up batch loss
                val_loss += loss * len(labels)
                pred = torch.max(y_pred, 1)[1]
                correct += pred.eq(labels.view_as(pred)).sum().item()
        val_losses.append(val_loss / len(X_val))
        accuracy = 100. * correct / len(X_val)
        val_acc.append(accuracy)

    return losses, val_losses, train_acc, val_acc


def k_fold(Net, k, X_train, y_train, num_epochs=50, learning_rate=0.005, batch_size=20):

    train_loss_sum, valid_loss_sum = 0, 0
    train_acc_sum, valid_acc_sum = 0, 0

    for i in range(k):
        #get the spilt data
        data = get_kfold_data(k, i, X_train, y_train)
        #My neural network
        net = Net
        #train each part of the data
        train_loss, val_loss, train_acc, val_acc = train_fold(net, *data, num_epochs, learning_rate, batch_size)

        train_loss_sum += train_loss[-1]
        valid_loss_sum += val_loss[-1]
        train_acc_sum += train_acc[-1]
        valid_acc_sum += val_acc[-1]
    print('average train loss:{:.4f}, average train accuracy:{:.3f}%'.format(train_loss_sum / k, train_acc_sum / k))
    print('average valid loss:{:.4f}, average valid accuracy:{:.3f}%'.format(valid_loss_sum / k, valid_acc_sum / k))

    return

if __name__ == '__main__':

    #load and process the data
    mat1 = io.loadmat('train_images.mat')  # train_images 10000*784
    mat2 = io.loadmat('train_labels.mat')  # train_labels 1*10000
    mat3 = io.loadmat('test_images.mat')
    mat4 = io.loadmat('test_labels.mat')

    images = mat1['train_images']
    labels = np.transpose(mat2['train_labels'])
    test_images = mat3['test_images']
    test_labels = mat4['test_labels']
    test_labels[0]
    labels1 = []
    labels2 = test_labels[0]
    for i in range(10000):
        labels1.append(labels[i][0])
    input = torch.FloatTensor(images)
    label = torch.LongTensor(labels1)
    test_input = torch.FloatTensor(test_images)
    test_label = torch.LongTensor(labels2)

    MyNet = Net(n_feature=784, n_hidden1=75, n_hidden2=20, n_output=10) #You can change the parameters here!

    #10-fold cross validation to choose the best hidden units
    #k_fold(MyNet,10,input,label)

    #Testing
    start = time.time()
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(MyNet.parameters(), lr=0.005)
    
    for i in range(1000):
        out = MyNet(test_input)
        loss = loss_func(out, test_label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    out = MyNet(test_input)
    prediction = torch.max(out, 1)[1]
    pred_y = prediction.numpy()
    target_y = test_label.data.numpy()
    print(accuracy_score(target_y, pred_y))
    end = time.time()
    running_time = end - start
    print('time cost : %.5f sec' % running_time)


