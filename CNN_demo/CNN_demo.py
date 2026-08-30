#这是一个CNN简单案例，基于CIFA10数据集
#步骤一般是，准备数据集，搭建神经网络，模型训练，模型预测
#卷积层：提取图像局部特征。池化层：降维。
import torch.optim as optim
import torch
import torch.nn as nn
from torchvision.transforms import ToTensor
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import time
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"正在使用设备: {device}")

#准备数据集
def create_dataset():
    #torchvision.datasets是PyTorch官方提供的一个数据加载工具箱,里面现成的常见数据集
    train_dataset = CIFAR10(
        root='./data',
        train=True, #True是训练集，False是测试集
        download=True, #如果本来就有，就算是True，也不会再下一次
        transform=ToTensor() #把数据集转成张量
    )
    test_dataset = CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=ToTensor()

    )
    return train_dataset, test_dataset

#构建模型
class ImageClassification(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2, 2)  # 32x32 -> 16x16
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2, 2)  # 16x16 -> 8x8
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.linear1 = nn.Linear(64 * 4 * 4, 128)
        self.linear2 = nn.Linear(128, 64)
        self.out = nn.Linear(64, 10)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu(self.bn3(self.conv3(x))))
        #全连接层只能处理一维数据，所以要把x展平
        x = x.reshape(x.shape[0], -1)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.out(x)
        return x

def train(train_dataset):
    data_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    model = ImageClassification()
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    model.train()
    epochs = 15
    for epoch in range(epochs):
        total_loss,total_samples,total_correct,start = 0,0,0,time.time()
        for x,y in data_loader:
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            loss = criterion(y_pred, y)
            #梯度清零->反向传播->梯度更新
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_correct += (y_pred.argmax(dim=1) == y).sum().item()
            total_loss += loss.item()*len(y)
            total_samples += len(y)
        #每次都要算轮数，损失，准确率，时间
        print(f'Epoch:{epoch+1},Loss:{total_loss},Acc:{total_correct/total_samples:.2f},Time:{time.time()-start}')
    torch.save(model.state_dict(),'data/CNN_demo.pth')


def evaluate(test_dataset):
    data_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    model = ImageClassification()
    model = model.to(device)
    model.load_state_dict(torch.load('data/CNN_demo.pth', map_location=device))
    total_correct, total_samples = 0,0
    model.eval()
    for x,y in data_loader:
        x, y = x.to(device), y.to(device)
        y_pred=torch.argmax(model(x),dim=1)
        total_correct += (y_pred == y).sum().item()
        total_samples += len(y)
    print(f'Accuracy:{total_correct/total_samples:.2f}')


if __name__ == '__main__':
    train_dataset, test_dataset = create_dataset()
    train(train_dataset)
    evaluate(test_dataset)