#这是一个ANN简单案例，基于手机的二十个特征预测手机价格区间
#步骤一般是，准备数据集，构建模型，模型训练，模型预测
#numpy->torch.Tensor->TensorDataset->DataLoader

import numpy as np
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import pandas as pd
import time


#导入数据集，这个可以在网上随便搜个数据集。我用的数据集是2000个样本，每个样本有20个特征，最后一列是价格区间
def create_dataset():
    train_data = pd.read_csv("./data/train.csv")                                                       #由于我下的数据集，他给的测试集是不带答案的，我就只能从训练集中取一部分当测试集了
    x,y=train_data.iloc[:,:-1],train_data.iloc[:,-1]                                                   #iloc是一个提取数据的方式，第一个代表从开头取到倒数第二列，第二个代表取最后一列
    x=x.astype(np.float32)
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=7,stratify=y) #stratify=y是为了让训练集和测试集中各个标签的比例和原样本一致
    x_train=torch.tensor(x_train.values)
    x_test=torch.tensor(x_test.values)
    y_train=torch.tensor(y_train.values)
    y_test=torch.tensor(y_test.values)
    train_dataset=TensorDataset(x_train,y_train)
    test_dataset=TensorDataset(x_test,y_test)
    return train_dataset,test_dataset,x_train.shape[1],len(np.unique(y_train))                         #x_train.shape[1]是提取样本有几个特征，len(np.unique(y_train))是检查y有几个不同的值（标签），这样可以应对不同的数据集

#构建模型
class PhonePriceModel(nn.Module):
    #初始化
    def __init__(self,input_dim,output_dim):
        super().__init__()
        #全连接层具体有几层，是根据结果来定的，比如说先试试3层，效果好就定下来了，不好就试试2层或者4层
        self.linear1=nn.Linear(input_dim,64)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(64,32)
        self.relu2 = nn.ReLU()
        self.output=nn.Linear(32,output_dim)
    def forward(self,x):
        x=self.linear1(x)
        x=self.relu1(x)
        x=self.linear2(x)
        x=self.relu2(x)
        x=self.output(x)
        return x

def train(train_dataset,input_dim,output_dim):
    #加载训练数据，分批训练
    train_dataloader=DataLoader(train_dataset,batch_size=16,shuffle=True)
    model = PhonePriceModel(input_dim,output_dim)
    criterion = nn.CrossEntropyLoss()  #分类用交叉熵，回归用MSE
    optimizer = optim.Adam(model.parameters(),lr=0.01)
    epochs = 100
    for epoch in range(epochs):
        model.train()
        total_loss,batch_num=0.0,0
        start = time.time()
        for x,y in train_dataloader:
            y_pred = model(x)
            loss = criterion(y_pred,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batch_num += 1
        print(f'Epoch {epoch+1}: Loss {total_loss/batch_num:.4f},time:{time.time()-start:.2f}')
    torch.save(model.state_dict(),"./data/PhoneModel.pth")

def test(test_dataset,input_dim,output_dim):
    model=PhonePriceModel(input_dim,output_dim)
    model.load_state_dict(torch.load("./data/PhoneModel.pth"))
    test_dataloader=DataLoader(test_dataset,batch_size=16,shuffle=False)
    correct=0
    model.eval()
    for x,y in test_dataloader:
        y_pred=torch.argmax(model(x),dim=1)
        correct += (y_pred == y).sum().item()
    print(f'准确率：{correct/len(test_dataset)}')
if __name__ == "__main__":
    train_dataset,test_dataset,input_dim,output_dim=create_dataset()
    model=PhonePriceModel(input_dim,output_dim)
    mode=int(input())
    if mode==1:
        train(train_dataset,input_dim,output_dim)
    elif mode==2:
        test(test_dataset,input_dim,output_dim)
