import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC,NuSVC,SVR
import pandas as pd
from tqdm import tqdm
import subprocess
import torch
import torch.nn as nn
import math
from torch.nn import Linear, ReLU, CrossEntropyLoss, Conv2d, MaxPool2d, Module, MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader,Dataset
from sklearn.metrics import accuracy_score

def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True

setup_seed(2550)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 自注意力模型
class SelfAttention(nn.Module):
    def __init__(self, input_dim, num_heads):
        super(SelfAttention, self).__init__()
        self.attention = nn.MultiheadAttention(input_dim, num_heads)

    def forward(self, x):
        x = x.unsqueeze(0).to(device)  # 添加额外的维度，变为 (1, 1, 60)
        x = x.permute(1, 0, 2).to(device)  # 调整维度顺序，变为 (1, 1, 60)
        attention_output, _ = self.attention(x, x, x)
        attention_output = attention_output.squeeze(0)  # 移除额外的维度，变为 (1, 60)
        return attention_output

class CustomDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        return self.data[index], self.label[index]

script_ta = "pipline_ta.py"
script_word = "pipline_word.py"
subprocess.run(["python",script_ta],shell=True)
subprocess.run(["python",script_word],shell=True)

word_vec = pd.read_csv('./dataset/BiLSTM_word.txt',header=None)
ta_vec = pd.read_csv('./dataset/BiLSTM_TA.txt',header=None)
full_vec = pd.concat([ta_vec,word_vec],axis=1)
full_vec = full_vec.applymap(lambda x:str(x).strip('[]'))
full_vec = np.float32(full_vec)
label = pd.read_csv('TA_label.txt',header=None)

# print(label.value_counts())

label = np.ravel(label.iloc[5:len(label)-1,:])
full_vec = np.float32(full_vec)
data = torch.tensor(full_vec).to(device)
label = torch.from_numpy(label).to(device)
# print(data.size())

# 创建数据加载器
batch_size = 1
shuffle = False
num_workers = 4

# 创建数据集实例
dataset = CustomDataset(data,label)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


attention_model = SelfAttention(input_dim=20, num_heads=4).to(device)

# 遍历数据加载器
output_list=[]
print('Attention Training')
for batch_inputs, batch_labels in tqdm(dataloader):

    attention_output = attention_model(batch_inputs)
    output_array = attention_output.detach().cpu().numpy()
    output_list.append(output_array[0].tolist())

x = pd.DataFrame(output_list)
y = label.cpu().numpy()

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.6, random_state=10)

# print(y_test)

clf = SVC(kernel='sigmoid',gamma = 'auto')
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
# print(y_pred)
score = accuracy_score(y_pred,y_test)
print("The score is : %f"%score)

score = clf.score(x,y)
print("The score of is : %f"%score)


