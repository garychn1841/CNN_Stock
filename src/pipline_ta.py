import torch
import torch.nn as nn
import numpy as np
import math
from sklearn.model_selection import train_test_split
from torch.nn import Linear, ReLU, CrossEntropyLoss, Conv2d, MaxPool2d, Module, MSELoss
from torch.optim import Adam
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader,Dataset

def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True

setup_seed(20)

####################### Data_function #######################
class CNNDataset(Dataset):

    # data loading
    def __init__(self,TA_data,y_data):
    
        self.x = torch.from_numpy(TA_data).to(device)
        self.y = torch.from_numpy(y_data).to(device)
        self.n_samples = len(TA_data)

    # working for indexing
    def __getitem__(self, index):
        
        return self.x[index], self.y[index]

    # return the length of our dataset
    def __len__(self):
        
        return self.n_samples


class LSTM_Dataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)
    
    
def nn_seq_us(B,data):
    dataset = data
    # split
    train = dataset[:math.ceil(len(dataset) * 0.6)]
    val = dataset[math.ceil(len(dataset) * 0.6):math.ceil(len(dataset) * 0.8)]
    test = dataset[math.ceil(len(dataset) * 0.8):len(dataset)]
    m, n = np.max(train[train.columns[1]]), np.min(train[train.columns[1]])

    def process(data, batch_size, shuffle):
        seq = []
        for i in range(len(data) - 5):
            train_seq = []
            train_label = []
            for j in range(i, i + 5):   
                x = data.iloc[j,:].to_list()
                train_seq.append(x)

            train_label.append(data.iloc[i + 5,:].to_list())
            train_seq = torch.FloatTensor(train_seq)
            train_label = torch.FloatTensor(train_label).view(-1)
            seq.append((train_seq, train_label))

        seq = LSTM_Dataset(seq)
        seq = DataLoader(dataset=seq, batch_size=batch_size, shuffle=shuffle, num_workers=0, drop_last=True)

        return seq


    Dtr = process(train, B, False)
    Val = process(val, B, False)
    Dte = process(test, B, False)
    Full = process(dataset, B, False)

    return Dtr, Val, Dte, Full
    
 


####################### Model_Struction #######################

class CNN_Model(Module):
    #列出需要哪些層
    def __init__(self):
        super(CNN_Model, self).__init__()
        # Convolution 1 , input_shape=(1,15,15)
        self.cnn1 = Conv2d(1, 16, kernel_size=3, stride=1,padding= 1) 
        self.relu1 = ReLU(inplace=True) 
        # Max pool 1 , input_shape=(4,13,13)
        self.maxpool1 = MaxPool2d(kernel_size=2,stride=2)
    
         
        # Fully connected 1 ,#input_shape=(4*6*6)
        self.fc = Linear(16 * 7 * 7,10)   
        self.fc2 = Linear(in_features=10, out_features=4)
        self.fc3 = Linear(in_features=4, out_features=2)

      

    #列出forward的路徑，將init列出的層代入
    def forward(self, x):
        x = self.cnn1(x) 
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = x.view(x.size(0), -1) 
        x = self.fc(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
    
class DevCNN_Model(Module):
    #列出需要哪些層
    def __init__(self):
        super(DevCNN_Model, self).__init__()
        # Convolution 1 , input_shape=(1,15,15)
        self.cnn1 = model.cnn1
        self.relu1 = model.relu1
        # Max pool 1 , input_shape=(4,13,13)
        self.maxpool1 =  model.maxpool1
        self.fc = model.fc

    #列出forward的路徑，將init列出的層代入
    def forward(self, x):
        x = self.cnn1(x) 
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = x.view(x.size(0), -1) 
        x = self.fc(x)
        return x


class BiLSTM(Module):
    def __init__(self,input_size, hidden_size, num_layers, output_size, batch_size):
        super(BiLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.num_directions = 2
        self.batch_size = batch_size
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(self.num_directions * self.hidden_size, self.output_size)

    def forward(self, input_seq):
        h_0 = torch.randn(self.num_directions * self.num_layers, self.batch_size, self.hidden_size).to(device)
        c_0 = torch.randn(self.num_directions * self.num_layers, self.batch_size, self.hidden_size).to(device)
        # print(input_seq.size())
        seq_len = input_seq.shape[1]
        # input(batch_size, seq_len, input_size)
        # output(batch_size, seq_len, num_directions * hidden_size)
        output, _ = self.lstm(input_seq, (h_0, c_0))
        pred = self.linear(output)  # pred()
        pred = pred[:, -1, :]
        
        return pred

####################### helper_function #######################
def BiLSTM_val(model, datalaoder, device):
    model.eval()
    loss_list = []

    for (seq, label) in datalaoder:
        seq = seq.to(device)
        label = label.to(device)
        output = model(seq)
        loss = loss_func(output, label)
        loss_list.append(loss.item())
    
    return np.mean(loss_list)
        

#check GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#load data
x = pd.read_csv('TA_data.txt',names=['TA'])
y = pd.read_csv('TA_label.txt',names=['label'])

x = np.float32(x['TA'].to_numpy())
x = np.reshape(x,(-1,1,15,15))
y = np.longlong(y['label'].to_numpy())


training_size = math.ceil(len(y)*0.7)
train_x = x[:training_size]
train_y = y[:training_size]
test_x = x[training_size:]
test_y = y[training_size:]


#Hyperparams
LR = 0.01
TRAIN_BATCH_SIZE=5
TEST_BATCH_SIZE=len(test_y)
DEV_BATCH_SIZE=len(x)
NUM_EPOCHS = 250
TICKER = '2330'


#training data
training_dataset = CNNDataset(train_x,train_y)
train_dataloader = DataLoader(
    training_dataset,
    shuffle=False,
    batch_size=TRAIN_BATCH_SIZE,
    drop_last=True
)

#testing data
testing_dataset = CNNDataset(test_x,test_y)
testing_dataloader = DataLoader(
    testing_dataset,
    shuffle=False,
    batch_size=TEST_BATCH_SIZE,
)

#developing data
DEV_dataset = CNNDataset(x,y)
dev_dataloader = DataLoader(
    DEV_dataset,
    shuffle=False,
    batch_size=DEV_BATCH_SIZE
)


#model training
model = CNN_Model()
model.to(device)
optimizer = Adam(model.parameters(), lr=LR)
loss_func = CrossEntropyLoss()
output_list = []

print("TA CNN Training")
for epoch in tqdm(range(NUM_EPOCHS)):

    model.train()
    for step , (x,y) in enumerate(train_dataloader):
        output = model(x)
        loss = loss_func(output , y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pred = output.data.max(dim = 1, keepdim = True)[1]

        if step % 50 == 0:
            model.eval()
            for t_x,t_y in testing_dataloader:
                output = model(t_x)
                loss = loss_func(output,t_y)
                pred = output.data.max(dim = 1, keepdim = True)[1].data.squeeze()
                acc = sum(pred == t_y)/t_y.size(0)
            # print('epoch:',epoch,'loss:',loss.item(),'acc:',acc.item())


new_model = DevCNN_Model()
new_model.eval()


with torch.no_grad():
    for epoch in range(1):

        for step , (x,y) in enumerate(dev_dataloader):
            new_model = DevCNN_Model()
            output = new_model(x)
            # print(output)
FILE = (f'model_{TICKER}.pt')
torch.save(new_model.state_dict(), FILE)


df_CNN = pd.DataFrame(output.cpu().numpy())
Dtr, Val, Dte, Full = nn_seq_us(2,df_CNN)

for i in Dtr:
    INPUT_SIZE = i[0].size()[2]

#Hyperparams
LR = 0.01
TRAIN_BATCH_SIZE=2
TEST_BATCH_SIZE=len(test_y)
DEV_BATCH_SIZE=len(x)
NUM_EPOCHS = 250
# NUM_EPOCHS = 100
INPUT_SIZE = INPUT_SIZE
OUTPUT_SIZE = INPUT_SIZE
HIDDEN_SIZE = 1
NUM_LAYERS = 1
TICKER = '2330'


# model BiLSTM Training
model = BiLSTM(input_size = INPUT_SIZE, 
               hidden_size = HIDDEN_SIZE,
               num_layers = NUM_LAYERS,
               output_size = OUTPUT_SIZE,
               batch_size=TRAIN_BATCH_SIZE).to(device)
optimizer = Adam(model.parameters(), lr=LR)
loss_func = MSELoss().to(device)

print("TA BiLSTM Training")
for epoch in tqdm(range(NUM_EPOCHS)):
    train_loss = []
    model.train()
    for (seq, label) in Dtr:
            seq = seq.to(device)
            label = label.to(device)
            output = model(seq)
            loss = loss_func(output, label)
            train_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print(output)
    if epoch % 50 == 0 and epoch > 1:
        loss = BiLSTM_val(model, Val, device)
        # print("epoch:",epoch,'loss:',loss)



model.eval()
with torch.no_grad():
    BiLSTM_output = []
    for (seq,label) in Full:
        seq = seq.to(device)
        output = model(seq)
        numpy_array = output.cpu().numpy()
        BiLSTM_output.append(numpy_array[0].tolist())
        BiLSTM_output.append(numpy_array[1].tolist())


file_path = "./dataset/BiLSTM_TA.txt"

with open(file_path, "w") as file:
    # 遍历列表中的每个元素并写入文件
    for item in BiLSTM_output:
        file.write(str(item) + "\n")