import pandas as pd
import talib as ta
import numpy as np 
import math
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from time import sleep


np.set_printoptions(suppress=True,threshold = np.inf)

def HMA(close,period):
    hma = ta.WMA((2*ta.WMA(close,period)-ta.WMA(close,period)),math.sqrt(period))
    return hma

def PPO(close, fastperiod, slowperiod, period):
    ppo = ((ta.EMA(close, fastperiod) - ta.EMA(close, slowperiod))/ta.EMA(close, slowperiod))*100
    ppo_histogram = ppo - ta.EMA(ppo,period)
    return ppo_histogram

row_df = pd.read_csv('dataset/2330.csv')
row_df.drop(['證券代碼','簡稱'],axis=1,inplace=True)
row_df.set_index('年月日',inplace=True)


row_df.index = pd.to_datetime(row_df.index,format = '%Y%m%d')
first = row_df.index.get_loc('2019-01-02')
end = row_df.index.get_loc('2020-01-02')

df = row_df[first - 57:]
count = len(row_df[first:end])


arr_2d = []
ta_arr = []
progress = tqdm(total=count)

for i in range(count):
    df = row_df[(first+i) - 57:]
    for j in range(6,21):

        sma = ta.SMA(df['收盤價(元)'][58-j:],j)[j-1]
        wma = ta.WMA(df['收盤價(元)'][58-j:],j)[j-1]
        macd = ta.MACD(df['收盤價(元)'][33-j:],fastperiod=12, slowperiod=26, signalperiod=j)[0][24+j]
        rsi = ta.RSI(df['收盤價(元)'][57-j:],j)[j]
        willr = ta.WILLR(df['最高價(元)'][58-j:],df['最低價(元)'][58-j:],df['收盤價(元)'][58-j:],j)[j-1]
        cci = ta.CCI(df['最高價(元)'][58-j:],df['最低價(元)'][58-j:],df['收盤價(元)'][58-j:],j)[j-1]
        roc = ta.ROC(df['收盤價(元)'][57-j:],j)[j] 
        ema = ta.EMA(df['收盤價(元)'][58-j:],j)[j-1]
        if j <= 8:
            hma = HMA(df['收盤價(元)'][57-j:],j)[j]
        elif j>8 and j<=15:    
            hma = HMA(df['收盤價(元)'][56-j:],j)[j+1]
        else:
            hma = HMA(df['收盤價(元)'][55-j:],j)[j+2]

        tema = ta.TEMA(df['收盤價(元)'][57 - (j*2 + (j-3)):],j)[(j*2 + (j-3))]
        cmo = ta.CMO(df['收盤價(元)'][57-j:],j)[j]
        ppo = PPO(df['收盤價(元)'][33-j:], fastperiod=12, slowperiod=26, period=j)[24+j]
        cmfi = ta.MFI(df['最高價(元)'][57-j:],df['最低價(元)'][57-j:],df['收盤價(元)'][57-j:],df['成交量(千股)'][57-j:],j)[j]
        dmi = ta.DX(df['最高價(元)'][57-j:],df['最低價(元)'][57-j:],df['收盤價(元)'][57-j:],j)[j]
        sar = ta.SAR(df['最高價(元)'][56:],df['最低價(元)'][56:])[1]
        
        # print(j)
        # print(ema[:30])
        # print(ema[j-1])



        arr = np.array([sma,wma,macd,rsi,willr,cci,roc,ema,hma,tema,cmo,ppo,cmfi,dmi,sar])
        arr_2d.append(arr)

    arr_2d = np.array(arr_2d) 

    #建立MinMaxScaler物件
    minmax = MinMaxScaler(feature_range = (1,255))
    # 資料標準化
    data_minmax = minmax.fit_transform(arr_2d)
    arr_2d = []

    ta_arr.append(data_minmax)
    progress.update(1)

# print(ta_arr[10])
# print(len(ta_arr))


first = row_df.index.get_loc('2019-01-02')
end = row_df.index.get_loc('2020-01-02')
df = row_df[first :end+1]
# df['Label'] = df['收盤價(元)'].diff()
df.loc[:,'Label'] = df['收盤價(元)'].diff()
df.Label[df['Label'] >= 0] = 1
df.Label[df['Label'] <= 0] = 0
print(len(ta_arr))
print(len(df.Label[1:]))

# print(type(df.Label))

np.savetxt('TA_data.txt',np.array(ta_arr).reshape(-1, 1))
np.savetxt('TA_label.txt',np.array(df.Label[1:]))