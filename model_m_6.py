# coding:utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module): # EncoderはClassNetもPositionNetも共通。Decoderが違うだけ
    def __init__(self):
        super().__init__()
        #python3ではsuper(サブクラス名, self)の引数は省略できる  
        self.features = nn.Sequential(
            
            #RuntimeError: size mismatch, m1: [8 x 12544], m2: [9216 x 4096] 
            # at /pytorch/aten/src/TH/generic/THTensorMath.cpp:197
        
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            #第一引数が入力チャネル　第二引数が出力チャネル数　縦かける横は自分で計算しないといけない
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

        )
            
    def forward(self, x):
        y = self.features(x)
        return y

class Decoder(nn.Module):  #座標の出力機
    def __init__(self, fc_size):
        super().__init__()
        #fc_sizeで計算した形状を指定
        self.decode_layers = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(fc_size, 4096), 
            #Linearが恒等関数
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 6)  #nn.Linear(4096, 座標数)
            #ここが最後の出力層　ここのクラス分類を連続値にして90個出力

        )
    def forward(self, x):
        y = self.decode_layers(x)
        return y

class Classifier(nn.Module):  #物体名の出力機
    def __init__(self, fc_size):
        super().__init__()
        # 単なる変数の初期化ブロック
        self.dropout1 = nn.Dropout(p=0.5)
        self.linear1 = nn.Linear(fc_size, 4096)
        nn.init.kaiming_normal_(self.linear1.weight, nonlinearity = "relu")
        nn.init.zeros_(self.linear1.bias)
        self.relu1 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(4096, 4096)
        nn.init.kaiming_normal_(self.linear2.weight, nonlinearity = "relu")
        nn.init.zeros_(self.linear2.bias)
        self.relu2 = nn.ReLU()
        # self.linear3 = nn.Linear(4096, 4) #o
        self.linear3 = nn.Linear(4096, 7) #m
        nn.init.zeros_(self.linear3.bias)

    def forward(self, x):
        #勉強のためにDecoder関数とは違う方法で定義してみた
        y = self.relu1(self.linear1(self.dropout1(x)))
        y = self.relu2(self.linear2(self.dropout2(y)))
        #y = self.relu1(self.linear1(x))
        #y = self.relu2(self.linear2(y))
        y = self.linear3(y)
        y=F.log_softmax(y) # import torch.nn.functional as F
        # 出力層はlog_softmax
        return y

class ClassNet(nn.Module):
    def __init__(self, hidden_size=1024):#
        super(AlexNet, self).__init__()
        #これはpython2の書き方。python3では省略できる
        #ここに入れてるAlexNetってのは自分の名前
        
        self.features = Encoder()
        size_check = torch.FloatTensor(256, 3, 256, 256) 
        
        fc_size = self.features(size_check).view(size_check.size(0), -1).size()[1]#
        #view(size_check.size(0), -1) ここで入力の順番を逆転させてるから実際の入力も逆転させないといけない？
        self.linear=nn.Linear(fc_size, hidden_size * 9)
        self.classifier = Classifier(hidden_size)
        self.hidden_size=hidden_size

    def forward(self, x):
        x = self.features(x)
        # x: (batch_size, rgb, height, width)
        x = x.view(x.size(0), -1) # (bs, fc_size)
        x = self.linear(x) # (bs, 9 * hiddensize)
        x = x.view(x.size(0)*9,self.hidden_size) # ここも変える必要がある？
        classes = self.classifier(x)
        #classifierは物体名の出力機　回帰でも使う
        return classes

class PositionNet(nn.Module):
    def __init__(self, hidden_size=1024):#
        super(PositionNet, self).__init__()
        
        self.features = Encoder()#
        size_check = torch.FloatTensor(256, 3, 256, 256) 
        
        fc_size = self.features(size_check).view(size_check.size(0), -1).size()[1]#
        #view(size_check.size(0), -1) ここで入力の順番を逆転させてるから実際の入力も逆転させないといけない？
        self.linear=nn.Linear(fc_size, hidden_size * 9)
        nn.init.kaiming_normal_(self.linear.weight, nonlinearity = "relu")
        nn.init.zeros_(self.linear.bias)
        self.Decoder = Decoder(hidden_size)

        self.hidden_size=hidden_size

    def forward(self, x):
        x = self.features(x)
        # x: (batch_size, rgb, height, width)
        x = x.view(x.size(0), -1) # (bs, fc_size)
        x = self.linear(x) # (bs, 9 * hiddensize)
        x = x.view(x.size(0)*9,self.hidden_size)
        positions = self.Decoder(x)
        #Decoderは座標の出力機　回帰でも使う
        return positions