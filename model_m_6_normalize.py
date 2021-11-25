# coding:utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F

# やること
# 1. ネットワークを一つにする。(出力形式を変更する。)
# 2. 出力を標準化する。
# 標準化は「平均を0，分散を1とするスケーリング手法」で、正規化は「最小値を0，最大値を1とする0-1スケーリング手法」です。
# Decoderを変更するだけで良い。

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
            # nn.Linear(4096, 6)  #nn.Linear(4096, 座標数)なんで1物体だけなんだ？ #o
            nn.Linear(4096, 13)  #nn.Linear(4096, 座標数)なんで1物体だけなんだ？ #m 6+7=13
            #ここが最後の出力層　ここのクラス分類を連続値にして90個出力

        )
    def forward(self, x):
        y = self.decode_layers(x)
        return y

# 物体数が9だから、9になっている。今回は18にしないといけない。
class PoseNet(nn.Module):
    def __init__(self, hidden_size=1024):#
        super(PoseNet, self).__init__()
        #これはpython2の書き方。python3では省略できる
        #ここに入れてるAlexNetってのは自分の名前
        
        self.features = Encoder()
        size_check = torch.FloatTensor(256, 3, 256, 256) 
        
        fc_size = self.features(size_check).view(size_check.size(0), -1).size()[1]#
        #view(size_check.size(0), -1) ここで入力の順番を逆転させてるから実際の入力も逆転させないといけない？
        # self.linear=nn.Linear(fc_size, hidden_size * 9) #o
        self.linear=nn.Linear(fc_size, hidden_size * 18) #m
        self.Decoder = Decoder(hidden_size)
        self.hidden_size=hidden_size

    def forward(self, x):
        x = self.features(x)
        # x: (batch_size, rgb, height, width)
        x = x.view(x.size(0), -1) # (bs, fc_size)
        x = self.linear(x) # (bs, 9 * hiddensize)
        # x = x.view(x.size(0)*9,self.hidden_size) # ここも変える必要がある？ #o
        x = x.view(x.size(0)*18,self.hidden_size) # ここも変える必要がある？ #m
        class_and_poses = self.Decoder(x)
        #classifierは物体名の出力機　回帰でも使う
        return class_and_poses