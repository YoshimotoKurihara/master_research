# coding:utf-8
from fastprogress import master_bar, progress_bar
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

output1 = open("/home/gonken2019/Desktop/subProject/output/class_list.txt","w")
output2 = open("/home/gonken2019/Desktop/subProject/output/class_predictts.txt","w")


#plt.rcParams['font.family'] ='sans-serif'#使用するフォント
#plt.rcParams['xtick.direction'] = 'in'#x軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
#plt.rcParams['ytick.direction'] = 'in'#y軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
#plt.rcParams['xtick.major.width'] = 2.0#x軸主目盛り線の線幅
#plt.rcParams['ytick.major.width'] = 1.0#y軸主目盛り線の線幅
#plt.rcParams['font.size'] = 8 #フォントの大きさ
#plt.rcParams['axes.linewidth'] = 1.0# 軸の線幅edge linewidth。囲みの太さ

#全部定義
"""class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self,yhat,y):
        return torch.sqrt(self.mse(yhat,y))
    """
class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps
        
    def forward(self,yhat,y):
        loss = torch.sqrt(self.mse(yhat,y) + self.eps)
        return loss

class MyTrainer:
    def __init__(self, model, dataloaders, optimizer, device, task):
    #       def __init__(self, model, dataloaders, optimizer, scheduler, device, task):
        self.model = model

        self.device = device
        self.model = self.model.to(device)
        self.task = task

        self.dataloaders = dataloaders
        self.optimizer = optimizer
        #       self.scheduler = scheduler
        self.rmse_loss = RMSELoss()
        #self.mse_loss = nn.MSELoss()
        self.nll_loss = nn.NLLLoss()

        self.epoch=0

        self.train_loss_list = []
        self.train_acc_list = []
        self.val_loss_list = []
        self.val_acc_list = []        

        self.counter=0

    def run(self, epoch_num):
        self.master_bar = master_bar(range(epoch_num))
        for epoch in self.master_bar:
            self.epoch += 1
            self.train()
            self.test()
            self.save()
            self.draw_graph()
            if self.task == "Classification":
                self.draw_acc()
        output1.close()
        output2.close()
    
    def train(self):
        self.iter(train=True)
    
    def test(self):
        self.iter(train=False)

    def iter(self, train):
        if train:
            self.model.train()
            dataloader = self.dataloaders.train
        else:
            self.model.eval()
            dataloader = self.dataloaders.test
        
        total_loss = 0.
        total_acc = 0.

        data_iter = progress_bar(dataloader, parent=self.master_bar)
        for i, batch in enumerate(data_iter):
            image_list = batch["image"].to(self.device)
            if self.task == "Classification":
                class_list = batch["class"].to(self.device)
                # forward
                classes = self.model(image_list)
                class_list= class_list.view(-1)

                #print(class_list.dtype)
                # calc loss
                """
                class_list = torch.Tensor(np.zeros_like(batch["class"])).to(self.device)
                class_list= class_list.view(-1).long()
                """
                loss = self.nll_loss(classes, class_list)

                """
                print(loss)

                class_list = torch.Tensor(np.zeros_like(batch["class"])).to(self.device)
                class_list= class_list.view(-1).long()
                class_list+=1
                loss = self.nll_loss(classes, class_list)
                print(loss)
                class_list = torch.Tensor(np.zeros_like(batch["class"])).to(self.device)
                class_list= class_list.view(-1).long()
                class_list+=2
                loss = self.nll_loss(classes, class_list)
                print(loss)
                class_list = torch.Tensor(np.zeros_like(batch["class"])).to(self.device)
                class_list= class_list.view(-1).long()
                class_list+=3
                loss = self.nll_loss(classes, class_list)
                print(loss)
                """

                """print(classes)
                print("************************************************************************************")
                print(class_list)"""
            if self.task == "Regression":
                pos_list = batch["pos"].to(self.device)
                positions = self.model(image_list)
                # classes = torch.view(-1, classes)
                pos_list=pos_list.view(-1,6)
                loss = self.rmse_loss(positions, pos_list)

            '''
            loss=CustomLossFunction()
            loss.forward(classes, positions, class_list, pos_list)
            '''

            # backward
            if train:
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad
            total_loss += loss.item()

            if self.task == "Classification":
                class_predicts=torch.argmax(classes,dim=1)
                acc = (class_predicts == class_list).sum().item() / len(class_list)
                """
                print(class_predicts)
                print(class_list)
                print((class_predicts == class_list).sum().item(), len(class_list))
                print(acc)
                self.counter+=1
                if self.counter > 1:
                    import sys
                    sys.exit()
                    """
                #calss_listもindexを返させないといけない？
                #calss_listはindexになってる
                #全要素を表示させたい_01_06
                #print("\n{}".format(class_predicts))
                #print(class_list)
                #print("************************************************************************************")
                
                path_claas_list="/home/gonken2019/Desktop/subProject/output/class_list.txt"
                path_claas_predicts="/home/gonken2019/Desktop/subProject/output/class_predictts.txt"
                
                np.set_printoptions(threshold=np.inf)
                
                print(class_list.cpu().numpy(), file=output1)
                #output.close()
                
                print(class_predicts.cpu().numpy(), file=output2)
                #output.close()
                #with open(path_w, mode='w') as f:
                    #f.write(class_list)

                """with open(path_claas_list, mode='w') as f:
                    f.write(class_list)#'\n'.join(class_list))
                #with open(path_claas_list) as f:
                    #print(f.read())
                #f.write('\n'.join(class_list))
                    #TypeError: sequence item 0: expected str instance, Tensor found


                with open(path_claas_predict, mode='w') as f:
                    f.write(class_predicts)#'\n'.join(class_predicts))
                #with open(path_claas_predict) as f:
                    #print(f.read())"""
                """writelines()では改行コードは挿入されず、要素がそのまま連結されて書き込まれる。
                リストの要素ごとに改行して書き込みたい場合は、改行コードとjoin()メソッドで改行込みの文字列を作成し、write()メソッドで書き込む。

                with open(path_w, mode='w') as f:
                    f.write('\n'.join(l))

                with open(path_w) as f:
                    print(f.read())"""
                """output=open（"/home/gonken2019/Desktop/subProject/output/outtput.txt","r"）
                print（class_list, file=output）
                output.close()"""
                """source: file_io_with_open.py
                上の例のようにopen()したファイルオブジェクトはclose()メソッドでクローズする必要がある。
                withブロックを使うとブロックの終了時に自動的にクローズされる。閉じ忘れがなくなるので便利。"""
                """output=open（パス,"r"）って宣言して、print（class_list, file=output）って指定するとファイルに出力してくれる"""

                total_acc += acc
        #       self.scheduler.step()

        if train:
            self.train_loss_list.append(total_loss / (i + 1))
            if self.task == "Classification":
                self.train_acc_list.append(total_acc / (i + 1))
        else:
            self.val_loss_list.append(total_loss / (i + 1))
            if self.task == "Classification":
                self.val_acc_list.append(total_acc / (i + 1))

        train = "train" if train else "test"
        print("[Info] epoch {}@{}: loss = {}".format(self.epoch, train, total_loss/ (i + 1)))
        if self.task == "Classification":
            print("[Info] epoch {}@{}: loss = {}, acc = {}".format(self.epoch, train, total_loss/ (i + 1), total_acc/(i + 1)))
    
    def save(self, out_dir="./output"):
        model_state_dict = self.model.state_dict()

        checkpoint = {
            "model": model_state_dict,
            "epoch": self.epoch,
        }

        if self.task == "Classification":
            model_name = "pose_acc_{acc:3.3f}.chkpt".format(
                acc = self.val_acc_list[-1]
            )
        if self.task == "Regression":
            model_name = "regression.chkpt"
        torch.save(checkpoint, model_name)
    
    def draw_graph(self):
        x = np.arange(self.epoch)
        y = np.array([self.train_loss_list, self.val_loss_list]).T
        plt.figure()
        plots = plt.plot(x, y)
        plt.legend(plots, ("train", "test"), loc="best", framealpha=0.2, prop={"size": "small", "family": "monospace"})
        plt.xlabel("Epoch")
        plt.xlim(xmin=0, xmax=20)
        #plt.ylim(ymin=0, ymax=1)
        plt.xticks([0,2,4,6,8,10,12,14,16,18,20])
        #plt.yticks([0.0,0.2,0.4,0.6,0.8,1.0])
        #plt.xticks(0, 20) 
        #plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        #plt.set_xticks([0,2,4,6,8,10,12,14,16,18,20])
        #plt.set_xticks(np.linspace(0, np.pi * 4, 5))

        #plt.tight_layer()
        graph_name = {
            "Classification" : "graph_loss.png",
            "Regression" : "graph_reg.png"
        }
        plt.savefig(graph_name[self.task])
        # plt.show()
    def draw_acc(self):
        x = np.arange(self.epoch)
        acc = np.array([self.train_acc_list, self.val_acc_list]).T
        plt.figure()
        plots_acc = plt.plot(x, acc)
        plt.legend(plots_acc, ("train", "test"), loc="best", framealpha=0.2, prop={"size": "small", "family": "monospace"})
        plt.xlabel("Epoch")
        plt.xlim(xmin=0, xmax=20)
        plt.ylim(ymin=0, ymax=1)
        plt.xticks([0,2,4,6,8,10,12,14,16,18,20])
        plt.yticks([0.0,0.2,0.4,0.6,0.8,1.0])
        #plt.xticks(0, 20)
        #plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        #plt.set_xticks(np.linspace(0, np.pi * 4, 5))
        #plt.set_xticks([0,2,4,6,8,10,12,14,16,18,20])
        plt.savefig("graph_acc.png")
