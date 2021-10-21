# coding:utf-8
from fastprogress import master_bar, progress_bar
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

output1 = open("/home/gonken2019/Desktop/subProject/output/class_list.txt","w")
output2 = open("/home/gonken2019/Desktop/subProject/output/class_predictts.txt","w")

path_claas_list="/home/gonken2019/Desktop/subProject/output/class_list.txt"
path_claas_predicts="/home/gonken2019/Desktop/subProject/output/class_predictts.txt"

#plt.rcParams['font.family'] ='sans-serif'#使用するフォント
#plt.rcParams['xtick.direction'] = 'in'#x軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
#plt.rcParams['ytick.direction'] = 'in'#y軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
#plt.rcParams['xtick.major.width'] = 2.0#x軸主目盛り線の線幅
#plt.rcParams['ytick.major.width'] = 1.0#y軸主目盛り線の線幅
#plt.rcParams['font.size'] = 8 #フォントの大きさ
#plt.rcParams['axes.linewidth'] = 1.0# 軸の線幅edge linewidth。囲みの太さ

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
        #付随する関数名でoptunaみたいに自動でやってくれる。
        #    def __init__(self, model, dataloaders, optimizer, scheduler, device, task):
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
            # ここが1 epochの肝
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
                
                np.set_printoptions(threshold=np.inf)
                
                print(class_list.cpu().numpy(), file=output1)
                #output.close()
                
                print(class_predicts.cpu().numpy(), file=output2)
                
                total_acc += acc
                #   self.scheduler.step()
                # ここまでfor文でぐるぐる回す。

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
        # ここまでが1 epoch

    def save(self, out_dir="./output"):
        model_state_dict = self.model.state_dict() #モデルの状態が保存される。重みとかかね

        checkpoint = {
            "model": model_state_dict,
            "epoch": self.epoch,
        }

        if self.task == "Classification":
            model_name = "pose_acc_{acc:3.3f}.chkpt".format(
                acc = self.val_acc_list[-1] #val_acc_listの最後尾の値が{acc:3.3f}に入る。
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
