# coding:utf-8
from fastprogress import master_bar, progress_bar
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

fig_out_dir = "6_result/"
os.makedirs(fig_out_dir, exist_ok=True)

path_claas_list = fig_out_dir + "class_list.txt"
path_claas_predicts = fig_out_dir + "class_predictts.txt"

num_epoch = 30
output1 = open(path_claas_list,"w")
output2 = open(path_claas_predicts,"w")

class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps
        
    def forward(self,yhat,y):
        loss = torch.sqrt(self.mse(yhat,y) + self.eps)
        return loss

class MyTrainer:
    def __init__(self, model, dataloaders, optimizer, device):
        #付随する関数名でoptunaみたいに自動でやってくれる。
        #    def __init__(self, model, dataloaders, optimizer, scheduler, device, task):
        self.model = model

        self.device = device
        self.model = self.model.to(device)
        self.dataloaders = dataloaders
        self.optimizer = optimizer
        #       self.scheduler = scheduler
        self.rmse_loss = RMSELoss()
        #self.mse_loss = nn.MSELoss()
        # self.nll_loss = nn.NLLLoss()

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
            class_and_pose_list = batch["class_and_pose"].to(self.device)
            class_and_pose = self.model(image_list)
            class_and_pose_list = class_and_pose_list.view(-1,234)
            loss = self.rmse_loss(class_and_pose, class_and_pose_list)
            
            # backward
            if train:
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad
            total_loss += loss.item()

            # 現状testの際にaccが上手く計算されていない。
            # + memory overflow
            # testの際のaccが0.010であったことより、重みを読み込めていない？
            
            class_predicts=torch.argmax(class_and_pose[:,:7],dim=1)
            class_list = torch.argmax(class_and_pose_list[:,:7],dim=1)
            acc = (class_predicts == class_list).sum().item() / len(class_list)
            np.set_printoptions(threshold=np.inf)
            print("acc:",acc)
            print(class_and_pose_list[7].cpu().numpy(), file=output1)
            print(class_predicts[7].cpu().numpy(), file=output2)

            total_acc += acc
            print("total_acc_in_batch",total_acc)

        if train:
            self.train_loss_list.append(total_loss / (i + 1))
            self.train_acc_list.append(total_acc / (i + 1))
        else:
            print("total_acc",total_acc)
            self.val_loss_list.append(total_loss / (i + 1))
            self.val_acc_list.append(total_acc / (i + 1))
            print("total_acc",total_acc)

        train = "train" if train else "test"
        print("[Info] Class and pose : epoch {}@{}: loss = {}".format(self.epoch, train, total_loss/ (i + 1)))
        print("[Info] Class : epoch {}@{}: acc = {}".format(self.epoch, train, total_acc/(i + 1)))
        # ここまでが1 epoch

    def save(self, out_dir="./output"):
        model_state_dict = self.model.state_dict() #モデルの状態が保存される。重みとかかね

        checkpoint = {
            "model": model_state_dict,
            "epoch": self.epoch,
        }

        # model_name = "pose_acc_{acc:3.3f}.chkpt".format(
                # acc = self.val_acc_list[-1] #val_acc_listの最後尾の値が{acc:3.3f}に入る。
        # )
        model_name = "regression.chkpt"
        torch.save(checkpoint, fig_out_dir + model_name)
    
    def draw_graph(self):
        x = np.arange(self.epoch)
        y = np.array([self.train_loss_list, self.val_loss_list]).T
        plt.figure()
        plots = plt.plot(x, y)
        plt.legend(plots, ("train", "test"), loc="best", framealpha=0.2, prop={"size": "small", "family": "monospace"})
        plt.xlabel("Epoch")
        plt.xlim(xmin=0, xmax=num_epoch)
        #plt.ylim(ymin=0, ymax=1)
        # plt.xticks([range(0,51,2)])
        #plt.yticks([0.0,0.2,0.4,0.6,0.8,1.0])
        #plt.xticks(0, 20) 
        #plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        #plt.set_xticks([0,2,4,6,8,10,12,14,16,18,20])
        #plt.set_xticks(np.linspace(0, np.pi * 4, 5))

        #plt.tight_layer()
        plt.savefig(fig_out_dir + "graph_loss.png")
        # plt.show()
    def draw_acc(self):
        x = np.arange(self.epoch)
        acc = np.array([self.train_acc_list, self.val_acc_list]).T
        plt.figure()
        plots_acc = plt.plot(x, acc)
        plt.legend(plots_acc, ("train", "test"), loc="best", framealpha=0.2, prop={"size": "small", "family": "monospace"})
        plt.xlabel("Epoch")
        plt.xlim(xmin=0, xmax=num_epoch)
        plt.ylim(ymin=0, ymax=1)
        # plt.xticks(range(0,51,2))
        plt.yticks([0.0,0.2,0.4,0.6,0.8,1.0])
        #plt.xticks(0, 20)
        #plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        #plt.set_xticks(np.linspace(0, np.pi * 4, 5))
        #plt.set_xticks([0,2,4,6,8,10,12,14,16,18,20])
        plt.savefig(fig_out_dir + "graph_acc.png")
