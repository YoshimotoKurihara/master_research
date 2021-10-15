# coding:utf-8
import os
import numpy as np
from natsort import natsorted
from PIL import Image

import torch
import torch.utils.data
from torchvision import transforms

class Dataloaders:
    def __init__(self, images_path, labels_path, batch_size):
        self.images=os.listdir(images_path)
        for i in range(len(self.images)):
            self.images[i] = os.path.join(images_path,self.images[i])
        self.images = natsorted(self.images)    

        self.labels=os.listdir(labels_path)
        self.labels = natsorted(self.labels)
        for i in range(len(self.labels)):
            self.labels[i] = os.path.join(labels_path,self.labels[i])

        self.train_test_ratio = 0.8
        self.batch_size = batch_size

        self.prepare_dataloaders()

    def prepare_dataloaders(self):
        data_size = len(self.images)
        data_ids = np.arange(data_size)#
        np.random.seed(0)# shuffleを固定する為に用いる
        """numpy.random.seed(seed=シードに用いる値) をシード (種) を指定することで、
        発生する乱数をあらかじめ固定することが可能です。乱数を用いる分析や処理で、再現性が必要な場合などに用いられます。
        seed番号で一度作った乱数を何度でも利用できる"""
        np.random.shuffle(data_ids)#
        train_ids, test_ids = data_ids[:int(data_size * self.train_test_ratio)], data_ids[int(data_size * self.train_test_ratio):]#

        torch.manual_seed(0)
        """
        torch.seed（）[SOURCE]
        乱数を生成するためのシードを非決定的な乱数に設定します。 RNGのシードに使用される64ビットの数値を返します。
​
        torch.manual_seed（seed）[SOURCE]
        乱数を生成するためのシードを設定します。 torch.Generatorオブジェクトを返します。
        """
        self.train = torch.utils.data.DataLoader(
            MyDataset(self.images, self.labels, train_ids, transforms.ToTensor()),#
            batch_size = self.batch_size,
            shuffle=True,
            num_workers=2
        )

        self.test = torch.utils.data.DataLoader(
            MyDataset(self.images, self.labels, test_ids, transforms.ToTensor()),#
            batch_size = self.batch_size,
            shuffle=False,
            num_workers=2
        )

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, ids, transform = None):# transormは前処理flag
        self.images = images
        self.labels = labels
        self.ids = ids
        self.transform = transform#
    
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        index = self.ids[idx]
        image = self.images[index] 
        
        # 画像ファイルパスから画像を読み込みます。
        with open(image, 'rb') as f:
            image = Image.open(f)
            image = image.convert('RGB')
            #image=torch.FloatTensor(image)

        # 前処理がある場合は前処理をいれます。
        if self.transform is not None:
            image = self.transform(image)# 前処理入れたければ自分で定義する
            #image=torch.FloatTensor(image) #torch.from_numpy(image)
        image=torch.FloatTensor(image)

        label = self.labels[index]
        with open(label, 'r') as text:#
            data = []#
            for t in text.readlines():#
                data.append(float(t.strip()))#
        data=np.array(data).reshape(9,10)
        classes, positions = np.hsplit(data, [4])
        classes = np.argmax(classes, axis=1)
        classes = torch.tensor(classes)
        positions = torch.tensor(positions,dtype=torch.float32)

        output = {
            "image": image,
            "pos": positions,
            "class": classes,
        }

        return output