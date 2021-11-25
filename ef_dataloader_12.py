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
        np.random.shuffle(data_ids)#
        train_ids, test_ids = data_ids[:int(data_size * self.train_test_ratio)], data_ids[int(data_size * self.train_test_ratio):]#

        torch.manual_seed(0)
        """
        torch.seed（）[SOURCE]
        乱数を生成するためのシードを非決定的な乱数に設定します。 RNGのシードに使用される64ビットの数値を返します。
        torch.manual_seed（seed）[SOURCE]
        乱数を生成するためのシードを設定します。 torch.Generatorオブジェクトを返します。
        """
        self.train = torch.utils.data.DataLoader(
            MyDataset(self.images, self.labels, train_ids, transforms.ToTensor()),#
            batch_size = self.batch_size,
            shuffle=True,
            num_workers=1
        )
        # num_workers変えると速くなるか？

        self.test = torch.utils.data.DataLoader(
            MyDataset(self.images, self.labels, test_ids, transforms.ToTensor()),#
            batch_size = self.batch_size,
            shuffle=False,
            num_workers=1
        )

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, ids, transform = None):# transormは前処理flag ここに正規化処理 
        #transformに関数を入れる
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
            #MyDatasetの引数
            #image=torch.FloatTensor(image) #torch.from_numpy(image)
        image=torch.FloatTensor(image)

        label = self.labels[index]
        with open(label, 'r') as text:#
            data = []#
            for t in text.readlines():#
                data.append(float(t.strip()))#
        #data=np.array(data).reshape(9,10) #O
        # 9物体各10データずつだから9*10今回は18物体13データずつ。
        # 今回は物体の種類が増えたから、データセットも修正しないといけない！！！
        num_objects = 36
        len_row = 21
        data=np.array(data).reshape(num_objects,len_row) #m
        # classes, positions = np.hsplit(data, [4]) #前半4データを抽出 #o
        classes, positions = np.hsplit(data, [len_row - 6]) #m
        coordinates, angles = np.hsplit(positions,[3])
        coordinates_flat = coordinates.flatten()
        angles_flat = angles.flatten()
        max_minus_min_coordinate = 10 - (-10)
        max_minus_min_angle = 360 - 0

        # 標準化処理 result = (x-x_min)/(x_max-x_min)
        for i in range(len(coordinates_flat)):
            coordinates_flat[i] = (coordinates_flat[i]+10)/max_minus_min_coordinate
        for i in range(len(coordinates_flat)):
            angles_flat[i] = (angles_flat[i])/max_minus_min_angle

        coordinates = np.array(coordinates_flat).reshape(num_objects,3)#座標数は3だから3
        angles = np.array(angles_flat).reshape(num_objects,3)#角度数は3だから3
        positions = np.concatenate([coordinates,angles], 1)

        # どちら方向に結合すれば良いのかを考える。
        class_and_pose = np.concatenate([classes,positions], 1)
        class_and_pose = torch.tensor(class_and_pose,dtype=torch.float32) #ここを標準化して、その後結合。
        # 正規化(最小値が0最大値が1のデータに変換)の方がいいか？クラス分類が0-1だから。
        # いや、クラス分類は最大値を取るから平均0標準偏差1の方がいい。

        output = {
            "image": image,
            "class_and_pose": class_and_pose
        }

        return output