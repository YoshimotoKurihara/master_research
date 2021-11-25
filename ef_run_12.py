# coding:utf-8
#export PYTHONPATH=/usr/local/lib/python3.6/dist-packages
#毎回これを実行
#インタープリタはpython3.7.4のbase condaのやつを使う

#データ正規化　バッチ正規化　学習率のスケジューラ
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import copy

#同ディレクトリ内の別のファイルからのクラスの読み込み
from ef_model_12 import effnetv2_m
from ef_dataloader_12 import Dataloaders
from ef_trainer_12 import MyTrainer
# import hparams
# import datasets
"""
MODEL = 'efficientnetv2-b0'  #@param
import effnetv2_model

with tf.compat.v1.Graph().as_default():
  model = effnetv2_model.EffNetV2Model(model_name=MODEL)
  _ = model(tf.ones([1, 224, 224, 3]), training=False)

ckpt_path = tf.train.latest_checkpoint(ckpt_path)
model.load_weights(ckpt_path)
"""

def main():
    dataset_dir = "data_12/"
    IMAGE_PATH = dataset_dir + "data"
    LABELS_PATH = dataset_dir + "label"

    # model_name='efficientnetv2-s'
    # dataset_cfg='Imagenet'
    # hparam_str= ''
    # sweeps= ''
    # use_tpu= False
    # tpu_job_name= None
    # # Cloud TPU Cluster Resolvers
    # tpu= None
    # gcp_project= None
    # tpu_zone= None
    # # Model specific flags
    # data_dir= None
    # eval_name= None
    # archive_ckpt= True 
    # model_dir= None
    # mode= 'train'
    # export_to_tpu= False


    BATCH_SIZE = 64 #こことsubmodel.py 85行目と113行目の最初の引数を変える #model.pyの96行目と119行目も変える必要がある。
    #BATCH_SIZE = 10
    #RuntimeError: size mismatch, m1: [10 x 12544], m2: [9216 x 4096] at /pytorch/aten/src/TH/generic/THTensorMath.cpp:197
    #9216×4096=37748736
    #37748736÷12544=3009.306122449
    #4096=2**12

    #BATCH_SIZE = 8
    #RuntimeError: size mismatch, m1: [8 x 12544], m2: [9216 x 4096] at /pytorch/aten/src/TH/generic/THTensorMath.cpp:197

    NUM_EPOCH = 30 #多くて20~30

    if torch.cuda.is_available():
        device = "cuda";
        print("[Info] Use CUDA")
    else:
        device = "cpu"

    # config = copy.deepcopy(hparams.base_config)
    # # config.override(effnetv2_configs.get_model_config(model_name))
    # # config.override(datasets.get_dataset_config(dataset_cfg))
    # # config.override(hparam_str)
    # # config.override(sweeps)
    # train_split = config.train.split or 'train'
    # eval_split = config.eval.split or 'eval'
    # num_train_images = config.data.splits[train_split].num_images
    # num_eval_images = config.data.splits[eval_split].num_images

    # train_size = 256
    # eval_size = 256
    # if train_size <= 16.:
    #     train_size = int(eval_size * train_size) // 16 * 16
    # input_image_size = eval_size if mode == 'eval' else train_size

    # image_dtype = 'float16'

    # dataset_eval = datasets.build_dataset_input(False, input_image_size,
    #                                           image_dtype, data_dir,
    #                                           eval_split, config.data)
    # dataset_train = datasets.build_dataset_input(True, input_image_size,
    #                                                  image_dtype,
    #                                                  data_dir,
    #                                                  train_split, config.data)
    
    model1 = effnetv2_m(num_classes=21*36).to(device)
    dataloaders = Dataloaders(IMAGE_PATH, LABELS_PATH, BATCH_SIZE)

    #ハイパーパラメータ最適化も入れてみよう

    # optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    optimizer1 = torch.optim.AdamW(model1.parameters(), lr=1e-5, weight_decay=5e-4)
    # ここではOptimizerに入れただけで実際の学習は下のMyTrainerで行っている。

    #       scheduler1 = optim.lr_scheduler.ExponentialLR(optimizer1, gamma=0.99)
    #       scheduler2 = optim.lr_scheduler.ExponentialLR(optimizer2, gamma=0.99)
    #lossがnanになるのはよくあるので、こういうときはoptimizerを変えるか学習率変えるかするといい
    trainer1 = MyTrainer(model1, dataloaders, optimizer1, device)
    
    #       trainer1 = MyTrainer(model1, dataloaders, optimizer1, scheduler1, device, "Classification")
    #       trainer2 = MyTrainer(model2, dataloaders, optimizer2, scheduler2, device, "Regression")

    trainer1.run(NUM_EPOCH)#
    
if __name__ == '__main__':
	main()