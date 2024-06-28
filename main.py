"""
-*- coding: utf-8 -*-
2024/1/16 20:31 main.py
"""
import argparse

import yaml
import torch
from torch.utils.data import DataLoader, Dataset

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from model import cls_MBart
from dataModule import SummaryDataModule
# 把细节搬到模型前向的外层

if __name__ == '__main__':
    # prepare for parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="running config path")
    args = parser.parse_args()
    args = yaml.load(open(args.config, "r", encoding="utf-8").read(), Loader=yaml.FullLoader)
    args = argparse.Namespace(**args)

    torch.set_float32_matmul_precision('medium')
    # random seed
    seed_everything(args.random_seed)

    # 创建 EarlyStopping 回调
    early_callback = EarlyStopping(monitor='val_R2', mode='max', patience=2)
    # early_callback = EarlyStopping(
    #     monitor='val_loss',  # 监控验证集上的损失
    #     min_delta=0.0001,  # 最小变化量，小于此值时认为没有改善
    #     patience=2,  # 忍耐次数，如果连续这么多次都没有改善就停止
    #     verbose=True,  # 是否打印信息
    #     mode='min'  # 模式为最小化（损失值更小才算改善）
    # )
    # save checkpoint
    checkpoint_callback = ModelCheckpoint(monitor='val_R2',  # 验证集上的R2分数
                                          save_last=True,
                                          save_top_k=2,
                                          mode='max', )
    # checkpoint_callback = ModelCheckpoint(monitor='val_loss',  # 验证集上的R2分数
    #                                       save_last=True,
    #                                       save_top_k=2,
    #                                       mode='min', )


    trainer = Trainer(**args.train_params,
                      fast_dev_run=False,  # 关闭“快速开发运行”模式。在这种模式下，模型仅在少量数据上运行几个步骤，以便快速检查模型是否能够正常训练
                      callbacks=[checkpoint_callback, early_callback],
                      )

    summary_data = SummaryDataModule(args)
    print('成功初始化数据')

    # my_dict = vars(args).copy()
    # if args.ch2en_mode == 'dd_summary':
    #     model = cls_MBart.load_from_checkpoint(checkpoint_path=args.checkpoint, strict=False, **my_dict)

    model = cls_MBart(args)
    print('成功初始化模型')

    if args.continue_train:
        trainer.fit(model=model, ckpt_path=args.checkpoint, datamodule=summary_data)

    else:
        trainer.fit(model=model, datamodule=summary_data)  # datamodule=summary_data train_dataloaders=summary_data.train_loader  ## ckpt_path自动回复所有包括优化器步数，学习率等

    # # 在训练过程中，获取最佳模型的路径
    # best_model_path = checkpoint_callback.best_model_path
    # # 在测试时加载最佳模型进行测试
    # best_model = cls_MBart.load_from_checkpoint(best_model_path)
    trainer.test(model=model, dataloaders=summary_data.test_loader, ckpt_path='best')
