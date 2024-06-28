"""
-*- coding: utf-8 -*-
2024/2/26 17:20 translate.py
用于单独测试某个checkpoint文件的效果
"""

import argparse

import yaml
import torch
from torch.utils.data import DataLoader, Dataset

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from model import cls_MBart
from dataModule import SummaryDataModule


if __name__ == '__main__':
    # prepare for parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="config file")
    args = parser.parse_args()
    args = yaml.load(open(args.config, "r", encoding="utf-8").read(), Loader=yaml.FullLoader)
    args = argparse.Namespace(**args)

    torch.set_float32_matmul_precision('medium')

    # trainer = Trainer(**args.train_params,
    #                   fast_dev_run=False,  # 关闭“快速开发运行”模式。在这种模式下，模型仅在少量数据上运行几个步骤，以便快速检查模型是否能够正常训练
    #                   callbacks=[checkpoint_callback],
    #                   )
    trainer = Trainer(accelerator='gpu')

    model = cls_MBart.load_from_checkpoint(args.checkpoint)
    print('成功初始化模型')

    summary_data = SummaryDataModule(args)
    print('成功初始化数据')

    # my_dict = vars(args).copy()
    # if args.ch2en_mode == 'dd_summary':
    #     model = cls_MBart.load_from_checkpoint(checkpoint_path=args.checkpoint, strict=False, **my_dict)




    trainer.test(model=model, dataloaders=summary_data.test_loader)


# def main():
#     # 载入 Hugging Face 模型
#     hf_model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
#
#     # 转换为 PyTorch 模型
#     model = YourLightningModule(hf_model)
#
#     # 初始化 Trainer
#     trainer = Trainer(gpus=1)  # 如果您有多个 GPU，可以设置 gpus 参数为需要的 GPU 数量
#
#     # 在测试数据上进行评估
#     trainer.test(model)
