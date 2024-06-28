"""
-*- coding: utf-8 -*-
2024/1/19 15:02 dataModule.py
"""

from datasets import load_dataset
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from transformers import AutoTokenizer
import os

class SummaryDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, src_lang=args.src_lang,
                                                       tgt_lang=args.tgt_lang)

        data_files = {'train': os.path.join(args.dataset_path, 'train.csv'),  # 完整的训练集：包含源语言文档列：src，单语摘要列：ms_tgt，目标语言摘要列：tgt
                      'val': os.path.join(args.dataset_path, 'valid.csv'),  # 验证集
                      'test': os.path.join(args.dataset_path, 'test.csv')}  # 测试集
        raw_datasets = load_dataset('csv', data_files=data_files, )
        # split Dataset 小样本设置的使用
        if args.train_dataset_length != -1:
            raw_datasets["train"] = raw_datasets["train"].select(range(args.train_dataset_length))
        if args.val_dataset_length != -1:
            raw_datasets["val"] = raw_datasets["val"].select(range(args.val_dataset_length))
        if args.test_dataset_length != -1:
            raw_datasets["test"] = raw_datasets["test"].select(range(args.test_dataset_length))
        # Tokenize
        self.train_loader = DataLoader(dataset=raw_datasets["train"], \
                                       batch_size=self.args.batch_size, \
                                       num_workers=args.num_workers, \
                                       shuffle=True, \
                                       collate_fn=self.collate_fn)
        self.val_loader = DataLoader(dataset=raw_datasets["val"], \
                                     batch_size=self.args.val_batch_size, \
                                     num_workers=args.num_workers, \
                                     collate_fn=self.collate_fn_test)
        self.test_loader = DataLoader(dataset=raw_datasets["test"], \
                                      batch_size=self.args.test_batch_size, \
                                      num_workers=args.num_workers, \
                                      collate_fn=self.collate_fn_test)

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader

    def collate_fn(self, data):
        articles = [d['src'].lower() for d in data]
        ms_tgt = [d['ms_tgt'].lower() for d in data]
        # refers = [d['tgt'].split('O.o')[1].lower() for d in data]
        # zh_refers = [d['tgt'].split('O.o')[0] for d in data]
        refers = [d['tgt'] for d in data]  # zh2en

        model_inputs = self.tokenizer(articles,
                                      text_target=refers,
                                      max_length=self.args.sent_token_len,
                                      padding=True,
                                      truncation=self.args.truncation,
                                      return_tensors=self.args.return_tensors)

        ms_inputs = self.tokenizer(ms_tgt,
                                  max_length=self.args.sent_token_len,
                                  padding=True,
                                  truncation=self.args.truncation,
                                  return_tensors=self.args.return_tensors)
        ms_inputs['ms_inputs_ids'] = ms_inputs.pop('input_ids')
        ms_inputs['ms_attention_mask'] = ms_inputs.pop('attention_mask')

        if self.args.padding == "max_length" and self.args.ignore_pad_token_for_loss:
            for batch_i in range(model_inputs['labels'].size()[0]):
                for token_i in range(model_inputs['labels'].size()[1]):
                    if model_inputs['labels'][batch_i][token_i].item() == self.tokenizer.pad_token_id:
                        model_inputs['labels'][batch_i][token_i] = -100

        return {'tokenized_contents': model_inputs,
                'tokenized_ms': ms_inputs,
                'refers': refers,
                }

    def collate_fn_test(self, data):
        articles = [d['src'].lower() for d in data]
        # articles = [d['ms_tgt'].lower() for d in data]
        refers = [d['tgt'] for d in data]  # zh2en

        model_inputs = self.tokenizer(articles,
                                      text_target=refers,
                                      max_length=self.args.sent_token_len,
                                      padding=True,
                                      truncation=self.args.truncation,
                                      return_tensors=self.args.return_tensors)

        if self.args.padding == "max_length" and self.args.ignore_pad_token_for_loss:
            for batch_i in range(model_inputs['labels'].size()[0]):
                for token_i in range(model_inputs['labels'].size()[1]):
                    if model_inputs['labels'][batch_i][token_i].item() == self.tokenizer.pad_token_id:
                        model_inputs['labels'][batch_i][token_i] = -100

        return {'tokenized_contents': model_inputs,
                'refers': refers,
                }
