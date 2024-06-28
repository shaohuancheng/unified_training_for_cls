"""
-*- coding: utf-8 -*-
2024/1/18 20:17 model.py
"""

import argparse
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, MBartForConditionalGeneration

from metrics import ROUGEScore


def kl_loss(p_dis, q_dis, target):
    # decoder的近似
    p = torch.log_softmax(p_dis, dim=-1) + 1e-9
    p_tec = torch.softmax(p_dis, dim=-1) + 1e-9
    q = torch.log_softmax(q_dis, dim=-1) + 1e-9
    q_tec = torch.softmax(q_dis, dim=-1) + 1e-9

    pad_mask = target.unsqueeze(-1).eq(-100)  # label的ignore_index
    kl_loss = torch.nn.functional.kl_div(p, q_tec, reduction='none')
    reverse_kl_loss = torch.nn.functional.kl_div(q, p_tec, reduction='none')
    kl_loss.masked_fill_(pad_mask, 0.)
    reverse_kl_loss.masked_fill_(pad_mask, 0.)
    total_kl = 0.5 * (kl_loss.mean() + reverse_kl_loss.mean())
    return total_kl


def l2_loss(p_dis, q_dis, target):
    # p_diss 是cls方向的  q_diss 是ms方向的
    # 全部：p_dis.decoder_hidden_states  最后一层：p_dis.last_hidden_state
    pad_mask = target.unsqueeze(-1).eq(-100)  # label的ignore_index

    # 最后一层
    # mse_loss = F.mse_loss(p_dis.last_hidden_state, q_dis.last_hidden_state, reduction='none')  # 计算每个样本的均方误差
    # mse_loss.masked_fill_(pad_mask, 0.)
    # return mse_loss.mean()

    # 每一层
    # total_loss = torch.zeros(1).to(target.device)
    # decoder_hidden_states 的第0个是嵌入层，不需要计算进来
    # for p, q in zip(p_dis.decoder_hidden_states, q_dis.decoder_hidden_states):
    #     mse_loss = F.mse_loss(p, q, reduction='none')
    #     mse_loss.masked_fill_(pad_mask, 0.)
    #     total_loss += mse_loss.mean()
    # return total_loss/len(p_dis.decoder_hidden_states)  # 除以层数

    total_loss = torch.zeros(1).to(target.device)
    for layer_i in range(6, len(p_dis.decoder_hidden_states), 6):  # 跳过嵌入层
        mse_loss = F.mse_loss(p_dis.decoder_hidden_states[layer_i], q_dis.decoder_hidden_states[layer_i],
                              reduction='none')
        mse_loss.masked_fill_(pad_mask, 0.)
        total_loss += mse_loss.mean()
    return total_loss / 2  #  (len(p_dis.decoder_hidden_states)-1)  # 除以层数


class cls_MBart(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        if type(args) == dict:
            args = argparse.Namespace(**args)
        self.args = args
        self.learning_rate = args.learning_rate
        # 用的是mbart-cc-25
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path,
                                                              src_lang=args.src_lang,
                                                              tgt_lang=args.tgt_lang, )
        self.model = MBartForConditionalGeneration.from_pretrained(args.model_name_or_path)
        forced_bos_token_id = (
            self.tokenizer.lang_code_to_id[args.tgt_lang] if args.tgt_lang is not None else None
        )
        self.model.config.forced_bos_token_id = forced_bos_token_id
        self.model.config.decoder_start_token_id = forced_bos_token_id
        self.test_abs_rouge = ROUGEScore()
        self.save_hyperparameters(args)
        self.enc_loss = torch.nn.MSELoss()

    def forward(self, input_ids, attention_mask, labels, ms_inputs_ids, ms_attention_mask):
        outputs = self.model(input_ids=input_ids,
                          attention_mask=attention_mask,
                          labels=labels,
                          output_hidden_states=True
                          )   # [0]

        outputs2 = self.model(input_ids=ms_inputs_ids,
                             attention_mask=ms_attention_mask,
                             labels=labels,
                             output_hidden_states=True
                             )

        decoder_kl = kl_loss(outputs.logits, outputs2.logits, labels)
        # 每一层的情况需要修改output_hidden_states为True 不用最好关掉，节约资源
        layer_loss = l2_loss(outputs, outputs2, labels)

        losses = 0.5 * (outputs[0]+outputs2[0]) + layer_loss + decoder_kl

        return losses

    def training_step(self, batch, batch_idx):
        # get loss
        loss = self(**batch['tokenized_contents'], **batch['tokenized_ms'])  # [0]是cross entropy loss [1]是输出的logits

        return loss

    def validation_step(self, batch, batch_idx):
        # batch
        # get summary
        labels = batch['tokenized_contents'].pop('labels')
        summary_ids = self.model.generate(**batch['tokenized_contents'],
                                          num_beams=self.args.num_beams,
                                          max_length=self.args.max_output_len,
                                          early_stopping=True,
                                          no_repeat_ngram_size=self.args.no_repeat_ngram_size,
                                          )
        return [summary_ids, batch['refers']]
        # loss = self(**batch['tokenized_contents'], **batch['tokenized_ms'])
        # self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        # return loss

    def validation_epoch_end(self, outputs):
        summary = []
        reference = []
        for item in outputs:
            try:
                summary_id = item[0]
                if self.args.tgt_lang != "zh_CN":
                    one_summary = [self.tokenizer.decode([i for i in g if i != -100], skip_special_tokens=True,
                                                         clean_up_tokenization_spaces=False) for g in summary_id]
                    self.test_abs_rouge.update(one_summary, item[1])
                else:
                    one_summary = [self.tokenizer.decode([i for i in g if i != -100], skip_special_tokens=True,
                                                         clean_up_tokenization_spaces=False) for g in summary_id]

                    # 用当前的tokenizer分词后计算rouge todo [1:]可能不匹配mbart50，检查tokenize的输出
                    self.test_abs_rouge.update([' '.join(self.tokenizer.tokenize(sum))[1:] for sum in one_summary], #[1:]
                                               [' '.join(self.tokenizer.tokenize(sum))[1:] for sum in item[1]])  #[1:]
                summary += one_summary
                reference += item[1]
            except:
                print("某个生成出错啦")
        test_abs_rouge_results = self.test_abs_rouge.compute()
        self.log('val_R1', test_abs_rouge_results["rouge-1"]["f"], on_epoch=True, prog_bar=True,
                 sync_dist=True)
        self.log('val_R2', test_abs_rouge_results["rouge-2"]["f"], on_epoch=True, prog_bar=True,
                 sync_dist=True)
        self.log('val_RL', test_abs_rouge_results["rouge-L"]["f"], on_epoch=True, prog_bar=True,
                 sync_dist=True)
        self.test_abs_rouge.reset()
        self.save_txt(self.args.val_save_file + '_reference', reference)
        self.save_txt(self.args.val_save_file + '_summary', summary)

    def test_step(self, batch, batch_idx):
        # get summary
        labels = batch['tokenized_contents'].pop('labels')
        summary_ids = self.model.generate(**batch['tokenized_contents'],
                                          num_beams=self.args.num_beams,
                                          max_length=self.args.max_output_len,
                                          early_stopping=True,
                                          no_repeat_ngram_size=self.args.no_repeat_ngram_size,
                                          )
        return [summary_ids, batch['refers']]

    def test_epoch_end(self, outputs):
        summary = []
        reference = []
        for item in outputs:
            try:
                summary_id = item[0]
                if self.args.tgt_lang != "zh_CN":
                    one_summary = [self.tokenizer.decode([i for i in g if i != -100], skip_special_tokens=True,
                                                         clean_up_tokenization_spaces=False) for g in summary_id]
                    self.test_abs_rouge.update(one_summary, item[1])
                else:
                    one_summary = [self.tokenizer.decode([i for i in g if i != -100], skip_special_tokens=True,
                                                         clean_up_tokenization_spaces=False) for g in summary_id]

                    # 用当前的tokenizer分词后计算rouge
                    self.test_abs_rouge.update([' '.join(self.tokenizer.tokenize(sum))[1:] for sum in one_summary],
                                               # [1:]
                                               [' '.join(self.tokenizer.tokenize(sum))[1:] for sum in item[1]])  # [1:]
                summary += one_summary
                reference += item[1]
            except:
                print("某个生成出错啦")
        test_abs_rouge_results = self.test_abs_rouge.compute()
        self.log('test_R1', test_abs_rouge_results["rouge-1"]["f"], on_epoch=True, prog_bar=True,
                 sync_dist=True)
        self.log('test_R2', test_abs_rouge_results["rouge-2"]["f"], on_epoch=True, prog_bar=True,
                 sync_dist=True)
        self.log('test_RL', test_abs_rouge_results["rouge-L"]["f"], on_epoch=True, prog_bar=True,
                 sync_dist=True)
        self.test_abs_rouge.reset()
        self.save_txt(self.args.test_save_file + '_reference', reference)
        self.save_txt(self.args.test_save_file + '_summary', summary)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, betas=(0.9, 0.998))  # , eps=1e-9

    def save_txt(self, file_name, list_data):
        file = open(file_name, 'w', encoding='utf-8')
        list_data = [item + '\n' for item in list_data]
        file.writelines(list_data)
        file.close()