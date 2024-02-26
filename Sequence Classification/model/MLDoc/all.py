#new
import logging
import torch
import random
import pprint

import model.MLDoc.base
import util.tool

import torch.nn as nn
import numpy as np
import ot
import geomloss

# from transformers import BertTokenizer, BertModel, BertForMaskedLM, AdamW
from mbert_related_file.modeling_bert import BertModel
from transformers import BertTokenizer, BertForMaskedLM, AdamW
from torch.nn.parameter import Parameter
from torch.nn import functional as F

mean_W = torch.load('../mean_W.pt')
mean_W_sen_cla = torch.load('../CoSDA-ML/mean_W_sen_cla.pt')

class BERTTool(object):
    def init(args):
        BERTTool.multi_bert = BertModel.from_pretrained(args.multi_bert.location)
        BERTTool.multi_tokener = BertTokenizer.from_pretrained(args.multi_bert.location)
        BERTTool.multi_pad = BERTTool.multi_tokener.convert_tokens_to_ids(["[PAD]"])[0]
        BERTTool.multi_sep = BERTTool.multi_tokener.convert_tokens_to_ids(["[SEP]"])[0]
        BERTTool.multi_cls = BERTTool.multi_tokener.convert_tokens_to_ids(["[CLS]"])[0]
        #BERTTool.multi_bert.eval()
        #BERTTool.en_bert.eval()


class Model(model.MLDoc.base.Model):
    def __init__(self, args, DatasetTool, inputs):
        np.random.seed(args.train.seed)
        torch.manual_seed(args.train.seed)
        random.seed(args.train.seed)
        super().__init__(args, DatasetTool, inputs)
        _, _, _, ontology, worddict, _ = inputs
        self.ontology = ontology
        self.worddict = worddict
        BERTTool.init(self.args)
        self.bert = BERTTool.multi_bert
        self.tokener = BERTTool.multi_tokener
        self.pad = BERTTool.multi_pad
        self.sep = BERTTool.multi_sep
        self.cls = BERTTool.multi_cls
        self.P = torch.nn.Linear(768, args.train.level)
        # W will not be used in this version
        self.W = torch.nn.Linear(768, 768)
        self.Loss = torch.nn.CrossEntropyLoss()
        self.linear_after_SVD = nn.Linear(768, 768)
        self.linear_after_SVD.weight = Parameter(mean_W)
        self.linear_after_SVD_sen_cla = nn.Linear(768, 768)
        self.linear_after_SVD_sen_cla.weight = Parameter(mean_W_sen_cla)
        self.last_k = args.train.last_k
        self.eta = args.train.eta

    def set_optimizer(self):
        all_params = set(self.parameters())
        if self.args.train.bert == False:
            bert_params = set(BERTTool.multi_bert.parameters())
            for para in bert_params:
                para.requires_grad=False
            params = [{"params": list(all_params - bert_params), "lr": self.args.lr.default}]
        else:
            bert_params = set(BERTTool.multi_bert.parameters())
            params = [{"params": list(all_params - bert_params), "lr": self.args.lr.default},
                      {"params": list(bert_params), "lr": self.args.lr.bert}
                      ]
        self.optimizer = AdamW(params)

    def run_eval(self, train, dev, test):
        logging.info("Starting evaluation")
        self.eval()
        summary = {}
        ds = {"train": train, "dev": dev}
        ds.update(test)
        for set_name, dataset in ds.items():
            self.args.need_w = False
            tmp_summary, pred = self.run_test(dataset)
            self.DatasetTool.record(pred, dataset, set_name, self.args)
            summary.update({"eval_{}_{}".format(set_name, k): v for k, v in tmp_summary.items()})
        logging.info(pprint.pformat(summary))

    def run_train(self, train, dev, test):
        self.set_optimizer()
        iteration = 0
        best = {}
        for epoch in range(self.args.train.epoch):
            self.ontology = self.ontology
            self.train()
            logging.info("Starting training epoch {}".format(epoch))
            summary = self.get_summary(epoch, iteration)
            loss, iter = self.run_batches(train, epoch)
            iteration += iter
            summary.update({"loss": loss})
            if not self.args.train.not_eval:
                ds = {"train": train, "dev": dev}
                ds.update(test)
                for set_name, dataset in ds.items():
                    self.args.need_w = False
                    tmp_summary, pred = self.run_test(dataset)
                    self.DatasetTool.record(pred, dataset, set_name, self.args)
                    summary.update({"eval_{}_{}".format(set_name, k): v for k, v in tmp_summary.items()})
            best = self.update_best(best, summary, epoch)
            logging.info(pprint.pformat(best))
            logging.info(pprint.pformat(summary))

    def cross(self, x, disable=False):
        if not disable and self.training and (self.args.train.cross >= random.random()):
            lan = random.randint(0,len(self.args.dict_list) - 1)
            if x in self.worddict.src2tgt[lan]:
                return self.worddict.src2tgt[lan][x][random.randint(0,len(self.worddict.src2tgt[lan][x]) - 1)]
            else:
                return x
        else:
            return x

    def cross_str(self, x, disable=False):
        raw = x.lower().split(" ")
        out = ""
        for xx in raw:
            out += self.cross(xx, disable)
            out += " "
        #print(out)
        return out

    def cross_list(self, x, disable=False):
        return [self.cross_str(xx, not (self.training and self.args.train.ratio >= random.random())) for xx in x]

    def forward(self, batch, batch_candidate_embeddings_list, batch_replace_str_label_list, task):
        if task == 'document_classification':
            tmp_x = util.tool.in_each(batch, lambda x: x[0])
            para_of_bert_model_input = util.convert.List.to_bert_info(tmp_x, self.tokener, self.pad, self.cls,
                                                                      self.device, 128)
            batch_sen_pos = para_of_bert_model_input[2]
            # (_, batch_pos_x_tensor, batch_pos_y_tensor) = para_of_bert_model_input[2]
            tmp = self.bert(*para_of_bert_model_input[0], return_dict=False, encode_replacedEmbedding=None)
            utt = tmp[2]  # torch.Size([16, 106, 768])

            if batch_replace_str_label_list is not None:
                tmp_y_replace_str = util.tool.in_each(batch_replace_str_label_list, lambda x: x[0])
                para_of_bert_model_input_replace_str = util.convert.List.to_bert_info(tmp_y_replace_str, self.tokener,
                                                                                      self.pad, self.cls,
                                                                                      self.device, 128)
                tmp_hidden_replace_str = self.bert(*para_of_bert_model_input_replace_str[0], return_dict=False,
                                                   encode_replacedEmbedding=None)
                utt_replace_str = tmp_hidden_replace_str[2]  # torch.Size([16, 106, 768])
                # utt_replace_str = tmp_hidden_replace_str[0]  # torch.Size([16, 106, 768])

            if batch_candidate_embeddings_list is not None:
                total_distance_from_SVD_list = []
                total_sigma_last_k_list = []
                total_Wasserstein_Distance = []
                # batch_W_list = []
                for i in range(len(batch)):
                    # each_sen_sim = []
                    for j, pos in enumerate(batch_sen_pos[i]):
                        utt[i, pos, :] = batch_candidate_embeddings_list[i][j]
                        pos_focus = utt[i, pos, :]
                        if pos <= (utt_replace_str.size(1) - 1):
                            pos_focus_replace_str = utt_replace_str[i, pos, :]
                        else:
                            pos_focus_replace_str = utt_replace_str[i, (utt_replace_str.size(1) - 1), :]
                        if pos >= 2:
                            pos_a = utt[i, pos - 2, :]
                            pos_b = utt[i, pos - 1, :]
                            pos_a_replace_str = utt_replace_str[i, pos - 2, :]
                            pos_b_replace_str = utt_replace_str[i, pos - 1, :]
                        elif pos == 1:
                            pos_a = utt[i, pos - 1, :]
                            pos_b = utt[i, pos - 1, :]
                            pos_a_replace_str = utt_replace_str[i, pos - 1, :]
                            pos_b_replace_str = utt_replace_str[i, pos - 1, :]
                        else:
                            pos_a = utt[i, pos, :]
                            pos_b = utt[i, pos, :]
                            pos_a_replace_str = utt_replace_str[i, pos, :]
                            pos_b_replace_str = utt_replace_str[i, pos, :]

                        if pos <= (utt.size(1) - 2 - 1):
                            pos_c = utt[i, pos + 1, :]
                            pos_d = utt[i, pos + 2, :]
                        elif pos == utt.size(1) - 2:
                            pos_c = utt[i, pos + 1, :]
                            pos_d = utt[i, pos + 1, :]
                        else:
                            pos_c = utt[i, pos, :]
                            pos_d = utt[i, pos, :]

                        if pos <= (utt_replace_str.size(1) - 2 - 1):
                            pos_c_replace_str = utt_replace_str[i, pos + 1, :]
                            pos_d_replace_str = utt_replace_str[i, pos + 2, :]
                        elif pos == utt_replace_str.size(1) - 2:
                            pos_c_replace_str = utt_replace_str[i, pos + 1, :]
                            pos_d_replace_str = utt_replace_str[i, pos + 1, :]
                        else:
                            pos_c_replace_str = utt_replace_str[i, utt_replace_str.size(1) - 1, :]
                            pos_d_replace_str = utt_replace_str[i, utt_replace_str.size(1) - 1, :]

                        tmp_window_hidden = torch.cat((pos_a.unsqueeze(0), pos_b.unsqueeze(0), pos_focus.unsqueeze(0),
                                                       pos_c.unsqueeze(0), pos_d.unsqueeze(0)), dim=0)
                        tmp_window_hidden_replace_str = torch.cat((pos_a_replace_str.unsqueeze(0),
                                                                   pos_b_replace_str.unsqueeze(0),
                                                                   pos_focus_replace_str.unsqueeze(0),
                                                                   pos_c_replace_str.unsqueeze(0),
                                                                   pos_d_replace_str.unsqueeze(0)), dim=0)


                        after_linear_W = self.linear_after_SVD(tmp_window_hidden)
                        distance_from_SVD = torch.norm((after_linear_W - tmp_window_hidden_replace_str), p=2)
                        total_distance_from_SVD_list.append(distance_from_SVD)
                        #=============================ot Loss============================================
                        p = 1
                        entreg = 0.5  # entropy regularization factor for Sinkhorn

                        OTLoss = geomloss.SamplesLoss(
                            loss='sinkhorn', p=p,
                            cost=geomloss.utils.distances if p == 1 else geomloss.utils.squared_distances,
                            blur=entreg ** (1 / p), backend='tensorized')
                        Wasserstein_Distance = OTLoss(tmp_window_hidden, tmp_window_hidden_replace_str)
                        total_Wasserstein_Distance.append(Wasserstein_Distance)
                        #=============================ot Loss============================================


                        # SVD=============================
                        U, sigma, VT = torch.linalg.svd(tmp_window_hidden_replace_str.T @ tmp_window_hidden)
                        last_k_sigma = sigma[768 - self.last_k:]
                        last_k_sigma_sum = last_k_sigma.pow(2).sum(0)
                        total_sigma_last_k_list.append(last_k_sigma_sum)


                        # W = VT.T@U.T
                        # batch_W_list.append(W.detach().cpu())
                        # self.linear_after_SVD.weight = Parameter(W)
                        # after_linear_W = self.linear_after_SVD(tmp_window_hidden)
                        # distance_from_SVD = torch.norm((after_linear_W - tmp_window_hidden_replace_str), p=2)
                        # total_distance_from_SVD_list.append(distance_from_SVD)
                        # SVD=============================
                        # each_sen_sim.append(window_sim)
                    # batch_sens_sim.append(each_sen_sim)
                # mean_window_sim = torch.mean(torch.tensor(total_window_sim_list))
                mean_distance_from_SVD = torch.mean(torch.tensor(total_distance_from_SVD_list))
                mean_last_k_sigma_sum = torch.mean(torch.tensor(total_sigma_last_k_list))
                mean_Wasserstein_Distance = torch.mean(torch.tensor(total_Wasserstein_Distance))

            utt = self.bert(*para_of_bert_model_input[0], return_dict=False, encode_replacedEmbedding=utt)
            # _, utt = self.bert(*util.convert.List.to_bert_info(self.cross_list(util.tool.in_each(batch, lambda x : x[0])), self.tokener, self.pad, self.cls, self.device, 128)[0])
            out = self.P(utt[1])
            loss = torch.Tensor([0])
            if self.training:
                label = util.tool.in_each(batch, lambda x : x[1])
                loss_task = self.Loss(out, torch.Tensor(label).long().to(self.device))
                loss = 0.5*loss_task + 0.5*(0.4*mean_distance_from_SVD + 0.2*self.eta * mean_last_k_sigma_sum + 0.4*mean_Wasserstein_Distance)
               
                loss.requires_grad_(True)
            return loss, out, utt[1]
        
        if task == 'sentiment_binary_classification':
            tmp_x = util.tool.in_each(batch, lambda x: x[0])

            # after_cross_list = self.cross_list(tmp_x)
            # utt = self.bert(*util.convert.List.to_bert_info(after_cross_list, self.tokener, self.pad, self.cls, self.device, 128)[0]).last_hidden_state
            para_of_bert_model_input = util.convert.List.to_bert_info(tmp_x, self.tokener, self.pad, self.cls,
                                                                      self.device, 128)
            batch_sen_pos = para_of_bert_model_input[2]
            tmp = self.bert(*para_of_bert_model_input[0], return_dict=False, encode_replacedEmbedding=None)
            # utt = self.bert(*util.convert.List.to_bert_info(self.cross_list(tmp_x,self.tokener, self.pad, self.cls, self.device, 128)[0], return_dict=False, encode_replacedEmbedding=None))
            utt = tmp[2]

            if batch_replace_str_label_list is not None:
                tmp_y_replace_str = util.tool.in_each(batch_replace_str_label_list, lambda x: x[0])
                para_of_bert_model_input_replace_str = util.convert.List.to_bert_info(tmp_y_replace_str, self.tokener,
                                                                                      self.pad, self.cls,
                                                                                      self.device, 128)
                tmp_hidden_replace_str = self.bert(*para_of_bert_model_input_replace_str[0], return_dict=False,
                                                   encode_replacedEmbedding=None)
                utt_replace_str = tmp_hidden_replace_str[2]  # torch.Size([16, 106, 768])

            if batch_candidate_embeddings_list is not None:
                total_distance_from_SVD_list = []
                total_sigma_last_k_list = []
                total_Wasserstein_Distance = []
                batch_W_list = []
                for i in range(len(batch)):
                    for j, pos in enumerate(batch_sen_pos[i]):
                        utt[i, pos, :] =  batch_candidate_embeddings_list[i][j].to('cuda')
                        pos_focus = utt[i, pos, :]
                        if pos <= (utt_replace_str.size(1) - 1):
                            pos_focus_replace_str = utt_replace_str[i, pos, :]
                        else:
                            pos_focus_replace_str = utt_replace_str[i, (utt_replace_str.size(1) - 1), :]
                        if pos >= 2:
                            pos_a = utt[i, pos - 2, :]
                            pos_b = utt[i, pos - 1, :]
                            pos_a_replace_str = utt_replace_str[i, pos - 2, :]
                            pos_b_replace_str = utt_replace_str[i, pos - 1, :]
                        elif pos == 1:
                            pos_a = utt[i, pos - 1, :]
                            pos_b = utt[i, pos - 1, :]
                            pos_a_replace_str = utt_replace_str[i, pos - 1, :]
                            pos_b_replace_str = utt_replace_str[i, pos - 1, :]
                        else:
                            pos_a = utt[i, pos, :]
                            pos_b = utt[i, pos, :]
                            pos_a_replace_str = utt_replace_str[i, pos, :]
                            pos_b_replace_str = utt_replace_str[i, pos, :]

                        if pos <= (utt.size(1) - 2 - 1):
                            pos_c = utt[i, pos + 1, :]
                            pos_d = utt[i, pos + 2, :]
                        elif pos == utt.size(1) - 2:
                            pos_c = utt[i, pos + 1, :]
                            pos_d = utt[i, pos + 1, :]
                        else:
                            pos_c = utt[i, pos, :]
                            pos_d = utt[i, pos, :]

                        if pos <= (utt_replace_str.size(1) - 2 - 1):
                            pos_c_replace_str = utt_replace_str[i, pos + 1, :]
                            pos_d_replace_str = utt_replace_str[i, pos + 2, :]
                        elif pos == utt_replace_str.size(1) - 2:
                            pos_c_replace_str = utt_replace_str[i, pos + 1, :]
                            pos_d_replace_str = utt_replace_str[i, pos + 1, :]
                        else:
                            pos_c_replace_str = utt_replace_str[i, utt_replace_str.size(1) - 1, :]
                            pos_d_replace_str = utt_replace_str[i, utt_replace_str.size(1) - 1, :]

                        tmp_window_hidden = torch.cat((pos_a.unsqueeze(0), pos_b.unsqueeze(0), pos_focus.unsqueeze(0),
                                                       pos_c.unsqueeze(0), pos_d.unsqueeze(0)), dim=0)
                        tmp_window_hidden_replace_str = torch.cat((pos_a_replace_str.unsqueeze(0),
                                                                   pos_b_replace_str.unsqueeze(0),
                                                                   pos_focus_replace_str.unsqueeze(0),
                                                                   pos_c_replace_str.unsqueeze(0),
                                                                   pos_d_replace_str.unsqueeze(0)), dim=0)

                        
                        after_linear_W = self.linear_after_SVD_sen_cla(tmp_window_hidden)
                        distance_from_SVD = torch.norm((after_linear_W - tmp_window_hidden_replace_str), p=2)
                        total_distance_from_SVD_list.append(distance_from_SVD)

                        # =============================ot============================================
                        p = 1
                        entreg = 0.5  # entropy regularization factor for Sinkhorn

                        OTLoss = geomloss.SamplesLoss(
                            loss='sinkhorn', p=p,
                            
                            cost=geomloss.utils.distances if p == 1 else geomloss.utils.squared_distances,
                            blur=entreg ** (1 / p), backend='tensorized')
                        Wasserstein_Distance = OTLoss(tmp_window_hidden, tmp_window_hidden_replace_str)
                        total_Wasserstein_Distance.append(Wasserstein_Distance)
                        # =============================ot============================================

                        # SVD=============================
                        U, sigma, VT = torch.linalg.svd(tmp_window_hidden_replace_str.T @ tmp_window_hidden)
                        last_k_sigma = sigma[768 - self.last_k:]
                        last_k_sigma_sum = last_k_sigma.pow(2).sum(0)
                        total_sigma_last_k_list.append(last_k_sigma_sum)
                        # W = VT.T@U.T
                        # batch_W_list.append(W.detach().cpu())
                        # self.linear_after_SVD.weight = Parameter(W)
                        # after_linear_W = self.linear_after_SVD(tmp_window_hidden)
                        # distance_from_SVD = torch.norm((after_linear_W - tmp_window_hidden_replace_str), p=2)
                        # total_distance_from_SVD_list.append(distance_from_SVD)
                        # SVD=============================
                        # each_sen_sim.append(window_sim)
                    # batch_sens_sim.append(each_sen_sim)
                # utt, (_, _) = self.lstm_model(utt)
                # utt = self.linear_after_BiLstm(utt)
                # mean_window_sim = torch.mean(torch.tensor(total_window_sim_list))



                mean_distance_from_SVD = torch.mean(torch.tensor(total_distance_from_SVD_list))
                mean_last_k_sigma_sum = torch.mean(torch.tensor(total_sigma_last_k_list))
                mean_Wasserstein_Distance = torch.mean(torch.tensor(total_Wasserstein_Distance))
            utt = self.bert(*para_of_bert_model_input[0], return_dict=False, encode_replacedEmbedding=utt)



            out = self.P(utt[1])
            loss = torch.Tensor([0])
            if self.training:
                label = util.tool.in_each(batch, lambda x: x[1])
                loss_0 = self.Loss(out, torch.Tensor(label).long().to(self.device))
                loss = 0.3*loss_0 + 0.7*(0.4*mean_distance_from_SVD + 0.2*self.eta * mean_last_k_sigma_sum + 0.4*mean_Wasserstein_Distance)
                # loss = loss_0
                loss.requires_grad_(True)
            return loss, out, _
        
        

    def get_pred(self, out):
        return torch.argmax(out, dim = 1).tolist()

    def start(self, inputs):
        train, dev, test, _, _, _ = inputs
        if self.args.model.resume is not None:
            self.load(self.args.model.resume)
        if self.args.model.w is not None:
            self.load_w(self.args.model.w)
        if not self.args.model.test:
            self.run_train(train, dev, test)
        if self.args.model.resume is not None:
            self.run_eval(train, dev, test)

    def load_w(self, file):
        logging.info("Loading w from {}".format(file))
        state = torch.load(file)
        new_state = {"bias": torch.zeros(self.args.dimension.emb), "weight": state["weight"]}
        self.W.load_state_dict(new_state)

    def load(self, file):
        logging.info("Loading model from {}".format(file))
        state = torch.load(file)
        model_state = state["model"]
        model_state.update({"W.weight": self.W.weight, "W.bias": self.W.bias})
        self.load_state_dict(model_state)