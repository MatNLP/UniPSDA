import os, logging, json
from tqdm import tqdm
import torch
from torch import nn, optim
from .data_loader import SentenceRELoader
from .utils import AverageMeter
import random

class SentenceRE(nn.Module):
    def __init__(self,
                 model,
                 train_path, 
                 val_path, 
                 test_path,
                 ckpt, 
                 batch_size=32, 
                 max_epoch=100, 
                 lr=0.1, 
                 weight_decay=1e-5, 
                 warmup_step=300,
                 opt='sgd',
                 eta =0.001):
    
        super().__init__()
        self.max_epoch = max_epoch
        self.eta = eta
        # Load data
        if train_path != None:
            self.train_loader = SentenceRELoader(
                train_path,
                model.rel2id,
                model.sentence_encoder.tokenize,
                batch_size,
                # True)
                False)

        if train_path != None:
            self.train_loader_with_sep = SentenceRELoader(
                '../en-train_with_SEP.txt',
                model.rel2id,
                model.sentence_encoder.tokenize,
                batch_size,
                # True)
                False)

        if val_path != None:
            self.val_loader = SentenceRELoader(
                val_path,
                model.rel2id,
                model.sentence_encoder.tokenize,
                batch_size,
                False)
        
        if test_path != None:
            self.test_loader = SentenceRELoader(
                test_path,
                model.rel2id,
                model.sentence_encoder.tokenize,
                batch_size,
                False
            )
        # Model
        self.model = model
        self.parallel_model = nn.DataParallel(self.model)
        # Criterion
        self.criterion = nn.CrossEntropyLoss()
        # Params and optimizer
        params = self.parameters()
        self.lr = lr
        if opt == 'sgd':
            self.optimizer = optim.SGD(params, lr, weight_decay=weight_decay)
        elif opt == 'adam':
            self.optimizer = optim.Adam(params, lr, weight_decay=weight_decay)
        elif opt == 'adamw': # Optimizer for BERT
            from transformers import AdamW
            params = list(self.named_parameters())
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            grouped_params = [
                {
                    'params': [p for n, p in params if not any(nd in n for nd in no_decay)], 
                    'weight_decay': 0.01,
                    'lr': lr,
                    'ori_lr': lr
                },
                {
                    'params': [p for n, p in params if any(nd in n for nd in no_decay)], 
                    'weight_decay': 0.0,
                    'lr': lr,
                    'ori_lr': lr
                }
            ]
            self.optimizer = AdamW(grouped_params, correct_bias=False)
        else:
            raise Exception("Invalid optimizer. Must be 'sgd' or 'adam' or 'adamw'.")
        # Warmup
        if warmup_step > 0:
            from transformers import get_linear_schedule_with_warmup
            training_steps = self.train_loader.dataset.__len__() // batch_size * self.max_epoch
            self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=warmup_step, num_training_steps=training_steps)
        else:
            self.scheduler = None
        # Cuda
        if torch.cuda.is_available():
            self.cuda()
        # Ckpt
        self.ckpt = ckpt

    def load_batch_candidate_embeddings(self,iter, batchsize, senNumKey_wordsHiddenListValue, senNumKey_candidateTensorValue):

        batch_candidate_embeddings_keys = list(senNumKey_wordsHiddenListValue.keys())[iter*batchsize : (iter + 1)*batchsize]
        batch_candidate_embeddings_list = []
        for key in batch_candidate_embeddings_keys:
            batch_candidate_embeddings_list.append(senNumKey_wordsHiddenListValue[key])

        batch_candidate_embeddings_keys2 = list(senNumKey_candidateTensorValue.keys())[
                                          iter * batchsize: (iter + 1) * batchsize]
        batch_candidate_embeddings_list3 = []
        for i, key in enumerate(batch_candidate_embeddings_keys2):
            every_sen_candidate_tensor = torch.tensor(1)
            for iter, tensor_candidate_all in enumerate(senNumKey_candidateTensorValue[key]):
                random_index = random.randint(0, tensor_candidate_all.size(0)-1)
                random_pick_tensor = tensor_candidate_all[random_index, :].unsqueeze(0)
                if random_pick_tensor.sum(1) == 0:
                    random_pick_tensor = batch_candidate_embeddings_list[i][iter]
                if iter == 0:
                    every_sen_candidate_tensor = random_pick_tensor
                else:
                    every_sen_candidate_tensor = torch.cat((every_sen_candidate_tensor,random_pick_tensor),dim=0)
            batch_candidate_embeddings_list3.append(every_sen_candidate_tensor)
        return batch_candidate_embeddings_list3

    def pos_102_list(self,a):
        b = (a == 102).nonzero().tolist()
        # print(b)
        batch_sen_pos = []
        for i in range(a.size(0)):
            sen_pos_each = []
            for pos in b:
                if pos[0] == i:
                    sen_pos_each.append(pos[1])
            if len(sen_pos_each)>1:
                batch_sen_pos.append(sen_pos_each[:-1])
            elif len(sen_pos_each)>2:
                batch_sen_pos.append(sen_pos_each[:2])
            else:
                batch_sen_pos.append(sen_pos_each)
        # print(batch_sen_pos)
        return batch_sen_pos

    def data_with_SEP_from_dataloader(self):
        train_loader_with_sep_0 = (self.train_loader_with_sep)
        # data_with_SEP_list_input_ids = []
        # data_with_SEP_list_att_mask = []
        tokenizer_result_with_SEP_list = []
        for iter, data_with_SEP in enumerate(train_loader_with_sep_0):
            tokenizer_result_with_SEP_list.append(data_with_SEP)
            # data_with_SEP_list_input_ids.append(data_with_SEP[1])
            # data_with_SEP_list_att_mask.append(data_with_SEP[2])
        # return data_with_SEP_list_input_ids, data_with_SEP_list_att_mask
        return tokenizer_result_with_SEP_list

    def train_model(self, metric='acc'):
        best_metric = 0
        global_step = 0
        path = '../senNumKey_wordsHiddenListValue_re.pt'
        senNumKey_wordsHiddenListValue = torch.load(path)
        path = '../senNumKey_candidateTensorValue_re.pt'
        senNumKey_candidateTensorValue = torch.load(path)


        for epoch in range(self.max_epoch):
            self.train()
            logging.info("=== Epoch %d train ===" % epoch)
            avg_loss = AverageMeter()
            avg_acc = AverageMeter()
            tokenizer_result_with_SEP_list = self.data_with_SEP_from_dataloader()
            t = tqdm(self.train_loader)
            for iter, data in enumerate(t):
                batch_data_with_SEP = tokenizer_result_with_SEP_list[iter]

                if torch.cuda.is_available():
                    for i in range(len(data)):
                        try:
                            data[i] = data[i].cuda()
                        except:
                            pass
                label = data[0]
                args = data[1:]
                batch_sen_pos = self.pos_102_list(batch_data_with_SEP[1])
                batch_candidate_embeddings_list = self.load_batch_candidate_embeddings(iter, len(label),
                                                                                       senNumKey_wordsHiddenListValue,
                                                                                       senNumKey_candidateTensorValue)
                logits, mean_distance_from_SVD, mean_last_k_sigma_sum, mean_Wasserstein_Distance = self.parallel_model(len(label), batch_sen_pos, batch_candidate_embeddings_list, batch_data_with_SEP, *args)





                loss = self.criterion(logits, label)
                # loss = 0.8*loss + 0.05*mean_distance_from_SVD + 0.05*self.eta*mean_last_k_sigma_sum + 0.1*mean_Wasserstein_Distance
                # loss = 0.5*loss + 0.48*mean_distance_from_SVD + 0.01*self.eta*mean_last_k_sigma_sum + 0.01*mean_Wasserstein_Distance
                # loss = 0.6*loss + 0.3*mean_distance_from_SVD + 0.1*self.eta*mean_last_k_sigma_sum
                score, pred = logits.max(-1) # (B)
                acc = float((pred == label).long().sum()) / label.size(0)
                # Log
                avg_loss.update(loss.item(), 1)
                avg_acc.update(acc, 1)
                t.set_postfix(loss=avg_loss.avg, acc=avg_acc.avg)
                # Optimize
                loss.backward()
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                self.optimizer.zero_grad()
                global_step += 1
            # Val 
            logging.info("=== Epoch %d val ===" % epoch)
            result = self.eval_model(self.val_loader) 
            logging.info('Metric {} current / best: {} / {}'.format(metric, result[metric], best_metric))
            if result[metric] > best_metric:
                logging.info("Best ckpt and saved.")
                folder_path = '/'.join(self.ckpt.split('/')[:-1])
                if not os.path.exists(folder_path):
                    os.mkdir(folder_path)
                torch.save({'state_dict': self.model.state_dict()}, self.ckpt)
                best_metric = result[metric]
        logging.info("Best %s on val set: %f" % (metric, best_metric))

    def eval_model(self, eval_loader):
        self.eval()
        avg_acc = AverageMeter()
        pred_result = []
        with torch.no_grad():
            t = tqdm(eval_loader)
            for iter, data in enumerate(t):
                if torch.cuda.is_available():
                    for i in range(len(data)):
                        try:
                            data[i] = data[i].cuda()
                        except:
                            pass
                label = data[0]
                args = data[1:]        
                logits = self.parallel_model(None, None, None, None,*args)
                score, pred = logits.max(-1) # (B)
                # Save result
                for i in range(pred.size(0)):
                    pred_result.append(pred[i].item())
                # Log
                acc = float((pred == label).long().sum()) / label.size(0)
                avg_acc.update(acc, pred.size(0))
                t.set_postfix(acc=avg_acc.avg)
        result = eval_loader.dataset.eval(pred_result)
        return result

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

