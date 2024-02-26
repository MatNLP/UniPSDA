import torch
from torch import nn, optim
from .base_model import SentenceRE
from torch.nn.parameter import Parameter
import geomloss

class SoftmaxNN(SentenceRE):
    """
    Softmax classifier for sentence-level relation extraction.
    """

    def __init__(self, sentence_encoder, num_class, rel2id, last_k):
        """
        Args:
            sentence_encoder: encoder for sentences
            num_class: number of classes
            id2rel: dictionary of id -> relation name mapping
        """
        super().__init__()
        self.sentence_encoder = sentence_encoder
        self.num_class = num_class
        self.fc = nn.Linear(self.sentence_encoder.hidden_size, num_class)
        self.softmax = nn.Softmax(-1)
        self.rel2id = rel2id
        self.last_k = last_k
        # self.eta = eta
        self.id2rel = {}
        self.drop = nn.Dropout()
        self.linear_after_SVD = nn.Linear(768, 768)
        # self.linear_after_SVD.weight = Parameter(mean_W)
        for rel, id in rel2id.items():
            self.id2rel[id] = rel

    def infer(self, item):
        self.eval()
        _item = self.sentence_encoder.tokenize(item)
        item = []
        for x in _item:
            item.append(x.to(next(self.parameters()).device))
        logits = self.forward(*item)
        logits = self.softmax(logits)
        score, pred = logits.max(-1)
        score = score.item()
        pred = pred.item()
        return self.id2rel[pred], score
    
    def forward(self, batch_size, batch_sen_pos, batch_candidate_embeddings_list, batch_data_with_SEP, *args):
        """
        Args:
            args: depends on the encoder
        Return:
            logits, (B, N)
        """
        if batch_candidate_embeddings_list is not None:
            utt_replace_str = self.sentence_encoder(encode_replacedEmbedding=None,*args) # (B, H)
            arg_with_SEP = batch_data_with_SEP[1:]
            utt = self.sentence_encoder(encode_replacedEmbedding=None,*arg_with_SEP) # (B, H)

            total_distance_from_SVD_list = []
            total_sigma_last_k_list = []
            total_Wasserstein_Distance = []
            for i in range(batch_size):
                for j, pos in enumerate(batch_sen_pos[i][:2]):
                    a = batch_candidate_embeddings_list[i][j]
                    b = utt[i, pos, :]
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

                    # 计算替换str和替换embedding后hidden的相似度=============================
                    # window_sim = torch.cosine_similarity(tmp_window_hidden, tmp_window_hidden_replace_str)
                    # total_window_sim_list.append(window_sim.sum(0) / 5)
                    # 计算替换str和替换embedding后hidden的相似度=============================

                    after_linear_W = self.linear_after_SVD(tmp_window_hidden)
                    distance_from_SVD = torch.norm((after_linear_W - tmp_window_hidden_replace_str), p=2)
                    total_distance_from_SVD_list.append(distance_from_SVD)

                    # =============================最优传输Loss计算============================================
                    p = 1
                    entreg = 0.1  # entropy regularization factor for Sinkhorn

                    # 若以欧式距离为metric，则cost function可以直接用geomloss提供的
                    # Sinkhorn快速解
                    OTLoss = geomloss.SamplesLoss(
                        loss='sinkhorn', p=p,
                        # 对于p=1或p=2的情形
                        cost=geomloss.utils.distances if p == 1 else geomloss.utils.squared_distances,
                        blur=entreg ** (1 / p), backend='tensorized')
                    Wasserstein_Distance = OTLoss(tmp_window_hidden, tmp_window_hidden_replace_str)
                    total_Wasserstein_Distance.append(Wasserstein_Distance)
                    # =============================最优传输Loss计算============================================

                    # SVD计算W后WX-Y的L2范数=============================
                    U, sigma, VT = torch.linalg.svd(tmp_window_hidden_replace_str.T @ tmp_window_hidden)
                    last_k_sigma = sigma[768 - self.last_k:]
                    last_k_sigma_sum = last_k_sigma.pow(2).sum(0)
                    total_sigma_last_k_list.append(last_k_sigma_sum)

            mean_distance_from_SVD = torch.mean(torch.tensor(total_distance_from_SVD_list))
            mean_last_k_sigma_sum = torch.mean(torch.tensor(total_sigma_last_k_list))
            mean_Wasserstein_Distance = torch.mean(torch.tensor(total_Wasserstein_Distance))
            utt = self.sentence_encoder(encode_replacedEmbedding=utt, *args)
            # rep = self.drop(utt).sum(1) / utt.size(1)
            rep = self.drop(utt)
            logits = self.fc(rep)  # (B, N)
            return logits, mean_distance_from_SVD, mean_last_k_sigma_sum, mean_Wasserstein_Distance
        else:
            utt = self.sentence_encoder(encode_replacedEmbedding=None, *args)
            utt = self.sentence_encoder(encode_replacedEmbedding=utt,*args)
            # rep = self.drop(utt).sum(1)/utt.size(1)
            rep = self.drop(utt)
            logits = self.fc(rep) # (B, N)
            return logits

    def logit_to_score(self, logits):
        return torch.softmax(logits, -1)
