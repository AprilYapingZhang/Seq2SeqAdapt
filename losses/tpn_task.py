# pytorch implementation for Transferrable Prototypical Networks for Unsupervised Domain Adaptation
# Sample-level discrepancy loss in Section 3.4 Task-specific Domain Adaptation
# https://arxiv.org/pdf/1904.11227.pdf

import torch
import torch.functional as F
from torch import nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TpnTaskLoss(nn.Module):
    def __init__(self):
        super(TpnTaskLoss, self).__init__()

    def forward(self, src_feat, trg_feat, src_label, trg_label):
        labels = list(src_label.data.cpu().numpy())
        labels = list(set(labels))

        dim = src_feat.size(1)
        center_num = len(labels)

        u_s = torch.zeros(center_num, dim).to(device)
        u_t = torch.zeros(center_num, dim).to(device)
        u_st = torch.zeros(center_num, dim).to(device)

        for i, l in enumerate(labels):
            s_feat = src_feat[src_label == l]
            t_feat = trg_feat[trg_label == l]

            u_s[i, :] = s_feat.mean(dim=0)
            u_t[i, :] = t_feat.mean(dim=0)
            u_st[i, :] = (s_feat.sum(dim=0) + t_feat.sum(dim=0)) / (s_feat.size(0) + t_feat.size(0))

        feats = torch.cat((src_feat, trg_feat), dim=0)
        p_s = torch.matmul(feats, u_s.t())
        p_t = torch.matmul(feats, u_t.t())
        p_st = torch.matmul(feats, u_st.t())

        loss_st = (F.kl_div(F.log_softmax(p_s, dim=-1), F.log_softmax(p_t, dim=-1),
                            reduction='mean') +
                   F.kl_div(F.log_softmax(p_t, dim=-1), F.log_softmax(p_s, dim=-1),
                            reduction='mean')) / 2
        loss_sst = (F.kl_div(F.log_softmax(p_s, dim=-1), F.log_softmax(p_st, dim=-1),
                             reduction='mean') +
                    F.kl_div(F.log_softmax(p_st, dim=-1), F.log_softmax(p_s, dim=-1),
                             reduction='mean')) / 2
        loss_tst = (F.kl_div(F.log_softmax(p_t, dim=-1), F.log_softmax(p_st, dim=-1),
                             reduction='mean') +
                    F.kl_div(F.log_softmax(p_st, dim=-1), F.log_softmax(p_t, dim=-1),
                             reduction='mean')) / 2
        tpn_task = (loss_st + loss_sst + loss_tst) / 3
        return tpn_task, ('04. tpn_task loss: ', tpn_task.data.cpu().numpy())
