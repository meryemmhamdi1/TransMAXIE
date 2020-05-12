# @Author : bamtercelboo
# @Datetime : 2018/9/14 9:51
# @File : CRF.py
# @Last Modify Time : 2018/9/14 9:51
# @Contact : bamtercelboo@{gmail.com, 163.com}

"""
  TESTING SYNCING TEST
    FILE :  CRF.py
    FUNCTION : None
    REFERENCE : https://github.com/jiesutd/NCRFpp/blob/master/model/crf.py
"""
import torch
from torch.autograd.variable import Variable
import torch.nn as nn


def log_sum_exp(vec, m_size):
    """
    Args:
        vec: size=(batch_size, vanishing_dim, hidden_dim)
        m_size: hidden_dim

    Returns:
        size=(batch_size, hidden_dim)
    """
    _, idx = torch.max(vec, 1)  # B * 1 * M
    max_score = torch.gather(vec, 1, idx.view(-1, 1, m_size)).view(-1, 1, m_size)  # B * M
    return max_score.view(-1, m_size) + torch.log(torch.sum(
        torch.exp(vec - max_score.expand_as(vec)), 1)).view(-1, m_size)


def log_sum_exp_batch(log_Tensor, axis=-1): # shape (batch_size,n,m)
    return torch.max(log_Tensor, axis)[0]+torch.log(torch.exp(log_Tensor-torch.max(log_Tensor, axis)[0].view(log_Tensor.shape[0],-1,1)).sum(axis))


class CRF(nn.Module):
    """
        CRF
    """
    def __init__(self, **kwargs):
        """
        kwargs:
            target_size: int, target size
            device: str, device
        """
        super(CRF, self).__init__()

        for k in kwargs:
            self.__setattr__(k, kwargs[k])
        device = self.device

        # Matrix of transition parameters.  Entry i,j is the score of transitioning *to* i *from* j.
        self.transitions = nn.Parameter(torch.randn(self.target_size, self.target_size))
        self.START_TAG, self.STOP_TAG = -2, -1

        # These two statements enforce the constraint that we never transfer *to* the start tag(or label),
        # and we never transfer *from* the stop label (the model would probably learn this anyway,
        # so this enforcement is likely unimportant) # TODO INVESTIGATE THIS ISSUE BCOS WE DON'T USE CLS AND SEP HERE
        #self.transitions.data[start_label_id, :] = -10000
        #self.transitions.data[:, stop_label_id] = -10000

        nn.init.xavier_uniform_(self.hidden2label.weight)
        nn.init.constant_(self.hidden2label.bias, 0.0)

    def _viterbi_decode(self, feats):
        """
            input:
                feats: (batch, seq_len, self.tag_size)
                mask: (batch, seq_len)
            output:
                decode_idx: (batch, seq_len) decoded sequence
                path_score: (batch, 1) corresponding score for each sequence (to be implementated)
        """
        batch_size = feats.size(0)
        T = feats.size(1)

        # batch_transitions=self.transitions.expand(batch_size,self.target_size,self.target_size)

        log_delta = torch.Tensor(batch_size, 1, self.target_size).fill_(-10000.).to(self.device)
        #log_delta[:, 0, self.start_label_id] = 0

        # psi is for the vaule of the last latent that make P(this_latent) maximum.
        psi = torch.zeros((batch_size, T, self.target_size), dtype=torch.long).to(self.device)  # psi[0]=0000 useless
        for t in range(1, T):
            # delta[t][k]=max_z1:t-1( p(x1,x2,...,xt,z1,z2,...,zt-1,zt=k|theta) )
            # delta[t] is the max prob of the path from  z_t-1 to z_t[k]
            log_delta, psi[:, t] = torch.max(self.transitions + log_delta, -1)
            # psi[t][k]=argmax_z1:t-1( p(x1,x2,...,xt,z1,z2,...,zt-1,zt=k|theta) )
            # psi[t][k] is the path choosed from z_t-1 to z_t[k],the value is the z_state(is k) index of z_t-1
            log_delta = (log_delta + feats[:, t]).unsqueeze(1)

        # trace back
        path = torch.zeros((batch_size, T), dtype=torch.long).to(self.device)

        # max p(z1:t,all_x|theta)
        max_logLL_allz_allx, path[:, -1] = torch.max(log_delta.squeeze(), -1)

        for t in range(T-2, -1, -1):
            # choose the state of z_t according the state choosed of z_t+1.
            path[:, t] = psi[:, t+1].gather(-1,path[:, t+1].view(-1,1)).squeeze()

        return max_logLL_allz_allx, path

    def forward(self, feats):
        """
        :param feats:
        :param mask:
        :return:
        """
        path_score, best_path = self._viterbi_decode(feats)
        return path_score, best_path

    def _forward_alg(self, feats, mask):
        """
        Do the forward algorithm to compute the partition function (batched).

        Args:
            feats: size=(batch_size, seq_len, self.target_size+2)
            mask: size=(batch_size, seq_len)

        Returns:
            xxx
        """
        #print("feats.size():", feats.size())
        batch_size = feats.size(0)
        seq_len = feats.size(1)
        tag_size = feats.size(2)
        mask = mask.transpose(1, 0).contiguous()
        ins_num = seq_len * batch_size
        """ be careful the view shape, it is .view(ins_num, 1, tag_size) but not .view(ins_num, tag_size, 1) """
        feats = feats.transpose(1,0).contiguous().view(ins_num,1, tag_size).expand(ins_num, tag_size, tag_size)
        """ need to consider start """
        scores = feats + self.transitions.view(1, tag_size, tag_size).expand(ins_num, tag_size, tag_size)
        scores = scores.view(seq_len, batch_size, tag_size, tag_size)
        # build iter
        seq_iter = enumerate(scores)
        _, inivalues = next(seq_iter)  # bat_size * from_target_size * to_target_size
        """ only need start from start_tag """
        partition = inivalues[:, self.START_TAG, :].clone().view(batch_size, tag_size, 1)  # bat_size * to_target_size

        """
        add start score (from start to all tag, duplicate to batch_size)
        partition = partition + self.transitions[START_TAG,:].view(1, tag_size, 1).expand(batch_size, tag_size, 1)
        iter over last scores
        """
        for idx, cur_values in seq_iter:
            """
            previous to_target is current from_target
            partition: previous results log(exp(from_target)), #(batch_size * from_target)
            cur_values: bat_size * from_target * to_target
            """
            cur_values = cur_values + partition.contiguous().view(batch_size, tag_size, 1).expand(batch_size, tag_size, tag_size)
            cur_partition = log_sum_exp(cur_values, tag_size)

            mask_idx = mask[idx, :].view(batch_size, 1).expand(batch_size, tag_size)

            """ effective updated partition part, only keep the partition value of mask value = 1 """
            masked_cur_partition = cur_partition.masked_select(mask_idx)
            """ let mask_idx broadcastable, to disable warning """
            mask_idx = mask_idx.contiguous().view(batch_size, tag_size, 1)

            """ replace the partition where the maskvalue=1, other partition value keeps the same """
            partition.masked_scatter_(mask_idx, masked_cur_partition)
        """
        until the last state, add transition score for all partition (and do log_sum_exp)
        then select the value in STOP_TAG
        """
        cur_values = self.transitions.view(1, tag_size, tag_size).expand(batch_size, tag_size, tag_size) + partition.contiguous().view(batch_size, tag_size, 1).expand(batch_size, tag_size, tag_size)
        cur_partition = log_sum_exp(cur_values, tag_size)
        final_partition = cur_partition[:, self.STOP_TAG]
        return final_partition.sum(), scores

    def _score_sentence(self, scores, mask, tags):
        """
        Args:
            scores: size=(seq_len, batch_size, tag_size, tag_size)
            mask: size=(batch_size, seq_len)
            tags: size=(batch_size, seq_len)

        Returns:
            score:
        """
        # print(scores.size())
        batch_size = scores.size(1)
        seq_len = scores.size(0)
        tag_size = scores.size(-1)
        tags = tags.view(batch_size, seq_len)
        """ convert tag value into a new format, recorded label bigram information to index """
        # new_tags = Variable(torch.LongTensor(batch_size, seq_len))
        new_tags = torch.empty(batch_size, seq_len, device=self.device, requires_grad=True).long()
        for idx in range(seq_len):
            if idx == 0:
                new_tags[:, 0] = (tag_size - 2) * tag_size + tags[:, 0]
            else:
                new_tags[:, idx] = tags[:, idx-1] * tag_size + tags[:, idx]

        """ transition for label to STOP_TAG """
        end_transition = self.transitions[:, self.STOP_TAG].contiguous().view(1, tag_size).expand(batch_size, tag_size)
        """ length for batch,  last word position = length - 1 """
        length_mask = torch.sum(mask, dim=1).view(batch_size, 1).long()
        """ index the label id of last word """
        end_ids = torch.gather(tags, 1, length_mask-1)

        """ index the transition score for end_id to STOP_TAG """
        end_energy = torch.gather(end_transition, 1, end_ids)

        """ convert tag as (seq_len, batch_size, 1) """
        new_tags = new_tags.transpose(1, 0).contiguous().view(seq_len, batch_size, 1)
        """ need convert tags id to search from 400 positions of scores """
        tg_energy = torch.gather(scores.view(seq_len, batch_size, -1), 2, new_tags).view(seq_len, batch_size)
        tg_energy = tg_energy.masked_select(mask.transpose(1, 0))

        """
        add all score together
        gold_score = start_energy.sum() + tg_energy.sum() + end_energy.sum()
        """
        gold_score = tg_energy.sum() + end_energy.sum()

        return gold_score

    def neg_log_likelihood_loss(self, feats, mask, tags):
        """
        Args:
            feats: size=(batch_size, seq_len, tag_size)
            mask: size=(batch_size, seq_len)
            tags: size=(batch_size, seq_len)
        """
        forward_score, scores = self._forward_alg(feats, mask)
        gold_score = self._score_sentence(scores, mask, tags)
        return forward_score - gold_score
