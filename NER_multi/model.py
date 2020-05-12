# %%
import numpy as np
import torch
import torch.nn as nn

from data_utils import *
from TripletLoss.evaluators import *
from TripletLoss.losses import *


lang_dict = {0: "en", 1: "ar", 2: "de", 3: "es", 4: "zh"}


def log_sum_exp_1vec(vec):  # shape(1,m)
    max_score = vec[0, np.argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


def log_sum_exp_mat(log_M, axis=-1):  # shape(n,m)
    return torch.max(log_M, axis)[0] + torch.log(torch.exp(log_M - torch.max(log_M, axis)[0][:, None]).sum(axis))


def log_sum_exp_batch(log_Tensor, axis=-1):  # shape (batch_size,n,m)
    return torch.max(log_Tensor, axis)[0] + torch.log(
        torch.exp(log_Tensor - torch.max(log_Tensor, axis)[0].view(log_Tensor.shape[0], -1, 1)).sum(axis))


class TRANSFORMER_CRF_NER(nn.Module):

    def __init__(self, aligning_files, pre_model, use_multi_task, pooling_choice, start_label_id, stop_label_id, num_labels, max_seq_length, batch_size, device):
        super(TRANSFORMER_CRF_NER, self).__init__()
        self.hidden_size = 768
        self.start_label_id = start_label_id
        self.stop_label_id = stop_label_id
        self.num_labels = num_labels
        # self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.device = device
        self.use_multi_task = use_multi_task

        strategies = pooling_choice.split(",")
        self.pooling_mode_cls_token = False
        self.pooling_mode_mean_tokens = False
        self.pooling_mode_max_tokens = False
        self.pooling_mode_mean_sqrt_len_tokens = False

        pooling_mode_multiplier = 0
        for strategy in strategies:
            pooling_mode_multiplier += 1
            if strategy == "cls":
                self.pooling_mode_cls_token = True
            elif strategy == "mean":
                self.pooling_mode_mean_tokens = True
            elif strategy == "max":
                self.pooling_mode_max_tokens = True
            elif strategy == "mean_sqrt":
                self.pooling_mode_mean_sqrt_len_tokens = True

        self.pooling_output_dimension = (pooling_mode_multiplier * self.hidden_size)


        # use pretrainded transformer
        self.pre_model = pre_model
        self.dropout = torch.nn.Dropout(0.2)
        # Maps the output of the bert into label space.
        self.hidden2label = nn.Linear(self.hidden_size, self.num_labels)#.to(device)

        # Matrix of transition parameters.  Entry i,j is the score of transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            torch.randn(self.num_labels, self.num_labels))#.to(device)

        # These two statements enforce the constraint that we never transfer *to* the start tag(or label),
        # and we never transfer *from* the stop label (the model would probably learn this anyway,
        # so this enforcement is likely unimportant)
        self.transitions.data[start_label_id, :] = -10000
        self.transitions.data[:, stop_label_id] = -10000

        nn.init.xavier_uniform_(self.hidden2label.weight)
        nn.init.constant_(self.hidden2label.bias, 0.0)

        if aligning_files is not None:
            self.use_alignment = True
            for lang in aligning_files:
                name = "aligning_%s" % lang
                aligning_matrix = torch.eye(self.hidden_size)
                if aligning_files[lang] != '':
                    aligning_path = aligning_files[lang]
                    if "MUSE" in aligning_path:
                        aligning_matrix = torch.FloatTensor(torch.load(aligning_path))
                        aligning = torch.nn.Linear(self.hidden_size, self.hidden_size, bias=False)
                        aligning.weight = torch.nn.Parameter(aligning_matrix, requires_grad=False)
                    else:
                        aligning_matrix = torch.load(aligning_path, map_location=lambda storage, loc: storage)
                        aligning.load_state_dict(aligning_matrix)
                else:
                    aligning = torch.nn.Linear(self.hidden_size, self.hidden_size, bias=False)
                    aligning.weight = torch.nn.Parameter(aligning_matrix, requires_grad=False)

                self.add_module(name, aligning)
        else:
            self.use_alignment = False

    def _forward_alg(self, feats):
        '''
        this also called alpha-recursion or forward recursion, to calculate log_prob of all barX
        '''
        # T = self.max_seq_length
        T = feats.shape[1]
        batch_size = feats.shape[0]

        # alpha_recursion,forward, alpha(zt)=p(zt,bar_x_1:t)
        log_alpha = torch.Tensor(batch_size, 1, self.num_labels).fill_(-10000.).to(self.device)
        # normal_alpha_0 : alpha[0]=Ot[0]*self.PIs
        # self.start_label has all of the score. it is log,0 is p=1
        log_alpha[:, 0, self.start_label_id] = 0

        # feats: sentances -> word embedding -> lstm -> MLP -> feats
        # feats is the probability of emission, feat.shape=(1,tag_size)
        for t in range(1, T):
            log_alpha = (log_sum_exp_batch(self.transitions + log_alpha, axis=-1) + feats[:, t]).unsqueeze(1)

        # log_prob of all barX
        log_prob_all_barX = log_sum_exp_batch(log_alpha)
        return log_prob_all_barX

    def _get_sentence_embeddings(self, sentence_features):
        sentence_embeddings = {}
        for i, features in enumerate(sentence_features):
            if self.training:
                """
                print("features['input_ids']:", features["input_ids"].shape)
                print("features['token_type_ids']:", features["token_type_ids"].shape)
                print("features['input_mask']:", features['input_mask'].shape)
                print("***********************************************")
                GPUtil.showUtilization()
                print("***********************************************")
                """
                self.pre_model.train()
                lm_output, _ = self.pre_model(input_ids=features["input_ids"],
                                         token_type_ids=features["token_type_ids"],
                                         attention_mask=features['input_mask'])#, output_all_encoded_layers=False)

            else:
                self.pre_model.eval()
                with torch.no_grad():
                    lm_output, _ = self.pre_model(input_ids=features["input_ids"], token_type_ids=features["token_type_ids"],
                                                  attention_mask=features['input_mask'])#, output_all_encoded_layers=False)

            cls_token = lm_output[:, 0, :]

            ## 2. Pooling layer
            output_vectors = []
            if self.pooling_mode_cls_token:  # CLS Pooling
                output_vectors.append(cls_token)

            if self.pooling_mode_max_tokens:  # Max Pooling
                input_mask_expanded = features['input_mask'].unsqueeze(-1).expand(lm_output.size()).float()
                lm_output[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
                max_over_time = torch.max(lm_output, 1)[0]
                output_vectors.append(max_over_time)

            if self.pooling_mode_mean_tokens or self.pooling_mode_mean_sqrt_len_tokens:
                input_mask_expanded = features['input_mask'].unsqueeze(-1).expand(lm_output.size()).float()
                sum_embeddings = torch.sum(lm_output * input_mask_expanded, 1)

                # If tokens are weighted (by WordWeights layer), feature 'token_weights_sum' will be present
                if 'token_weights_sum' in features:
                    sum_mask = features['token_weights_sum'].unsqueeze(-1).expand(sum_embeddings.size())
                else:
                    sum_mask = input_mask_expanded.sum(1)

                sum_mask = torch.clamp(sum_mask, min=1e-9)

                if self.pooling_mode_mean_tokens:
                    output_vectors.append(sum_embeddings / sum_mask)
                if self.pooling_mode_mean_sqrt_len_tokens:
                    output_vectors.append(sum_embeddings / torch.sqrt(sum_mask))

            output_vector = torch.cat(output_vectors, 1)

            sentence_embeddings.update({i: output_vector})

        return sentence_embeddings

    def _get_trans_features(self, lang, input_ids, segment_ids, input_mask):
        '''
        sentances -> word embedding -> lstm -> MLP -> feats
        '''

        trans_seq_out, _ = self.pre_model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask)#, output_all_encoded_layers=False)
        if self.use_alignment:
            aligning = getattr(self, 'aligning_{}'.format(lang_dict[lang.item()]))
            trans_seq_out = aligning(trans_seq_out)
        trans_seq_out = self.dropout(trans_seq_out)
        trans_feats = self.hidden2label(trans_seq_out)
        return trans_feats

    def _score_sentence(self, feats, label_ids):
        '''
        Gives the score of a provided label sequence
        p(X=w1:t,Zt=tag1:t)=...p(Zt=tag_t|Zt-1=tag_t-1)p(xt|Zt=tag_t)...
        '''

        # T = self.max_seq_length
        T = feats.shape[1]
        batch_size = feats.shape[0]

        batch_transitions = self.transitions.expand(batch_size, self.num_labels, self.num_labels)
        batch_transitions = batch_transitions.flatten(1)

        score = torch.zeros((feats.shape[0], 1)).to(self.device)
        # the 0th node is start_label->start_word,the probability of them=1. so t begin with 1.
        for t in range(1, T):
            score = score + \
                    batch_transitions.gather(-1, (label_ids[:, t] * self.num_labels + label_ids[:, t - 1]).view(-1, 1)) \
                    + feats[:, t].gather(-1, label_ids[:, t].view(-1, 1)).view(-1, 1)
        return score

    def _viterbi_decode(self, feats):
        '''
        Max-Product Algorithm or viterbi algorithm, argmax(p(z_0:t|x_0:t))
        '''

        # T = self.max_seq_length
        T = feats.shape[1]
        batch_size = feats.shape[0]

        # batch_transitions=self.transitions.expand(batch_size,self.num_labels,self.num_labels)

        log_delta = torch.Tensor(batch_size, 1, self.num_labels).fill_(-10000.).to(self.device)
        log_delta[:, 0, self.start_label_id] = 0

        # psi is for the vaule of the last latent that make P(this_latent) maximum.
        psi = torch.zeros((batch_size, T, self.num_labels), dtype=torch.long)#.to(self.device)  # psi[0]=0000 useless
        for t in range(1, T):
            # delta[t][k]=max_z1:t-1( p(x1,x2,...,xt,z1,z2,...,zt-1,zt=k|theta) )
            # delta[t] is the max prob of the path from  z_t-1 to z_t[k]
            log_delta, psi[:, t] = torch.max(self.transitions + log_delta, -1)
            # psi[t][k]=argmax_z1:t-1( p(x1,x2,...,xt,z1,z2,...,zt-1,zt=k|theta) )
            # psi[t][k] is the path choosed from z_t-1 to z_t[k],the value is the z_state(is k) index of z_t-1
            log_delta = (log_delta + feats[:, t]).unsqueeze(1)

        # trace back
        path = torch.zeros((batch_size, T), dtype=torch.long)#.to(self.device)

        # max p(z1:t,all_x|theta)
        max_logLL_allz_allx, path[:, -1] = torch.max(log_delta.squeeze(), -1)

        for t in range(T - 2, -1, -1):
            # choose the state of z_t according the state choosed of z_t+1.
            path[:, t] = psi[:, t + 1].gather(-1, path[:, t + 1].view(-1, 1)).squeeze()

        return max_logLL_allz_allx, path

    def forward(self, lang, sentence_features, labels, input_ids, segment_ids, input_mask, label_ids):
        if self.use_multi_task:
            sentence_embeddings = self._get_sentence_embeddings(sentence_features)
            intrinsic_loss = TripletLoss()(sentence_embeddings, labels)
        else:
            intrinsic_loss = None
        trans_feats = self._get_trans_features(lang, input_ids, segment_ids, input_mask)
        forward_score = self._forward_alg(trans_feats)
        # p(X=w1:t,Zt=tag1:t)=...p(Zt=tag_t|Zt-1=tag_t-1)p(xt|Zt=tag_t)...
        gold_score = self._score_sentence(trans_feats, label_ids)
        # - log[ p(X=w1:t,Zt=tag1:t)/p(X=w1:t) ] = - log[ p(Zt=tag1:t|X=w1:t) ]
        # Find the best path, given the features.
        score, label_seq_ids = self._viterbi_decode(trans_feats)
        #print("torch.mean(forward_score - forward_score).device:", torch.mean(forward_score - forward_score).device)
        #print("score.device:", score.device)
        #print("label_seq_ids.device:", label_seq_ids.device)

        #print("torch.mean(forward_score - forward_score):", torch.mean(forward_score - forward_score))
        #print("intrinsic_loss:", intrinsic_loss)
        #print("score:", score)
        #print("label_seq_ids:", label_seq_ids)
        return torch.mean(forward_score - gold_score), intrinsic_loss, score, label_seq_ids.to(self.device)

