"""
Code adapted from JMEE
"""
import os
cwd = os.getcwd()
import sys
from pathlib import Path
sys.path.append(os.path.join(Path(os.getcwd()).parent, "OnlineAlignment"))
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import numpy as np
from CRF import *

from pytorch_pretrained_bert import BertModel
from data_utils import *
from consts import NONE
from utils import find_triggers
from consts import *
from GCN import *
from SelfAttention import *
from losses import *

lang_dict = {0: "en", 1: "ar", 2: "es", 3: "zh"}

class Net(nn.Module):
    def __init__(self, use_multi_tasking, aligning_files, pre_model, pooling_choice, use_pos, use_ent, use_gcn, use_pred_trig, use_crf_trig, use_crf_arg,
                 psemb_size, gcn_layers=3, trigger_size=None, entity_size=None, all_postags=None, postag_embedding_dim=50,
                 argument_size=None, entity_embedding_dim=50, posi_embedding_dim = 50, device=torch.device("cpu")):
        super().__init__()
        self.use_multi_tasking = use_multi_tasking
        self.use_pos = use_pos
        self.use_ent = use_ent
        self.use_gcn = use_gcn
        self.use_pred_trig = use_pred_trig
        self.use_crf_trig = use_crf_trig
        self.use_crf_arg = use_crf_arg
        self.gcn_layers = gcn_layers

        self.trans_model = pre_model
        self.hidden_size = self.trans_model.config.hidden_size

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

        self.entity_embed = MultiLabelEmbeddingLayer(num_embeddings=entity_size, embedding_dim=entity_embedding_dim, device=device)
        self.postag_embed = nn.Embedding(num_embeddings=all_postags, embedding_dim=postag_embedding_dim)
        self.posi_embed = nn.Embedding(num_embeddings=psemb_size, embedding_dim=posi_embedding_dim)


        hidden_size = 768
        if use_pos:
            hidden_size += postag_embedding_dim

        if use_ent:
            hidden_size += entity_embedding_dim

        if use_gcn:
            hidden_size += posi_embedding_dim

        # GCN
        self.gcns = nn.ModuleList()
        for i in range(self.gcn_layers):
            gcn = GraphConvolution(in_features=hidden_size,
                                   out_features=hidden_size,
                                   edge_types=3,
                                   dropout=0.5 if i != 3 - 1 else None,
                                   use_bn=True,
                                   device=device)
            self.gcns.append(gcn)

        # Highway
        self.hws = nn.ModuleList()
        for i in range(3):
            hw = HighWay(size=hidden_size, dropout_ratio=0.5)
            self.hws.append(hw)

        self.sa = AttentionLayer(D=hidden_size, H=300, return_sequences=False)

        self.fc1 = nn.Sequential(
            # nn.Dropout(0.5),
            nn.Linear(hidden_size, hidden_size, bias=True),
            nn.ReLU(),
        )
        hidden2label = nn.Linear(hidden_size, trigger_size)
        hidden2arg = nn.Linear(hidden_size * 2, argument_size)
        self.fc_trigger = nn.Sequential(
            nn.Linear(hidden_size, trigger_size),
        )
        self.fc_argument = nn.Sequential(
            nn.Linear(hidden_size * 2, argument_size),
        )
        self.device = device

        kwargs = dict({'target_size': trigger_size, 'device': device, "hidden2label": hidden2label})
        self.tri_CRF1 = CRF(**kwargs)

        kwargs_a = dict({'target_size': argument_size, 'device': device, "hidden2label": hidden2arg})
        self.arg_CRF = [CRF(**kwargs_a) for _ in range(len(TRIGGERS))]

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

        self.arguments_2d = []
        self.adjm = []
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)


    def get_sentence_positional_feature(self, BATCH_SIZE, SEQ_LEN):
        positions = [[abs(j) for j in range(-i, SEQ_LEN - i)] for i in range(SEQ_LEN)]  # list [SEQ_LEN, SEQ_LEN]
        positions = [torch.LongTensor(position) for position in positions]  # list of tensors [SEQ_LEN]
        positions = [torch.cat([position] * BATCH_SIZE).resize_(BATCH_SIZE, position.size(0))
                     for position in positions]  # list of tensors [BATCH_SIZE, SEQ_LEN]
        return positions

    def _get_sentence_embeddings(self, sentence_features):
        sentence_embeddings = {}
        for i, features in enumerate(sentence_features):
            if self.training:
                self.trans_model.train()
                lm_output, _ = self.trans_model(input_ids=features["input_ids"],
                                                  token_type_ids=features["token_type_ids"],
                                                  attention_mask=features['input_mask'])

            else:
                self.trans_model.eval()
                with torch.no_grad():
                    lm_output, _ = self.trans_model(input_ids=features["input_ids"],
                                                      token_type_ids=features["token_type_ids"],
                                                      attention_mask=features['input_mask'])

            cls_token = lm_output[:, 0, :]

            ## 2. Pooling layer
            output_vectors = []
            if self.pooling_mode_cls_token: # CLS Pooling
                output_vectors.append(cls_token)

            if self.pooling_mode_max_tokens: # Max Pooling
                input_mask_expanded = features['input_mask'].unsqueeze(-1).expand(lm_output.size()).float()
                lm_output[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
                max_over_time = torch.max(lm_output, 1)[0]
                output_vectors.append(max_over_time)

            if self.pooling_mode_mean_tokens or self.pooling_mode_mean_sqrt_len_tokens:
                input_mask_expanded = features['input_mask'].unsqueeze(-1).expand(lm_output.size()).float()
                sum_embeddings = torch.sum(lm_output * input_mask_expanded, 1)

                #If tokens are weighted (by WordWeights layer), feature 'token_weights_sum' will be present
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

    def _predict_triggers(self, lang, tokens_x_2d, entities_x_3d, postags_x_2d, head_indexes_2d, triggers_y_2d):

        if self.training:
            self.trans_model.train()
            encoded_layers, _ = self.trans_model(tokens_x_2d)
            enc = encoded_layers #encoded_layers[-1]

        else:
            self.trans_model.eval()
            with torch.no_grad():
                encoded_layers, _ = self.trans_model(tokens_x_2d)
                enc = encoded_layers#encoded_layers[-1]

        if self.use_alignment:
            aligning = getattr(self, 'aligning_{}'.format(lang.item()))
            enc = aligning(enc)

        x = enc
        if self.use_ent:
            entity_x_2d = self.entity_embed(entities_x_3d)
            x = torch.cat([x, entity_x_2d], 2)

        if self.use_pos:
            postags_x_2d = self.postag_embed(postags_x_2d)
            x = torch.cat([x, postags_x_2d], 2)

        BATCH_SIZE = tokens_x_2d.shape[0]
        SEQ_LEN = x.size()[1]
        if self.use_gcn:
            """
            mask = np.zeros(shape=word_sequence.size(), dtype=np.uint8)
            for i in range(word_sequence.size()[0]):
                s_len = int(x_len[i])
                mask[i, 0:s_len] = np.ones(shape=(s_len), dtype=np.uint8)

            mask = torch.ByteTensor(mask).to(self.device)
            """


            positional_sequences = self.get_sentence_positional_feature(BATCH_SIZE, SEQ_LEN)
            adjm = torch.stack([torch.sparse.FloatTensor(torch.LongTensor(adjmm[0]),
                                                         torch.FloatTensor(adjmm[1]),
                                                         torch.Size([3, SEQ_LEN, SEQ_LEN])).to_dense() for adjmm in self.adjm])

            adjm = adjm.to(self.device)
            xx = []
            for i in range(SEQ_LEN):
                posi_x_2d = self.posi_embed(positional_sequences[i].to(self.device))
                gcn_in = torch.cat([x, posi_x_2d], 2)

                # gcns
                for i in range(self.gcn_layers):
                    x_gcn = self.gcns[i](gcn_in, adjm) #+ self.hws[i](gcn_in)  # (batch_size, seq_len, d')

                # self attention
                xx.append(self.sa(x_gcn, mask=None))  # (batch_size, d')

            x = torch.stack(xx, dim=1)

            print("AFTER POSI x_cat.shape:", x.shape)


        batch_size = tokens_x_2d.shape[0]

        for i in range(batch_size):
            x[i] = torch.index_select(x[i], 0, head_indexes_2d[i])

        trigger_logits = self.fc_trigger(x)

        if self.use_crf_trig:
            """
            xlen = [max(x) for x in head_indexes_2d]
            batch_size = tokens_x_2d.shape[0]
            SEQ_LEN = x.size()[1]
            mask = np.zeros(shape=[batch_size, SEQ_LEN], dtype=np.bool)
            for i in range(len(xlen)):
                # Slicing, changing position and removing
                mask[i, :xlen[i]] = True
            mask = torch.BoolTensor(mask).to(self.device)
            
            """

            trigger_logits = nn.functional.leaky_relu_(trigger_logits)
            _, trigger_hat_2d = self.tri_CRF1.forward(feats=trigger_logits)
        else:
            trigger_hat_2d = trigger_logits.argmax(-1)

        argument_hidden, argument_keys = [], []
        for i in range(batch_size):
            candidates = self.arguments_2d[i]['candidates']
            golden_entity_tensors = {}

            for j in range(len(candidates)):
                e_start, e_end, e_type_str = candidates[j]
                golden_entity_tensors[candidates[j]] = x[i, e_start:e_end, ].mean(dim=0)

            if self.use_pred_trig:
                used_triggers = find_triggers([idx2trigger[trigger] for trigger in trigger_hat_2d[i].tolist()])
            else:
                used_triggers = find_triggers([idx2trigger[trigger] for trigger in triggers_y_2d[i].tolist()])

            for used_trigger in used_triggers:
                t_start, t_end, t_type_str = used_trigger
                event_tensor = x[i, t_start:t_end, ].mean(dim=0)
                for j in range(len(candidates)):
                    e_start, e_end, e_type_str = candidates[j]
                    entity_tensor = golden_entity_tensors[candidates[j]]

                    argument_hidden.append(torch.cat([event_tensor, entity_tensor]))
                    argument_keys.append((i, t_start, t_end, t_type_str, e_start, e_end, e_type_str))

        if self.use_crf_trig:
            xlen = [max(x) for x in head_indexes_2d]
            batch_size = tokens_x_2d.shape[0]
            SEQ_LEN = tokens_x_2d.shape[1]
            mask = np.zeros(shape=[batch_size, SEQ_LEN], dtype=np.bool)
            for k in range(len(xlen)):
                # Slicing, changing position and removing
                mask[k, :xlen[k]] = True
            mask = torch.BoolTensor(mask).to(self.device)

            trigger_loss = self.tri_CRF1.neg_log_likelihood_loss(feats=trigger_logits, mask=mask, tags=triggers_y_2d)
        else:
            trigger_logits = trigger_logits.view(-1, trigger_logits.shape[-1])
            trigger_loss = criterion(trigger_logits, triggers_y_2d.view(-1))


        return trigger_loss, trigger_logits, triggers_y_2d, trigger_hat_2d, argument_hidden, argument_keys

    def _predict_arguments(self, argument_hidden, argument_keys):
        argument_hidden = torch.stack(argument_hidden)
        argument_logits = self.fc_argument(argument_hidden)
        argument_hat_1d = argument_logits.argmax(-1)

        arguments_y_1d = []
        for i, t_start, t_end, t_type_str, e_start, e_end, e_type_str in argument_keys:
            a_label = argument2idx[NONE]
            if (t_start, t_end, t_type_str) in self.arguments_2d[i]['events']:
                for (a_start, a_end, a_type_idx) in self.arguments_2d[i]['events'][(t_start, t_end, t_type_str)]:
                    if e_start == a_start and e_end == a_end:
                        a_label = a_type_idx
                        break
            arguments_y_1d.append(a_label)

        arguments_y_1d = torch.LongTensor(arguments_y_1d).to(self.device)

        batch_size = len(self.arguments_2d)
        argument_hat_2d = [{'events': {}} for _ in range(batch_size)]
        for (i, st, ed, event_type_str, e_st, e_ed, entity_type), a_label in zip(argument_keys, argument_hat_1d.cpu().numpy()):
            if a_label == argument2idx[NONE]:
                continue
            if (st, ed, event_type_str) not in argument_hat_2d[i]['events']:
                argument_hat_2d[i]['events'][(st, ed, event_type_str)] = []
            argument_hat_2d[i]['events'][(st, ed, event_type_str)].append((e_st, e_ed, a_label))

        return argument_logits, arguments_y_1d, argument_hat_1d, argument_hat_2d

    def forward(self,  features, labels, lang, tokens_x_2d, entities_x_3d, postags_x_2d, head_indexes_2d, triggers_y_2d):

        # Instrinsic loss
        print("self.use_multi_tasking:", self.use_multi_tasking)
        if self.use_multi_tasking:
            sentence_embeddings =  self._get_sentence_embeddings(features)

            intrinsic_loss =  TripletLoss()(sentence_embeddings, labels)
        else:
            intrinsic_loss = 0

        trigger_loss, trigger_logits, triggers_y_2d, trigger_hat_2d, argument_hidden, argument_keys \
            = self._predict_triggers(lang, tokens_x_2d, entities_x_3d, postags_x_2d, head_indexes_2d, triggers_y_2d)

        if len(argument_keys) > 0:
            argument_logits, arguments_y_1d, argument_hat_1d, argument_hat_2d = self._predict_arguments(argument_hidden, argument_keys)
            argument_loss = self.criterion(argument_logits, arguments_y_1d)
            event_loss = trigger_loss + 2 * argument_loss
        else:
            argument_loss = 100
            event_loss = trigger_loss

        return intrinsic_loss, triggers_y_2d, trigger_hat_2d, trigger_loss, argument_loss, event_loss

# Reused from https://github.com/lx865712528/EMNLP2018-JMEE
class MultiLabelEmbeddingLayer(nn.Module):
    def __init__(self,
                 num_embeddings=None, embedding_dim=None,
                 dropout=0.5, padding_idx=0,
                 max_norm=None, norm_type=2,
                 device=torch.device("cpu")):
        super(MultiLabelEmbeddingLayer, self).__init__()

        self.matrix = nn.Embedding(num_embeddings=num_embeddings,
                                   embedding_dim=embedding_dim,
                                   padding_idx=padding_idx,
                                   max_norm=max_norm,
                                   norm_type=norm_type)
        self.dropout = dropout
        self.device = device
        self.to(device)

    def forward(self, x):
        batch_size = len(x)
        seq_len = len(x[0])
        x = [self.matrix(torch.LongTensor(x[i][j]).to(self.device)).sum(0)
             for i in range(batch_size)
             for j in range(seq_len)]
        x = torch.stack(x).view(batch_size, seq_len, -1)

        if self.dropout is not None:
            return F.dropout(x, p=self.dropout, training=self.training)
        else:
            return x
