import torch
from torch import nn
import time
import torchtext
import numpy as np

import random
from transformers import *
from alg import *

import os
cwd = os.getcwd()
import sys
from pathlib import Path

sys.path.append(os.path.join(Path(os.getcwd()).parent, "OnlineAlignment"))

from evaluators import *
from losses import *

lang_dict = {0: "en", 1: "ar", 2: "de", 3: "es", 4: "zh"}

class SharedDropout(nn.Module):

    def __init__(self, p=0.5, batch_first=True):
        super(SharedDropout, self).__init__()

        self.p = p
        self.batch_first = batch_first

    def extra_repr(self):
        s = f"p={self.p}"
        if self.batch_first:
            s += f", batch_first={self.batch_first}"

        return s

    def forward(self, x):
        if self.training:
            if self.batch_first:
                mask = self.get_mask(x[:, 0], self.p)
            else:
                mask = self.get_mask(x[0], self.p)
            x *= mask.unsqueeze(1) if self.batch_first else mask

        return x

    @staticmethod
    def get_mask(x, p):
        mask = x.new_empty(x.shape).bernoulli_(1 - p)
        mask = mask / (1 - p)

        return mask

class IndependentDropout(nn.Module):

    def __init__(self, p=0.5):
        super(IndependentDropout, self).__init__()

        self.p = p

    def extra_repr(self):
        return f"p={self.p}"

    def forward(self, *items):
        if self.training:
            masks = [x.new_empty(x.shape[:2]).bernoulli_(1 - self.p)
                     for x in items]
            total = sum(masks)
            scale = len(items) / total.max(torch.ones_like(total))
            masks = [mask * scale for mask in masks]
            items = [item * mask.unsqueeze(dim=-1)
                     for item, mask in zip(items, masks)]

        return items

class EdgeFactoredParser(nn.Module):

    def __init__(self, use_multi_task, use_eisner, aligning_files, pooling_choice, fields, trans_model, use_rnn, pos_emb_dim, word_emb_dim, use_word_embed, rnn_size, rnn_depth, edge_mlp_size, rel_mlp_size):
        super().__init__()

        word_field = fields[0][1]
        pos_field = fields[1][1]
        deprel_field = fields[3][1]

        rel_size = len(deprel_field.vocab)
        self.use_eisner = use_eisner
        self.use_multi_task = use_multi_task

        self.dropout_ind = torch.nn.Dropout(0.33)#IndependentDropout(p=0.33)

        # Sentence encoder
        self.use_alignment = False
        if aligning_files is not None:
            self.use_alignment = True

        self.encoder = BERTEncoder(self.dropout_ind, self.use_alignment, aligning_files, pooling_choice, trans_model, word_field, word_emb_dim, pos_field, pos_emb_dim, use_word_embed, use_rnn, rnn_size, rnn_depth)
        self.use_rnn = use_rnn
        if use_rnn:
            in_size = 2 * rnn_size
        else:
            in_size = self.encoder.hidden_size + pos_emb_dim
            if use_word_embed:
                in_size += word_emb_dim


        # Edge scoring module
        self.edge_scorer = DeepBiaffineScorer(self.dropout_ind, in_size, edge_mlp_size, out_size=1, bias_x=True, bias_y=False)

        # Relation scoring module
        self.rel_scorer = DeepBiaffineScorer(self.dropout_ind, in_size, rel_mlp_size, out_size=rel_size, bias_x=True, bias_y=True)

        # To deal with the padding positions later, we need to know the
        # encoding of the padding dummy word.
        self.pad_id = word_field.vocab.stoi[word_field.pad_token]

        # Loss function that we will use during training.
        self.loss = torch.nn.CrossEntropyLoss()

    def word_tag_dropout(self, words, postags, p_drop):
        # Randomly replace some of the positions in the word and postag tensors with a zero.
        # This solution is a bit hacky because we assume that zero corresponds to the "unknown" token.
        w_dropout_mask = (torch.rand(size=words.shape, device=words.device) > p_drop).long()
        p_dropout_mask = (torch.rand(size=words.shape, device=words.device) > p_drop).long()
        return words*w_dropout_mask, postags*p_dropout_mask

    def forward(self, lang, sentence_features, labels, words, postags, input_ids, token_type_ids, input_mask, predict_mask, heads, deprels, evaluate=False):

        if self.training:
            # If we are training, apply the word/tag dropout to the word and tag tensors.
            p_dropout_mask = (torch.rand(size=postags.shape, device=postags.device) > 0.25).long()
            postags = postags*p_dropout_mask

        encoded, sentence_embeddings = self.encoder(lang, self.use_multi_task, sentence_features, words, postags, input_ids, token_type_ids, input_mask, predict_mask, evaluate)

        if sentence_embeddings is None:
            intrinsic_loss = None
        else:
            intrinsic_loss = TripletLoss()(sentence_embeddings, labels)


        edge_scores = self.edge_scorer(encoded)
        rel_scores = self.rel_scorer(encoded).permute(0, 2, 3, 1)

        # We don't want to evaluate the loss or attachment score for the positions
        # where we have a padding token. So we create a mask that will be zero for those
        # positions and one elsewhere.
        pad_mask = (words != self.pad_id)#.float()

        #edge_loss = self.compute_loss_edges(edge_scores, heads, pad_mask)
        #rel_loss = self.compute_loss_rels(rel_scores, rels, pad_mask)

        #loss = edge_loss + rel_loss
        loss = self.compute_loss(edge_scores, heads, rel_scores, deprels, pad_mask)

        if evaluate:
            arc_preds, rel_preds = self.decode(edge_scores, rel_scores, pad_mask)

            edge_mask = arc_preds.eq(heads)[pad_mask]
            rel_mask = rel_preds.eq(deprels)[pad_mask] & edge_mask

            total = len(edge_mask)
            correct_edge = edge_mask.sum().item()
            correct_rels = rel_mask.sum().item()

            #n_errors_edges, n_tokens_edges = self.evaluate_edges(edge_scores, heads, pad_mask.float())
            #n_errors_rels, n_tokens_rels = self.evaluate_rels(rel_scores, deprels, pad_mask.float())
            return intrinsic_loss, loss, total, correct_edge, correct_rels
        else:
            return intrinsic_loss, loss

    def compute_loss(self, edge_scores, heads, rel_scores, rels, pad_mask):

        edge_scores, heads, rel_scores, rels  = edge_scores[pad_mask], heads[pad_mask], rel_scores[pad_mask], rels[pad_mask]
        #print("edge_scores.shape:", edge_scores.shape)
        #print("heads.shape:", heads.shape)
        #print("torch.arange(len(heads):", torch.arange(len(heads)).shape)

        rel_scores = rel_scores[torch.arange(len(heads)), heads]
        edge_loss = self.loss(edge_scores, heads)
        rel_loss = self.loss(rel_scores, rels)
        loss = edge_loss + rel_loss

        avg_loss = loss
        #print("avg_loss:", avg_loss)
        return avg_loss

    def compute_loss_edges(self, edge_scores, heads, pad_mask):
        #print("------------ COMPUTE LOSS EDGES -------------")
        #print("edge_scores:", edge_scores)
        #print("edge_scores.shape:", edge_scores.shape)
        #print("pad_mask.shape:", pad_mask.shape)
        #print("heads.shape:", heads.shape)
        #print("pad_mask:", pad_mask)
        n_sentences, n_words, _ = edge_scores.shape
        edge_scores = edge_scores.view(n_sentences*n_words, n_words)
        #print("After view: edge_scores:", edge_scores.shape)
        heads = heads.view(n_sentences*n_words)
        #print("After view: heads:", heads.shape)
        pad_mask = pad_mask.view(n_sentences*n_words)
        #print("After view: pad_mask:", pad_mask.shape)
        loss = self.loss(edge_scores, heads)
        #print("loss:", loss.shape)
        avg_loss = loss.dot(pad_mask) / pad_mask.sum()
        #print("avg_loss:", torch.mean(avg_loss))
        return avg_loss

    def compute_loss_rels(self, rel_scores, rels, pad_mask):
        #print("------------ COMPUTE LOSS RELS -------------")
        n_sentences, n_rels, n_words, _ = rel_scores.shape
        #print("rel_scores.shape:", rel_scores.shape)
        #print("rel_scores:", rel_scores)
        rel_scores = rel_scores.view(n_sentences*n_words*n_rels, n_words)
        #print("After view rel_scores.shape:", rel_scores.shape)
        #print("pad_mask.shape:", pad_mask.shape)
        #print("rels.shape:", rels.shape)
        rels = rels.view(n_sentences*n_words)
        #print("After view: rels:", rels.shape)
        pad_mask = pad_mask.view(n_sentences*n_words)
        #print("After view: pad_mask:", pad_mask.shape)
        loss = self.loss(rel_scores, rels)
        avg_loss = loss.dot(pad_mask) / pad_mask.sum()
        return avg_loss

    def decode(self, edge_scores, rel_scores, pad_mask):
        if self.use_eisner:
            arc_preds = eisner(edge_scores, pad_mask)
        else:
            arc_preds = edge_scores.argmax(-1)
        rel_preds = rel_scores.argmax(-1)
        rel_preds = rel_preds.gather(-1, arc_preds.unsqueeze(-1)).squeeze(-1)

        return arc_preds, rel_preds

    def evaluate_edges(self, edge_scores, heads, pad_mask):
        #print("edge_scores.shape:", edge_scores.shape)
        n_sentences, n_words, _ = edge_scores.shape
        edge_scores = edge_scores.view(n_sentences*n_words, n_words)
        heads = heads.view(n_sentences*n_words)
        pad_mask = pad_mask.view(n_sentences*n_words)
        n_tokens = pad_mask.sum()
        predictions = edge_scores.argmax(dim=1)
        #print("predictions.shape:", predictions.shape)
        #print("heads.shape:", heads.shape)
        n_errors = (predictions != heads).float().dot(pad_mask)
        return n_errors.item(), n_tokens.item()

    def evaluate_rels(self, rels_scores, rels, pad_mask):
        #print("rels_scores.shape:", rels_scores.shape)
        #print("rels.shape:", rels.shape)
        n_sentences, n_words,_, n_rels = rels_scores.shape
        rels_scores = rels_scores.view(n_sentences*n_words*n_rels, n_words)
        rels = rels.view(n_sentences*n_words)
        pad_mask = pad_mask.view(n_sentences*n_words)
        n_tokens = pad_mask.sum()
        predictions = rels_scores.argmax(dim=-1)
        n_errors = (predictions != rels).float().dot(pad_mask)
        return n_errors.item(), n_tokens.item()

    def predict(self, words, postags):
        # This method is used to parse a sentence when the model has been trained.
        encoded = self.encoder(words, postags)
        edge_scores = self.edge_scorer(encoded)
        return edge_scores.argmax(dim=2)


class BERTEncoder(nn.Module):

    def __init__(self, dropout_ind, use_alignment, aligning_files, pooling_choice, trans_model, word_field, word_emb_dim, pos_field, pos_emb_dim, use_word_embed, use_rnn, rnn_size, rnn_depth):
        super().__init__()

        # POS-tag embeddings will always be trained from scratch.
        self.dropout_ind = dropout_ind
        self.pos_embedding = nn.Embedding(len(pos_field.vocab), pos_emb_dim)
        self.word_embedding = nn.Embedding(len(word_field.vocab), word_emb_dim)

        self.bert_model = trans_model
        self.bert_model = self.bert_model.requires_grad_(True)
        self.hidden_size = self.bert_model.config.hidden_size
        self.n_layers = self.bert_model.config.num_hidden_layers

        self.use_alignment = use_alignment
        self.use_rnn = use_rnn
        self.use_word_embed = use_word_embed

        """
        rnn_input_size = self.hidden_size+pos_emb_dim
        if use_word_embed:
            rnn_input_size += word_emb_dim
            
        """
        rnn_input_size = pos_emb_dim + self.hidden_size
        if self.use_word_embed:
            rnn_input_size += word_emb_dim

        self.rnn = nn.LSTM(input_size=rnn_input_size, hidden_size=rnn_size, batch_first=True,
                           bidirectional=True, dropout=0.33, num_layers=rnn_depth)

        if self.use_alignment:
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
                        print("LOADING GD MATRICES")
                        aligning_matrix = torch.load(aligning_path, map_location=lambda storage, loc: storage)
                        aligning.load_state_dict(aligning_matrix)
                        print("aligning:", aligning)
                else:
                    aligning = torch.nn.Linear(self.hidden_size, self.hidden_size, bias=False)
                    aligning.weight = torch.nn.Parameter(aligning_matrix, requires_grad=False)

                print("AFTER aligning:", aligning)
                self.add_module(name, aligning)

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

    def _get_sentence_embeddings(self, sentence_features, evaluate):
        sentence_embeddings = {}
        for i, features in enumerate(sentence_features):
            if not evaluate:
                self.bert_model.train()
                lm_output, _ = self.bert_model(input_ids=features["input_ids"],
                                              token_type_ids=features["token_type_ids"],
                                              attention_mask=features['input_mask'])#, output_all_encoded_layers=False)

            else:
                self.bert_model.eval()
                with torch.no_grad():
                    lm_output, _ = self.bert_model(input_ids=features["input_ids"], token_type_ids=features["token_type_ids"],
                                                  attention_mask=features['input_mask'])#, output_all_encoded_layers=False)

            lm_output = self.dropout_ind(lm_output)

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

    def forward(self, lang, use_multi_task, sentence_features, words, postags, input_ids, token_type_ids, input_mask, predict_mask, evaluate):
        if use_multi_task:
            sentence_embeddings = self._get_sentence_embeddings(sentence_features, evaluate)
        else:
            sentence_embeddings = None

        pos_emb = self.pos_embedding(postags)
        pos_emb = self.dropout_ind(pos_emb)

        if evaluate:
            self.bert_model.eval()
            with torch.no_grad():
                bert_embed, _ = self.bert_model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=input_mask)
        else:
            self.bert_model.train()
            bert_embed, _ = self.bert_model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=input_mask)

        bert_embed = self.dropout_ind(bert_embed)

        if self.use_alignment:
            aligning = getattr(self, 'aligning_{}'.format(lang_dict[lang.item()]))
            bert_embed = aligning(bert_embed)

        embed_new = torch.zeros([pos_emb.shape[0], pos_emb.shape[1], self.hidden_size], dtype=torch.float).to(bert_embed.device)
        for i, batch in enumerate(predict_mask):
            k = 0
            for j, feat in enumerate(batch):
                if feat == 1:
                    embed_new[i][k] = bert_embed[i][j]
                    k += 1

        """                
        bert_embed = torch.tensor(bert_embed_np, dtype=torch.double, requires_grad=self.requires_grad).type('torch.FloatTensor')
    
        idx = np.argwhere(np.asarray(features["predict_mask"])>=0)

        print("idx:", list(idx[0]))
        if [0, 1] in idx:
            print("YES")


        embed_new = torch.zeros([pos_emb.shape[0], pos_emb.shape[1], self.hidden_size], dtype=torch.float)
        for i in range(0, pos_emb.shape[0]):
            k = 0
            for j, embed in enumerate(bert_embed[i]):
                if [i, j] in idx:
                    print("i:", i, " j:", j)
                    embed_new[i][k] = embed
                    k += 1

        #y = bert_embed[idx[0][0]][[idx[0][1]]]

        print(embed_new.shape)
        bert_embed = bert_embed.new_zeros(pos_emb.shape[0], bert_embed.shape[1], self.hidden_size)
        bert_embed = bert_embed.masked_scatter_(features["predict_mask"].unsqueeze(-1)>0, bert_embed)
        """

        #print("embed_new.grad:", embed_new.grad)
        #print("bert_embed.grad:", bert_embed.grad)

        #print("embed_new.shape:", embed_new.shape)

        #print("pos_emb:", pos_emb)
        #print("embed_new:", embed_new)


        word_pos_emb = torch.cat((embed_new, pos_emb), dim=2)

        if self.use_word_embed:
            word_emb = self.word_embedding(words)
            word_pos_emb = torch.cat((word_pos_emb, word_emb), dim=2)
            
        """
        #print("word_pos_emb.shape:", word_pos_emb.shape)
        word_emb = self.word_embedding(words)
        word_pos_emb = torch.cat((word_emb, pos_emb), dim=2)
        """
        if self.use_rnn:
            rnn_out, _ = self.rnn(word_pos_emb)

            return rnn_out, sentence_embeddings

        else:
            return word_pos_emb, sentence_embeddings

class DeepBiaffineScorer(nn.Module):
    def __init__(self, dropout, rnn_size, mlp_size, out_size, bias_x, bias_y):
        super().__init__()

        self.dropout = dropout
        mlp_activation = nn.ReLU()
        self.bias_x = bias_x
        self.bias_y = bias_y

        self.head_mlp = nn.Sequential(nn.Linear(rnn_size, mlp_size), mlp_activation)
        self.dep_mlp = nn.Sequential(nn.Linear(rnn_size, mlp_size), mlp_activation)

        self.weight = nn.Parameter(torch.Tensor(out_size, mlp_size + bias_x, mlp_size + bias_y))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.weight)

    def forward(self, sentence_repr):
        H_head = self.dropout(self.head_mlp(sentence_repr))
        H_dep = self.dropout(self.dep_mlp(sentence_repr))
        if self.bias_x:
            H_head = torch.cat((H_head, torch.ones_like(H_head[..., :1])), -1)
        if self.bias_y:
            H_dep = torch.cat((H_dep, torch.ones_like(H_dep[..., :1])), -1)

        #print("H_head.shape:", H_head.shape)
        #print("H_dep.shape:", H_dep.shape)
        #print("weight.shape:", self.weight.shape)
        # [batch_size, n_out, seq_len, seq_len]
        s = torch.einsum('bxi,oij,byj->boxy', H_head, self.weight, H_dep)
        # remove dim 1 if n_out == 1
        s = s.squeeze(1)

        return s

