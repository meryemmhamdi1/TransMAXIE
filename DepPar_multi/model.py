import torch
from torch import nn
import time
import torchtext
import numpy as np

import random
from transformers import *

class EdgeFactoredParser(nn.Module):

    def __init__(self, fields, word_emb_dim, pos_emb_dim,
                 rnn_size, rnn_depth, mlp_size, update_pretrained=False):
        super().__init__()

        word_field = fields[0][1]
        pos_field = fields[1][1]
        deprel_field = fields[3][1]

        rel_size = len(deprel_field.vocab)

        # Sentence encoder module
        self.encoder = RNNEncoder(word_field, word_emb_dim, pos_field, pos_emb_dim, rnn_size, rnn_depth,
                                  update_pretrained)

        # Edge scoring module
        self.edge_scorer = DeepBiaffineScorer(2*rnn_size, mlp_size, out_size=1, bias_x=True, bias_y=False)

        # Relation scoring module
        self.rel_scorer = DeepBiaffineScorer(2*rnn_size, mlp_size, out_size=rel_size, bias_x=True, bias_y=True)

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

    def forward(self, words, postags, heads, rels, evaluate=False):

        if self.training:
            # If we are training, apply the word/tag dropout to the word and tag tensors.
            words, postags = self.word_tag_dropout(words, postags, 0.25)

        encoded = self.encoder(words, postags)
        #print("encoded.shape:", encoded.shape)

        edge_scores = self.edge_scorer(encoded)
        rel_scores = self.rel_scorer(encoded).permute(0, 2, 3, 1)

        # We don't want to evaluate the loss or attachment score for the positions
        # where we have a padding token. So we create a mask that will be zero for those
        # positions and one elsewhere.
        pad_mask = (words != self.pad_id)#.float()

        #edge_loss = self.compute_loss_edges(edge_scores, heads, pad_mask)
        #rel_loss = self.compute_loss_rels(rel_scores, rels, pad_mask)

        #loss = edge_loss + rel_loss
        loss = self.compute_loss(edge_scores, heads, rel_scores, rels, pad_mask)

        if evaluate:
            n_errors_edges, n_tokens_edges = self.evaluate_edges(edge_scores, heads, pad_mask)
            n_errors_rels, n_tokens_rels = self.evaluate_rels(rel_scores, rels, pad_mask)
            return loss, n_errors_edges, n_tokens_edges, n_errors_rels, n_tokens_rels
        else:
            return loss

    def compute_loss(self, edge_scores, heads, rel_scores, rels, pad_mask):

        edge_scores, heads, rel_scores, rels  = edge_scores[pad_mask], heads[pad_mask], rel_scores[pad_mask], rels[pad_mask]
        #print("edge_scores.shape:", edge_scores.shape)
        #print("heads.shape:", heads.shape)
        #print("torch.arange(len(heads):", torch.arange(len(heads)).shape)

        rel_scores = rel_scores[torch.arange(len(heads)), heads]
        edge_loss = self.loss(edge_scores, heads)
        rel_loss = self.loss(rel_scores, rels)
        loss = edge_loss + rel_loss

        avg_loss = loss/ pad_mask.sum()
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

    def evaluate_edges(self, edge_scores, heads, pad_mask):
        n_sentences, n_words, _ = edge_scores.shape
        edge_scores = edge_scores.view(n_sentences*n_words, n_words)
        heads = heads.view(n_sentences*n_words)
        pad_mask = pad_mask.view(n_sentences*n_words)
        n_tokens = pad_mask.sum()
        predictions = edge_scores.argmax(dim=1)
        n_errors = (predictions != heads).float().dot(pad_mask)
        return n_errors.item(), n_tokens.item()

    def evaluate_rels(self, rels_scores, rels, pad_mask):
        n_sentences, n_words, _ = rels_scores.shape
        rels_scores = rels_scores.view(n_sentences*n_words, n_words)
        heads = rels.view(n_sentences*n_words)
        pad_mask = pad_mask.view(n_sentences*n_words)
        n_tokens = pad_mask.sum()
        predictions = rels_scores.argmax(dim=1)
        n_errors = (predictions != heads).float().dot(pad_mask)
        return n_errors.item(), n_tokens.item()

    def predict(self, words, postags):
        # This method is used to parse a sentence when the model has been trained.
        encoded = self.encoder(words, postags)
        edge_scores = self.edge_scorer(encoded)
        return edge_scores.argmax(dim=2)


class DeepBiaffineScorer(nn.Module):
    def __init__(self, rnn_size, mlp_size, out_size, bias_x, bias_y):
        super().__init__()

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
        H_head = self.head_mlp(sentence_repr)
        H_dep = self.dep_mlp(sentence_repr)
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

class RNNEncoder(nn.Module):

    def __init__(self, word_field, word_emb_dim, pos_field, pos_emb_dim, rnn_size, rnn_depth, update_pretrained):
        super().__init__()

        self.word_embedding = nn.Embedding(len(word_field.vocab), word_emb_dim)
        # If we're using pre-trained word embeddings, we need to copy them.
        if word_field.vocab.vectors is not None:
            self.word_embedding.weight = nn.Parameter(word_field.vocab.vectors,
                                                      requires_grad=update_pretrained)

        # POS-tag embeddings will always be trained from scratch.
        self.pos_embedding = nn.Embedding(len(pos_field.vocab), pos_emb_dim)

        self.rnn = nn.LSTM(input_size=word_emb_dim+pos_emb_dim, hidden_size=rnn_size, batch_first=True,
                           bidirectional=True, num_layers=rnn_depth)

    def forward(self, words, postags):
        # Look u
        word_emb = self.word_embedding(words)
        pos_emb = self.pos_embedding(postags)
        word_pos_emb = torch.cat([word_emb, pos_emb], dim=2)

        #print("word_pos_emb.shape:", word_pos_emb.shape)

        rnn_out, _ = self.rnn(word_pos_emb)

        #print("rnn_out.shape:", rnn_out.shape)

        return rnn_out

class BiaffineScorer(nn.Module):

    def __init__(self, rnn_size, mlp_size, out_size, bias_W, bias_b):
        super().__init__()

        mlp_activation = nn.ReLU()

        # The two MLPs that we apply to the RNN output before the biaffine scorer.
        self.head_mlp = nn.Sequential(nn.Linear(rnn_size, mlp_size), mlp_activation)
        self.dep_mlp = nn.Sequential(nn.Linear(rnn_size, mlp_size), mlp_activation)

        # Weights for the biaffine part of the model.


        #print("rnn_size:", rnn_size)
        self.W = nn.Linear(mlp_size, mlp_size, bias=bias_W)

        self.b = nn.Linear(mlp_size, 1, bias=bias_b)

    def forward(self, sentence_repr):

        # MLPs applied to the RNN output: equations 4 and 5 in the paper.
        H_head = self.head_mlp(sentence_repr)
        #print("++++++++++++++++++++++++++++++++++++++++")
        #print("sentence_repr.shape:", sentence_repr.shape)
        #print("self.weight.shape:", self.weight.shape)
        #print("H_head.shape:", H_head.shape)
        H_dep = self.dep_mlp(sentence_repr)
        #print("H_dep.shape:", H_dep.shape)

        # Computing the edge scores for all edges using the biaffine model.
        # This corresponds to equation 9 in the paper. For readability we implement this
        # in a step-by-step fashion.
        Hh_W = self.W(H_head)
        #print("Hh_W.shape:", Hh_W.shape)
        Hh_W_Ha = H_dep.matmul(Hh_W.transpose(1, 2))
        #print("Hh_W_Ha.shape:", Hh_W_Ha.shape)
        Hh_b = self.b(H_head).transpose(1, 2)
        #print("Hh_b.shape:", Hh_b.shape)
        sum_ = Hh_b + Hh_W_Ha
        #print("sum_.shape:", sum_.shape)
        sum_ = Hh_W_Ha + Hh_b
        #print("sum_.shape:", sum_.shape)
        return sum_


