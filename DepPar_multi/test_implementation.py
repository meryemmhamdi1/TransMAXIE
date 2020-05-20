import torch
from torch import nn
import time
import torchtext
import numpy as np

import random

from collections import defaultdict, Counter
from get_arguments import *


class BiaffineEdgeScorer(nn.Module):

    def __init__(self, rnn_size, mlp_size):
        super().__init__()

        mlp_activation = nn.ReLU()

        # The two MLPs that we apply to the RNN output before the biaffine scorer.
        self.head_mlp = nn.Sequential(nn.Linear(rnn_size, mlp_size), mlp_activation)
        self.dep_mlp = nn.Sequential(nn.Linear(rnn_size, mlp_size), mlp_activation)

        # Weights for the biaffine part of the model.
        self.W_arc = nn.Linear(mlp_size, mlp_size, bias=False)
        self.b_arc = nn.Linear(mlp_size, 1, bias=False)

    def forward(self, sentence_repr):

        # MLPs applied to the RNN output: equations 4 and 5 in the paper.
        H_arc_head = self.head_mlp(sentence_repr)
        H_arc_dep = self.dep_mlp(sentence_repr)

        # Computing the edge scores for all edges using the biaffine model.
        # This corresponds to equation 9 in the paper. For readability we implement this
        # in a step-by-step fashion.
        Hh_W = self.W_arc(H_arc_head)
        Hh_W_Ha = H_arc_dep.matmul(Hh_W.transpose(1, 2))
        Hh_b = self.b_arc(H_arc_head).transpose(1, 2)
        return Hh_W_Ha + Hh_b

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

        self.rnn = nn.LSTM(input_size=768+pos_emb_dim, hidden_size=rnn_size, batch_first=True,
                           bidirectional=True, num_layers=rnn_depth)

        self.bert_model = MODELS_dict["BertBaseMultilingualCased"][0].from_pretrained(MODELS_dict["BertBaseMultilingualCased"][2])

        self.bert_model = self.bert_model.requires_grad_(True)
        self.hidden_size = self.bert_model.config.hidden_size
        self.n_layers = self.bert_model.config.num_hidden_layers

    def forward(self, words, postags, input_ids, token_type_ids, input_mask, predict_mask, evaluate):

        if evaluate:
            self.bert_model.eval()
            with torch.no_grad():
                bert_embed, _ = self.bert_model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=input_mask)
        else:
            self.bert_model.train()
            bert_embed, _ = self.bert_model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=input_mask)

        pos_emb = self.pos_embedding(postags)
        embed_new = torch.zeros([pos_emb.shape[0], pos_emb.shape[1], self.hidden_size], dtype=torch.float).to(bert_embed.device)
        for i, batch in enumerate(predict_mask):
            k = 0
            for j, feat in enumerate(batch):
                if feat == 1:
                    embed_new[i][k] = bert_embed[i][j]
                    k += 1

        # Look u
        word_emb = self.word_embedding(words)

        word_pos_emb = torch.cat([embed_new, pos_emb], dim=2)

        rnn_out, _ = self.rnn(word_pos_emb)

        return rnn_out

class EdgeFactoredParser(nn.Module):

    def __init__(self, fields, word_emb_dim, pos_emb_dim,
                 rnn_size, rnn_depth, mlp_size, update_pretrained=False):
        super().__init__()

        word_field = fields[0][1]
        pos_field = fields[1][1]

        # Sentence encoder module.
        self.encoder = RNNEncoder(word_field, word_emb_dim, pos_field, pos_emb_dim, rnn_size, rnn_depth,
                                  update_pretrained)

        # Edge scoring module.
        self.edge_scorer = BiaffineEdgeScorer(2*rnn_size, mlp_size)

        # To deal with the padding positions later, we need to know the
        # encoding of the padding dummy word.
        self.pad_id = word_field.vocab.stoi[word_field.pad_token]

        # Loss function that we will use during training.
        self.loss = torch.nn.CrossEntropyLoss(reduction='none')

    def word_tag_dropout(self, words, postags, p_drop):
        # Randomly replace some of the positions in the word and postag tensors with a zero.
        # This solution is a bit hacky because we assume that zero corresponds to the "unknown" token.
        w_dropout_mask = (torch.rand(size=words.shape, device=words.device) > p_drop).long()
        p_dropout_mask = (torch.rand(size=words.shape, device=words.device) > p_drop).long()
        return words*w_dropout_mask, postags*p_dropout_mask

    def forward(self, words, postags, heads, input_ids, token_type_ids, input_mask, predict_mask, evaluate=False):

        if self.training:
            # If we are training, apply the word/tag dropout to the word and tag tensors.
            words, postags = self.word_tag_dropout(words, postags, 0.25)

        encoded = self.encoder(words, postags, input_ids, token_type_ids, input_mask, predict_mask, evaluate)
        edge_scores = self.edge_scorer(encoded)

        # We don't want to evaluate the loss or attachment score for the positions
        # where we have a padding token. So we create a mask that will be zero for those
        # positions and one elsewhere.
        pad_mask = (words != self.pad_id).float()

        loss = self.compute_loss(edge_scores, heads, pad_mask)

        if evaluate:
            n_errors, n_tokens = self.evaluate(edge_scores, heads, pad_mask)
            return loss, n_errors, n_tokens
        else:
            return loss

    def compute_loss(self, edge_scores, heads, pad_mask):
        n_sentences, n_words, _ = edge_scores.shape
        edge_scores = edge_scores.view(n_sentences*n_words, n_words)
        heads = heads.view(n_sentences*n_words)
        pad_mask = pad_mask.view(n_sentences*n_words)
        print("edge_scores.shape:", edge_scores.shape)
        print("heads.shape:", heads.shape)
        loss = self.loss(edge_scores, heads)
        print("loss:", loss)
        avg_loss = loss.dot(pad_mask) / pad_mask.sum()
        return avg_loss

    def evaluate(self, edge_scores, heads, pad_mask):
        n_sentences, n_words, _ = edge_scores.shape
        edge_scores = edge_scores.view(n_sentences*n_words, n_words)
        heads = heads.view(n_sentences*n_words)
        pad_mask = pad_mask.view(n_sentences*n_words)
        n_tokens = pad_mask.sum()
        predictions = edge_scores.argmax(dim=1)
        n_errors = (predictions != heads).float().dot(pad_mask)
        return n_errors.item(), n_tokens.item()

    def predict(self, words, postags):
        # This method is used to parse a sentence when the model has been trained.
        encoded = self.encoder(words, postags)
        edge_scores = self.edge_scorer(encoded)
        return edge_scores.argmax(dim=2)


def read_data(corpus_file, datafields, tokenizer):
    max_seq = 0
    count = 0
    with open(corpus_file, encoding='utf-8') as f:
        examples = []
        words = []
        tokens = ['[CLS]']
        predict_mask = [-1]
        postags = []
        heads = []
        deprels = []
        for line in f:
            if line[0] == '#': # Skip comments.
                continue
            line = line.strip()
            if not line:
                # Blank line for the end of a sentence.
                tokens.append('[SEP]')
                predict_mask.append(-1)
                input_ids = tokenizer.convert_tokens_to_ids(tokens)
                segment_ids = [0] * len(input_ids)
                input_mask = [1] * len(input_ids)

                examples.append(torchtext.data.Example.fromlist([words, postags, heads, deprels, input_ids,
                                                                 segment_ids, input_mask, predict_mask], datafields))
                words = []
                postags = []
                heads = []
                deprels = []
                if len(tokens) > max_seq:
                    max_seq = len(tokens)

                if len(tokens) > 512:
                    count += 1
                tokens = ['[CLS]']
                predict_mask = [-1]
            else:
                columns = line.split('\t')
                # Skip dummy tokens used in ellipsis constructions, and multiword tokens.
                if '.' in columns[0] or '-' in columns[0]:
                    continue
                word = columns[1]
                words.append(word)

                sub_words = tokenizer.tokenize(text=word)
                if not sub_words:
                    sub_words = ['[UNK]']

                tokens.extend(sub_words)

                for j in range(len(sub_words)):
                    if j == 0:
                        predict_mask.append(1)
                    else:
                        predict_mask.append(-1)

                postags.append(columns[4])
                heads.append(int(columns[6]))
                deprels.append(columns[7])

        return torchtext.data.Dataset(examples, datafields)

class DependencyParser:

    def __init__(self, lower=False):
        pad = '<pad>'
        self.WORD = torchtext.data.Field(init_token=pad, pad_token=pad, sequential=True,
                                         lower=lower, batch_first=True)
        self.POS = torchtext.data.Field(init_token=pad, pad_token=pad, sequential=True,
                                        batch_first=True)
        self.HEAD = torchtext.data.Field(init_token=0, pad_token=0, use_vocab=False, sequential=True,
                                         batch_first=True)
        self.DEPREL = torchtext.data.Field(init_token=pad, pad_token=pad, sequential=True,
                                           batch_first=True)

        self.INPUT_ID = torchtext.data.Field(init_token=0, pad_token=0, use_vocab=False, sequential=True,
                                             batch_first=True)

        self.TOKEN_TYPE_ID = torchtext.data.Field(init_token=0, pad_token=0, use_vocab=False, sequential=True,
                                                  batch_first=True)

        self.INPUT_MASK = torchtext.data.Field(init_token=0, pad_token=0, use_vocab=False, sequential=True,
                                               batch_first=True)

        self.PREDICT_MASK = torchtext.data.Field(init_token=1, pad_token=0, use_vocab=False, sequential=True,
                                                 batch_first=True)

        self.fields = [('words', self.WORD), ('postags', self.POS), ('heads', self.HEAD), ('deprels', self.DEPREL),
                       ('input_ids', self.INPUT_ID), ('token_type_ids', self.TOKEN_TYPE_ID),
                       ('input_mask', self.INPUT_MASK), ('predict_mask', self.PREDICT_MASK)]

        self.device = 'cuda'


    def train(self):
        # Read training and validation data according to the predefined split.
        data_dir = "sample_data/"
        #data_dir = "/Users/d22admin/USCGDrive/ISI/BETTER/3.Datasets/Tasks/Raw/DepParAhmad/"
        #data_dir = "/nas/clear/users/meryem/Datasets/DepParsing/data2.2_more/"

        transName = MODELS_dict["BertBaseMultilingualCased"][2]
        tokenizer = MODELS_dict["BertBaseMultilingualCased"][1].from_pretrained(transName)

        train_examples = read_data(data_dir+'en_train.conllu', self.fields, tokenizer)
        val_examples = read_data(data_dir+'en_dev.conllu', self.fields, tokenizer)

        # Load the pre-trained word embeddings that come with the torchtext library.
        use_pretrained = False
        if use_pretrained:
            print('We are using pre-trained word embeddings.')
            self.WORD.build_vocab(train_examples, vectors="glove.840B.300d")
        else:
            print('We are training word embeddings from scratch.')
            self.WORD.build_vocab(train_examples, max_size=10000)

        self.POS.build_vocab(train_examples)
        self.DEPREL.build_vocab(train_examples)

        # Create one of the models defined above.
        self.model = EdgeFactoredParser(self.fields, word_emb_dim=300, pos_emb_dim=32,
                                        rnn_size=400, rnn_depth=3, mlp_size=256, update_pretrained=False)

        self.model.to(self.device)

        batch_size = 16

        train_iterator = torchtext.data.BucketIterator(
            train_examples,
            device=self.device,
            batch_size=batch_size,
            sort_key=lambda x: len(x.words),
            repeat=False,
            train=True,
            sort=True)

        val_iterator = torchtext.data.BucketIterator(
            val_examples,
            device=self.device,
            batch_size=batch_size,
            sort_key=lambda x: len(x.words),
            repeat=False,
            train=True,
            sort=True)

        train_batches = list(train_iterator)
        val_batches = list(val_iterator)

        # We use the betas recommended in the paper by Dozat and Manning. They also use
        # a learning rate cooldown, which we don't use here to keep things simple.
        optimizer = torch.optim.Adam(self.model.parameters(), betas=(0.9, 0.9), lr=0.01, weight_decay=1e-5)

        history = defaultdict(list)

        n_epochs = 30

        for i in range(1, n_epochs + 1):

            t0 = time.time()

            stats = Counter()

            self.model.train()
            for batch in train_batches:
                loss = self.model(batch.words, batch.postags, batch.heads, batch.input_ids, batch.token_type_ids,
                                  batch.input_mask, batch.predict_mask)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                stats['train_loss'] += loss.item()

            train_loss = stats['train_loss'] / len(train_batches)
            history['train_loss'].append(train_loss)

            self.model.eval()
            with torch.no_grad():
                for batch in val_batches:
                    loss, n_err, n_tokens = self.model(batch.words, batch.postags, batch.heads, batch.input_ids, batch.token_type_ids,
                                                       batch.input_mask, batch.predict_mask, evaluate=True)
                    stats['val_loss'] += loss.item()
                    stats['val_n_tokens'] += n_tokens
                    stats['val_n_err'] += n_err

            val_loss = stats['val_loss'] / len(val_batches)
            uas = (stats['val_n_tokens']-stats['val_n_err'])/stats['val_n_tokens']
            history['val_loss'].append(val_loss)
            history['uas'].append(uas)

            t1 = time.time()
            print(f'Epoch {i}: train loss = {train_loss:.4f}, val loss = {val_loss:.4f}, UAS = {uas:.4f}, time = {t1-t0:.4f}')

parser = DependencyParser()
parser.train()
