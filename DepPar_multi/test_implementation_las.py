import torch
from torch import nn
import time
import torchtext
import numpy as np

import random

from collections import defaultdict, Counter
from get_arguments import *

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

        rnn_out, _ = self.rnn(word_pos_emb)

        return rnn_out

class EdgeFactoredParser(nn.Module):

    def __init__(self, fields, word_emb_dim, pos_emb_dim,
                 rnn_size, rnn_depth, mlp_size, update_pretrained=False):
        super().__init__()

        word_field = fields[0][1]
        pos_field = fields[1][1]
        deprel_field = fields[3][1]

        rel_size = len(deprel_field.vocab)

        # Sentence encoder module.
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

    def forward(self, words, postags, heads, deprels, evaluate=False):

        if self.training:
            # If we are training, apply the word/tag dropout to the word and tag tensors.
            words, postags = self.word_tag_dropout(words, postags, 0.25)

        encoded = self.encoder(words, postags)

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
            return loss, total, correct_edge, correct_rels
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

        pad_mask = pad_mask.float()
        #print("loss:", loss)
        #print("pad_mask.shape:", pad_mask.shape)
        #print("edge_loss.shape:", edge_loss.shape)
        #print("rel_loss.shape:", rel_loss.shape)
        avg_loss = loss
        #print("avg_loss:", avg_loss)
        return avg_loss

    def decode(self, edge_scores, rel_scores, pad_mask):
        edge_preds = edge_scores.argmax(-1)
        rel_preds = rel_scores.argmax(-1)
        rel_preds = rel_preds.gather(-1, edge_preds.unsqueeze(-1)).squeeze(-1)

        return edge_preds, rel_preds

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

def read_data(corpus_file, datafields):
    with open(corpus_file, encoding='utf-8') as f:
        examples = []
        words = []
        postags = []
        heads = []
        deprels = []
        for line in f:
            if line[0] == '#': # Skip comments.
                continue
            line = line.strip()
            if not line:
                # Blank line for the end of a sentence.
                examples.append(torchtext.data.Example.fromlist([words, postags, heads, deprels], datafields))
                words = []
                postags = []
                heads = []
                deprels = []
            else:
                columns = line.split('\t')
                # Skip dummy tokens used in ellipsis constructions, and multiword tokens.
                if '.' in columns[0] or '-' in columns[0]:
                    continue
                words.append(columns[1])
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

        self.fields = [('words', self.WORD), ('postags', self.POS), ('heads', self.HEAD), ('deprels', self.DEPREL)]

        self.device = 'cpu'


    def train(self):
        # Read training and validation data according to the predefined split.
        #data_dir = "sample_data/"
        data_dir = "/Users/d22admin/USCGDrive/ISI/BETTER/3.Datasets/Tasks/Raw/DepParAhmad/"
        #data_dir = "/nas/clear/users/meryem/Datasets/DepParsing/data2.2_more/"

        transName = MODELS_dict["BertBaseMultilingualCased"][2]
        tokenizer = MODELS_dict["BertBaseMultilingualCased"][1].from_pretrained(transName)

        train_examples = read_data(data_dir+'en_train.conllu', self.fields)
        val_examples = read_data(data_dir+'en_dev.conllu', self.fields)

        test_langs = ["en"]#, "ar", "zh", "es", "de"]
        test_examples = {}
        for lang in test_langs:
            test_examples.update({lang: read_data(data_dir+lang+'_test.conllu', self.fields)})

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
                                        rnn_size=256, rnn_depth=3, mlp_size=256, update_pretrained=False)

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

        test_batches = {}
        for lang in test_langs:
            test_iterator = torchtext.data.BucketIterator(
                test_examples[lang],
                device=self.device,
                batch_size=batch_size,
                sort_key=lambda x: len(x.words),
                repeat=False,
                train=True,
                sort=True)

            test_batches.update({lang: test_iterator})

        train_batches = list(train_iterator)
        val_batches = list(val_iterator)

        # We use the betas recommended in the paper by Dozat and Manning. They also use
        # a learning rate cooldown, which we don't use here to keep things simple.
        optimizer = torch.optim.Adam(self.model.parameters(), betas=(0.9, 0.9), lr=0.005, weight_decay=1e-5)

        history = defaultdict(list)

        n_epochs = 30

        for i in range(1, n_epochs + 1):

            t0 = time.time()

            stats = Counter()

            self.model.train()
            for batch in train_batches:
                loss = self.model(batch.words, batch.postags, batch.heads, batch.deprels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                stats['train_loss'] += loss.item()

            train_loss = stats['train_loss'] / len(train_batches)
            history['train_loss'].append(train_loss)

            self.model.eval()
            with torch.no_grad():
                for batch in val_batches:
                    loss, total, correct_edge, correct_rels = self.model(batch.words, batch.postags, batch.heads, batch.deprels, evaluate=True)
                    stats['val_loss'] += loss.item()
                    stats['val_n_tokens'] += total
                    stats['val_n_err_edges'] += total - correct_edge
                    stats['val_n_err_rels'] += total - correct_rels

                val_loss = stats['val_loss'] / len(val_batches)
                uas = (stats['val_n_tokens']-stats['val_n_err_edges'])/stats['val_n_tokens']
                las = (stats['val_n_tokens']-stats['val_n_err_rels'])/stats['val_n_tokens']
                history['val_loss'].append(val_loss)
                history['uas'].append(uas)
                history['las'].append(las)

                t1 = time.time()
                print(f'Epoch {i}: train loss = {train_loss:.4f}, val loss = {val_loss:.4f}, UAS = {uas:.4f}, LAS = {las:.4f}, time = {t1-t0:.4f}')

                for lang in test_langs:
                    for batch in test_batches[lang]:
                        loss, total, correct_edge, correct_rels = self.model(batch.words, batch.postags, batch.heads, batch.deprels, evaluate=True)
                        stats['test_'+lang+'_loss'] += loss.item()
                        stats['test_'+lang+'_n_tokens'] += total
                        stats['test_'+lang+'_n_err_edges'] += total - correct_edge
                        stats['test_'+lang+'_n_err_rels'] += total - correct_rels

                    test_loss = stats['test_'+lang+'_loss'] / len(test_batches[lang])
                    test_uas = (stats['test_'+lang+'_n_tokens']-stats['test_'+lang+'_n_err_edges'])/stats['test_'+lang+'_n_tokens']
                    test_las = (stats['test_'+lang+'_n_tokens']-stats['test_'+lang+'_n_err_rels'])/stats['test_'+lang+'_n_tokens']
                    history['test_'+lang+'_loss'].append(test_loss)
                    history['test_'+lang+'_uas'].append(test_uas)
                    history['test_'+lang+'_las'].append(test_las)

                    print(f'-------------lang={lang}, test loss = {test_loss:.4f}, UAS = {test_uas:.4f}, LAS = {test_las:.4f}')



parser = DependencyParser()
parser.train()
