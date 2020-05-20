import os
cwd = os.getcwd()
import sys
from pathlib import Path
sys.path.append(os.path.join(Path(os.getcwd()).parent, "OnlineAlignment"))
from evaluators import *

import torch
import time
import torchtext

from collections import defaultdict, Counter

from data_utils import *
#from model import *
from model_bert import *
from get_arguments import *
import os
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import ast
import numpy as np
from torch.utils import data
import csv

hp = get_args()
tokenizer = MODELS_dict[hp.trans_model][1].from_pretrained(MODELS_dict[hp.trans_model][2])
cls_token_id = tokenizer.convert_tokens_to_ids(["[CLS]"])[0]
sep_token_id = tokenizer.convert_tokens_to_ids(["[SEP]"])[0]
lang_dict = {"en": 0, "ar": 1, "de": 2, "es": 3, "zh": 4}
max_seq_length = 512

def get_sentence_features(subtokens, pad_seq_length):
    pad_seq_length = min(pad_seq_length, max_seq_length)

    subtokens = subtokens[:pad_seq_length]
    input_ids = [cls_token_id] + subtokens + [sep_token_id]
    sentence_length = len(input_ids)

    pad_seq_length += 2  ##Add Space for CLS + SEP token

    token_type_ids = [0] * len(input_ids)
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length. BERT: Pad to the right
    padding = [0] * (pad_seq_length - len(input_ids))
    input_ids += padding
    token_type_ids += padding
    input_mask += padding

    assert len(input_ids) == pad_seq_length
    assert len(input_mask) == pad_seq_length
    assert len(token_type_ids) == pad_seq_length

    return {'input_ids': np.asarray(input_ids, dtype=np.int64),
            'token_type_ids': np.asarray(token_type_ids, dtype=np.int64),
            'input_mask': np.asarray(input_mask, dtype=np.int64),
            'sentence_lengths': np.asarray(sentence_length, dtype=np.int64)}


def smart_batching_collate(batch):
    """
    Transforms a batch from a SmartBatchingDataset to a batch of tensors for the model

    :param batch:
        a batch from a SmartBatchingDataset
    :return:
        a batch of tensors for the model
    """
    num_texts = len(batch[0][0])

    labels = []
    paired_texts = [[] for _ in range(num_texts)]
    max_seq_len = [0] * num_texts
    for tokens, label in batch:
        labels.append(label)
        for i in range(num_texts):
            paired_texts[i].append(tokens[i])
            max_seq_len[i] = max(max_seq_len[i], len(tokens[i]))

    features = []
    for idx in range(num_texts):
        max_len = max_seq_len[idx]
        feature_lists = {}
        for text in paired_texts[idx]:
            sentence_features = get_sentence_features(text, max_len)

            for feature_name in sentence_features:
                if feature_name not in feature_lists:
                    feature_lists[feature_name] = []
                feature_lists[feature_name].append(sentence_features[feature_name])

        for feature_name in feature_lists:
            feature_lists[feature_name] = torch.tensor(np.asarray(feature_lists[feature_name]))

        features.append(feature_lists)

    return {'features': features, 'labels': torch.stack(labels)}


def batch_to_device(batch, target_device):
    """
    send a batch to a device

    :param batch:
    :param target_device:
    :return: the batch sent to the device
    """
    features = batch['features']
    for paired_sentence_idx in range(len(features)):
        for feature_name in features[paired_sentence_idx]:
            features[paired_sentence_idx][feature_name] = features[paired_sentence_idx][feature_name].to(target_device)

    labels = batch['labels'].to(target_device)
    return features, labels


def load_instrinsic_triplet_data(batch_size, xintr_path, test_langs, train_lang="en", train_max_examples=1000000,
                                 test_max_examples=1000000):
    xintr_reader = XTripletReader(xintr_path, s1_col_idx=0, s2_col_idx=1, s3_col_idx=2, delimiter=',',
                                  quoting=csv.QUOTE_MINIMAL, has_header=True)

    print("Loading Intrinsic Training Data from Cross-Lingual Triplet train portion")
    train_examples = []
    for test_lang in test_langs:
        if test_lang != "en":
            train_examples.extend(xintr_reader.get_examples(test_lang + '-' + train_lang + '-train.csv', start=0,
                                                            max_examples=train_max_examples))

    train_data = SentencesDataset(examples=train_examples, model=tokenizer, show_progress_bar=True)
    train_dataloader = data.DataLoader(train_data, shuffle=True, batch_size=batch_size)
    train_dataloader.collate_fn = smart_batching_collate

    # --------------------------------------------------------------------------------
    print("Loading Intrinsic Data from Cross-Lingual Triplet test portion")
    test_examples = []
    for test_lang in test_langs:
        if test_lang != "en":
            test_examples.extend(xintr_reader.get_examples(test_lang + '-' + train_lang + '-test.csv', start=0,
                                                           max_examples=test_max_examples))

    batch_size = 1
    trip_test_data = SentencesDataset(examples=test_examples, model=tokenizer, show_progress_bar=True)

    trip_test_dataloader = data.DataLoader(trip_test_data, shuffle=True, batch_size=batch_size)
    trip_test_dataloader.collate_fn = smart_batching_collate
    test_triplet_evaluator = TripletEvaluator(trip_test_dataloader, main_distance_function=None, name="test")

    return train_dataloader, trip_test_dataloader, test_triplet_evaluator


def load_instrinsic_xsts_data(batch_size, stsb_path, test_lang, train_lang="en"):
    sts_reader = STSBDataReader(stsb_path)
    print("Loading Intrinsic Data from STS mono train_lang portion")

    sts_mono_train_data = SentencesDataset(
        examples=sts_reader.get_examples('STS.' + train_lang + '-' + train_lang + '.txt'), model=tokenizer,
        show_progress_bar=True)
    sts_mono_train_dataloader = data.DataLoader(sts_mono_train_data, shuffle=False, batch_size=batch_size)
    sts_mono_train_dataloader.collate_fn = smart_batching_collate
    sts_mono_train_evaluator = EmbeddingSimilarityEvaluator(sts_mono_train_dataloader,
                                                            name='sts_' + train_lang + '_mono')

    print("Loading Intrinsic Data from STS mono test_lang portion")
    if test_lang not in ["de", "zh"]:
        sts_mono_test_data = SentencesDataset(
            examples=sts_reader.get_examples('STS.' + test_lang + '-' + test_lang + '.txt'), model=tokenizer,
            show_progress_bar=True)
        sts_mono_test_dataloader = data.DataLoader(sts_mono_test_data, shuffle=False, batch_size=batch_size)
        sts_mono_test_dataloader.collate_fn = smart_batching_collate
        sts_mono_test_evaluator = EmbeddingSimilarityEvaluator(sts_mono_test_dataloader, name="sts_" + test_lang + "_mono")
    else:
        sts_mono_test_evaluator = None

    print("Loading Intrinsic Data from STS bilingual train/test lang portion")
    if test_lang != "zh":
        sts_bil_data = SentencesDataset(examples=sts_reader.get_examples('STS.' + train_lang + '-' + test_lang + '.txt'),
                                        model=tokenizer, show_progress_bar=True)
        sts_bil_dataloader = data.DataLoader(sts_bil_data, shuffle=False, batch_size=batch_size)
        sts_bil_dataloader.collate_fn = smart_batching_collate
        sts_bil_evaluator = EmbeddingSimilarityEvaluator(sts_bil_dataloader,
                                                         name='sts_' + train_lang + '-' + test_lang + '_bil')
    else:
        sts_bil_evaluator = None

    return sts_mono_train_evaluator, sts_mono_test_evaluator, sts_bil_evaluator


class DependencyParser:

    def __init__(self, hp, lower=False):
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

        cuda_yes = torch.cuda.is_available()
        if cuda_yes:
            import GPUtil

        self.device = torch.device("cuda" if cuda_yes else "cpu")
        self.train_langs = hp.train_langs.split(",")[0]
        self.test_langs = hp.test_langs.split(",")

        self.transName = MODELS_dict[hp.trans_model][2]
        self.transModel = MODELS_dict[hp.trans_model][0].from_pretrained(self.transName)
        self.transTokenizer = MODELS_dict[hp.trans_model][1].from_pretrained(self.transName)


        output_dir = hp.output_dir + "train-" + hp.train_langs+"_" + "test-" + hp.test_langs

        if hp.use_alignment:
            output_dir = os.path.join(output_dir, hp.alignment_choice)
        elif not hp.use_multi_task:
            output_dir = os.path.join(output_dir, "no_alig")
        if hp.use_multi_task:
            output_dir = os.path.join(output_dir, "multi-task")

            output_dir = os.path.join(output_dir, "pool_" + hp.pooling_choice)

        output_dir = os.path.join(output_dir, hp.trans_model)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if hp.use_multi_task:
            trip_dir = os.path.join(output_dir, "triplets")
            stsb_dir = os.path.join(output_dir, "triplets")
            if not os.path.exists(trip_dir):
                os.makedirs(trip_dir)

            if not os.path.exists(stsb_dir):
                os.makedirs(stsb_dir)

            self.trip_dir = trip_dir
            self.stsb_dir = stsb_dir

        self.writer = SummaryWriter(os.path.join(output_dir, 'runs'))
        self.output_dir = output_dir


    def train(self):
        # Read training and validation data according to the predefined split.
        train_examples = read_data(os.path.join(hp.data_dir, hp.train_langs+'_train.conllu'), self.fields,
                                   self.transTokenizer)
        val_examples = read_data(os.path.join(hp.data_dir, hp.train_langs+'_dev.conllu'), self.fields,
                                 self.transTokenizer)

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
        if hp.use_alignment:
            alignment_files = ast.literal_eval(hp.alignment_dict)
        else:
            alignment_files = None

        self.model = EdgeFactoredParser(hp.use_multi_task, hp.use_eisner, alignment_files, hp.pooling_choice, self.fields, self.transModel,
                                        use_rnn=hp.use_rnn, pos_emb_dim=hp.pos_emb_dim, word_emb_dim=hp.word_emb_dim, use_word_embed=hp.use_word_embed,
                                        rnn_size=hp.rnn_size, rnn_depth=hp.rnn_depth, edge_mlp_size=hp.edge_mlp_size, rel_mlp_size=hp.rel_mlp_size)

        if hp.use_multi_task:
            train_trip_dataloader, test_trip_dataloader, test_trip_evaluator = load_instrinsic_triplet_data(
                hp.batch_size_intr, hp.xintr_path, self.test_langs)

            sts_mono_train_evaluator = {}
            sts_mono_test_evaluator = {}
            sts_bil_evaluator = {}
            for lang in self.test_langs:
                sts_mono_train_evaluator.update({lang: []})
                sts_mono_test_evaluator.update({lang: []})
                sts_bil_evaluator.update({lang: []})
                sts_mono_train_evaluator[lang], sts_mono_test_evaluator[lang], sts_bil_evaluator[lang] = \
                    load_instrinsic_xsts_data(hp.batch_size_intr, hp.stsb_path, lang, train_lang="en")

        if not hp.use_multi_task:
            test_trip_dataloader = None

        self.model.to(self.device)

        train_iterator = torchtext.data.BucketIterator(
            train_examples,
            device=self.device,
            batch_size=hp.batch_size,
            sort_key=lambda x: len(x.words),
            repeat=False,
            train=True,
            sort=True)

        train_batches = list(train_iterator)

        val_iterator = torchtext.data.BucketIterator(
            val_examples,
            device=self.device,
            batch_size=hp.batch_size,
            sort_key=lambda x: len(x.words),
            repeat=False,
            train=True,
            sort=True)

        val_batches = list(val_iterator)

        test_iterator = {}
        test_batches = {}
        for lang in self.test_langs:
            test_examples = read_data(os.path.join(hp.data_dir, lang+'_test.conllu'), self.fields, self.transTokenizer)
            test_iterator.update({lang: torchtext.data.BucketIterator(test_examples,
                                                          device=self.device,
                                                          batch_size=hp.batch_size,
                                                          sort_key=lambda x: len(x.words),
                                                          repeat=False,
                                                          train=True,
                                                          sort=True)})
            test_batches.update({lang: list(test_iterator[lang])})

        # We use the betas recommended in the paper by Dozat and Manning. They also use
        # a learning rate cooldown, which we don't use here to keep things simple.
        #optimizer = torch.optim.Adam(self.model.parameters(), betas=(0.9, 0.9), lr=0.005, weight_decay=1e-5)
        optimizer = torch.optim.Adam(self.model.parameters(), betas=(0.9, 0.999), lr=0.001, weight_decay=1e-5)

        history = defaultdict(list)

        n_epochs = 30
        best_las = 0
        no_improv = 0

        for i in tqdm(range(1, n_epochs + 1)):

            t0 = time.time()

            stats = Counter()

            self.model.train()

            if hp.be_verbose:
                batch = train_batches[0]
                print("batch.postags[0]:", batch.postags[0])
                print("batch.words[0]:", batch.words[0])
                print("batch.input_ids[0]:", batch.input_ids[0])
                print("batch.token_type_ids[0]:", batch.token_type_ids[0])
                print("batch.input_mask[0]:", batch.input_mask[0])
                print("batch.predict_mask[0]:", batch.predict_mask[0])
                print("batch.predict_mask[0]:", batch.predict_mask[0])
                print("batch.words:", batch.words.shape)
                print("batch.input_ids:", batch.input_ids.shape)

            for batch in train_batches:
                if hp.use_multi_task:
                    train_iter_iterator = iter(train_trip_dataloader)
                    try:
                        intr_data = next(train_iter_iterator)
                    except StopIteration:
                        train_iter_iterator = iter(train_trip_dataloader)
                        intr_data = next(train_iter_iterator)

                    features, labels = batch_to_device(intr_data, self.device)
                else:
                    features = batch.input_ids
                    labels = batch.input_ids


                intrinsic_loss, loss = self.model(torch.tensor([lang_dict[self.train_langs]]), features, labels, batch.words, batch.postags,
                                          batch.input_ids, batch.token_type_ids, batch.input_mask, batch.predict_mask,
                                          batch.heads, batch.deprels)


                optimizer.zero_grad()

                if hp.use_multi_task:
                    #if epoch <= 5:
                    tr_loss = loss.mean() + intrinsic_loss.mean()
                    self.writer.add_scalar('intrinsic_loss', intrinsic_loss.mean(), i)
                else:
                    tr_loss = loss.mean()

                tr_loss.backward()
                optimizer.step()
                stats['train_loss'] += loss.item()

            train_loss = stats['train_loss'] / len(train_batches)

            history['train_loss'].append(train_loss)
            self.writer.add_scalar('train_loss', train_loss, i)

            self.model.eval()
            with torch.no_grad():
                for batch in val_batches:
                    if hp.use_multi_task:
                        test_iter_iterator = iter(test_trip_dataloader)
                        try:
                            intr_data = next(test_iter_iterator)
                        except StopIteration:
                            test_iter_iterator = iter(test_trip_dataloader)
                            intr_data = next(test_iter_iterator)

                        features, labels = batch_to_device(intr_data, self.device)
                    else:
                        features = batch.input_ids
                        labels = batch.input_ids

                    intrinsic_loss, loss, total, correct_edge, correct_rels = \
                        self.model(torch.tensor([lang_dict[self.train_langs]]), features, labels, batch.words, batch.postags,
                                   batch.input_ids, batch.token_type_ids, batch.input_mask, batch.predict_mask,
                                   batch.heads, batch.deprels,  evaluate=True)

                    stats['val_loss'] += loss.item()
                    stats['val_total'] += total
                    stats['val_correct_edge'] += correct_edge
                    stats['val_correct_rels'] += correct_rels

            val_loss = stats['val_loss'] / len(val_batches)
            val_uas = stats['val_correct_edge'] / (stats['val_total'] + 1e-5)
            val_las = stats['val_correct_rels'] / (stats['val_total'] + 1e-5)
            history['val_loss'].append(val_loss)
            history['val_uas'].append(val_uas)
            history['val_las'].append(val_las)
            self.writer.add_scalar('val_loss', val_loss, i)
            self.writer.add_scalar('val_uas', val_uas, i)
            self.writer.add_scalar('val_las', val_las, i)

            t1 = time.time()
            print(f'Epoch {i}: train loss = {train_loss:.4f}, val loss = {val_loss:.4f}, UAS = {val_uas:.4f},  LAS = {val_las:.4f}, time = {t1-t0:.4f}')

            average_las = 0
            for lang in self.test_langs:
                self.model.eval()
                with torch.no_grad():
                    for batch in test_batches[lang]:
                        if hp.use_multi_task:
                            test_iter_iterator = iter(test_trip_dataloader)
                            try:
                                intr_data = next(test_iter_iterator)
                            except StopIteration:
                                test_iter_iterator = iter(test_trip_dataloader)
                                intr_data = next(test_iter_iterator)

                            features, labels = batch_to_device(intr_data, self.device)
                        else:
                            features = batch.input_ids
                            labels = batch.input_ids
                        intrinsic_loss, loss, total, correct_edge, correct_rels = \
                            self.model(torch.tensor([lang_dict[lang]]), features, labels, batch.words, batch.postags, batch.input_ids,
                                       batch.token_type_ids, batch.input_mask, batch.predict_mask, batch.heads,
                                       batch.deprels, evaluate=True)

                        stats['test_'+lang+'_loss'] += loss.item()
                        stats['test_'+lang+'_total'] += total
                        stats['test_'+lang+'_correct_edge'] += correct_edge
                        stats['test_'+lang+'_correct_rels'] += correct_rels

                test_loss = stats['test_'+lang+'_loss'] / len(test_batches)
                test_uas = stats['test_'+lang+'_correct_edge'] / (stats['test_'+lang+'_total'] + 1e-5)
                test_las = stats['test_'+lang+'_correct_rels'] / (stats['test_'+lang+'_total'] + 1e-5)
                history['test_'+lang+'_loss'].append(test_loss)
                history['test_'+lang+'_uas'].append(test_uas)
                history['test_'+lang+'_las'].append(test_las)
                self.writer.add_scalar('test_'+lang+'_loss', test_loss, i)
                self.writer.add_scalar('test_'+lang+'_uas', test_uas, i)
                self.writer.add_scalar('test_'+lang+'_las', test_las, i)

                average_las += test_las

                print(f'-------------lang={lang}, test loss = {test_loss:.4f}, UAS = {test_uas:.4f}, LAS = {test_las:.4f}')

            average_las = average_las / len(self.test_langs)
            if average_las > best_las:
                best_las = average_las
                torch.save({'epoch': i, 'model_state': self.model.state_dict(), 'val_uas': val_uas, 'val_las': val_las},
                           os.path.join(self.output_dir, 'best_checkpoint.pt'))
            else:
                no_improv += 1

            if no_improv == hp.early_stop_ep:
                print("No Improvement Early Stopping!")
                break


    def parse(self, sentences):
        # This method applies the trained model to a list of sentences.

        # First, create a torchtext Dataset containing the sentences to tag.
        examples = []
        for tagged_words in sentences:
            words = [w for w, _ in tagged_words]
            tags = [t for _, t in tagged_words]
            heads = [0]*len(words) # placeholder
            examples.append(torchtext.data.Example.fromlist([words, tags, heads], self.fields))
        dataset = torchtext.data.Dataset(examples, self.fields)

        iterator = torchtext.data.Iterator(
            dataset,
            device=self.device,
            batch_size=len(examples),
            repeat=False,
            train=False,
            sort=False)

        # Apply the trained model to the examples.
        out = []
        self.model.eval()
        with torch.no_grad():
            for batch in iterator:
                predicted = self.model.predict(batch.words, batch.postags)
                out.extend(predicted.cpu().numpy())
        return out


if __name__=="__main__":
    hp = get_args()
    parser = DependencyParser(hp)
    parser.train()