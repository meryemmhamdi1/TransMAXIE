"""
 Code adapted from https://github.com/UKPLab/sentence-transformers
"""

import csv
import gzip
import os
from typing import Union, List
from torch.utils.data import Dataset
from typing import List
import torch
import logging
from tqdm import tqdm

class InputExample:
    """
    Structure for one input example with texts, the label and a unique id
    """
    def __init__(self, guid: str, texts: List[str], label: Union[int, float]):
        """
        Creates one InputExample with the given texts, guid and label

        str.strip() is called on both texts.

        :param guid
            id for the example
        :param texts
            the texts for the example
        :param label
            the label for the example
        """
        self.guid = guid
        self.texts = [text.strip() for text in texts]
        self.label = label

class XTripletReader(object):
    """
    Reads in the a Triplet Dataset: Each line contains (at least) 3 columns, one anchor column (s1),
    one positive example (s2) and one negative example (s3)
    """
    def __init__(self, dataset_folder, s1_col_idx=0, s2_col_idx=1, s3_col_idx=2, has_header=False, delimiter="\t",
                 quoting=csv.QUOTE_NONE):
        self.dataset_folder = dataset_folder
        self.s1_col_idx = s1_col_idx
        self.s2_col_idx = s2_col_idx
        self.s3_col_idx = s3_col_idx
        self.has_header = has_header
        self.delimiter = delimiter
        self.quoting = quoting

    def get_examples(self, filename, start=0, max_examples=0):
        import pandas
        #data = csv.reader(open(os.path.join(self.dataset_folder, filename), encoding="utf-8"), delimiter=self.delimiter,
                          #quoting=self.quoting)
        if start == 0:
            data = pandas.read_csv(os.path.join(self.dataset_folder, filename), delimiter=self.delimiter, quoting=self.quoting)
        else:
            data = pandas.read_csv(os.path.join(self.dataset_folder, filename), delimiter=self.delimiter, quoting=self.quoting, skiprows=(1,start))
        examples = []

        for id, row in data.iterrows():
            s1 = row["anch"]
            s2 = row["pos"]
            s3 = row["neg"]

            if not pandas.isna(s1) and not pandas.isna(s2) and not pandas.isna(s3):
                examples.append(InputExample(guid=filename+str(id), texts=[s1, s2, s3], label=1))

            if max_examples > 0 and len(examples) >= max_examples:
                break

        return examples

class SentencesDataset(Dataset):
    """
    Dataset for smart batching, that is each batch is only padded to its longest sequence instead of padding all
    sequences to the max length.
    The SentenceBertEncoder.smart_batching_collate is required for this to work.
    SmartBatchingDataset does *not* work without it.
    """
    def __init__(self, examples: List[InputExample], model, show_progress_bar: bool = None):
        """
        Create a new SentencesDataset with the tokenized texts and the labels as Tensor
        """
        if show_progress_bar is None:
            show_progress_bar = (logging.getLogger().getEffectiveLevel() == logging.INFO or logging.getLogger().getEffectiveLevel() == logging.DEBUG)
        self.show_progress_bar = show_progress_bar

        self.convert_input_examples(examples, model)

    def convert_input_examples(self, examples, tokenizer):
        """
        Converts input examples to a SmartBatchingDataset usable to train the model with
        smart_batching_collate as the collate_fn for the DataLoader

        smart_batching_collate as collate_fn is required because it transforms the tokenized texts to the tensors.

        :param examples:
            the input examples for the training
        :param model
            Tokenizer used
        :return: a SmartBatchingDataset usable to train the model with smart_batching_collate as the collate_fn
            for the DataLoader
        """
        num_texts = len(examples[0].texts)
        inputs = [[] for _ in range(num_texts)]
        labels = []
        too_long = [0] * num_texts
        label_type = None
        iterator = examples
        max_seq_length = 50 #510

        if self.show_progress_bar:
            iterator = tqdm(iterator, desc="Convert dataset")

        for ex_index, example in enumerate(iterator):
            if label_type is None:
                if isinstance(example.label, int):
                    label_type = torch.long
                elif isinstance(example.label, float):
                    label_type = torch.float
            tokenized_texts = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text[:512]))[:max_seq_length] for text in example.texts]

            for i, token in enumerate(tokenized_texts):
                if max_seq_length != None and max_seq_length > 0 and len(token) >= max_seq_length:
                    too_long[i] += 1

            labels.append(example.label)
            for i in range(num_texts):
                inputs[i].append(tokenized_texts[i])

        tensor_labels = torch.tensor(labels, dtype=label_type)

        logging.info("Num sentences: %d" % (len(examples)))
        for i in range(num_texts):
            logging.info("Sentences {} longer than max_seqence_length: {}".format(i, too_long[i]))

        self.tokens = inputs
        self.labels = tensor_labels

    def smart_batching_collate(self, batch):
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
                sentence_features = self.get_sentence_features(text, max_len)

                for feature_name in sentence_features:
                    if feature_name not in feature_lists:
                        feature_lists[feature_name] = []
                    feature_lists[feature_name].append(sentence_features[feature_name])

            for feature_name in feature_lists:
                feature_lists[feature_name] = torch.tensor(np.asarray(feature_lists[feature_name]))

            features.append(feature_lists)

        return {'features': features, 'labels': torch.stack(labels)}

    def __getitem__(self, item):
        return [self.tokens[i][item] for i in range(len(self.tokens))], self.labels[item]

    def __len__(self):
        return len(self.tokens[0])

class STSDataReader:
    """
    Reads in the STS dataset. Each line contains two sentences (s1_col_idx, s2_col_idx) and one label (score_col_idx)
    """
    def __init__(self, dataset_folder, s1_col_idx=5, s2_col_idx=6, score_col_idx=4, delimiter="\t",
                 quoting=csv.QUOTE_NONE, normalize_scores=True, min_score=0, max_score=5):
        self.dataset_folder = dataset_folder
        self.score_col_idx = score_col_idx
        self.s1_col_idx = s1_col_idx
        self.s2_col_idx = s2_col_idx
        self.delimiter = delimiter
        self.quoting = quoting
        self.normalize_scores = normalize_scores
        self.min_score = min_score
        self.max_score = max_score

    def get_examples(self, filename, max_examples=0):
        """
        filename specified which data split to use (train.csv, dev.csv, test.csv).
        """
        filepath = os.path.join(self.dataset_folder, filename)
        fIn = gzip.open(filepath, 'rt', encoding='utf8') if filename.endswith('.gz') else open(filepath, encoding="utf-8")
        data = csv.reader(fIn, delimiter=self.delimiter, quoting=self.quoting)
        examples = []
        for id, row in enumerate(data):
            score = float(row[self.score_col_idx])
            if self.normalize_scores:  # Normalize to a 0...1 value
                score = (score - self.min_score) / (self.max_score - self.min_score)

            s1 = row[self.s1_col_idx]
            s2 = row[self.s2_col_idx]
            examples.append(InputExample(guid=filename+str(id), texts=[s1, s2], label=score))

            if max_examples > 0 and len(examples) >= max_examples:
                break

        return examples

class STSBDataReader:
    """
    Reads in the STS benchmark dataset. Each line contains two sentences (s1_col_idx, s2_col_idx) and one label (score_col_idx)
    """
    def __init__(self, dataset_folder, s1_col_idx=0, s2_col_idx=1, score_col_idx=2, delimiter="\t",
                 quoting=csv.QUOTE_NONE, normalize_scores=True, min_score=0, max_score=5):
        self.dataset_folder = dataset_folder
        self.score_col_idx = score_col_idx
        self.s1_col_idx = s1_col_idx
        self.s2_col_idx = s2_col_idx
        self.delimiter = delimiter
        self.quoting = quoting
        self.normalize_scores = normalize_scores
        self.min_score = min_score
        self.max_score = max_score

    def get_examples(self, filename, max_examples=0):
        """
        filename specified which data split to use (train.csv, dev.csv, test.csv).
        """
        filepath = os.path.join(self.dataset_folder, filename)
        fIn = gzip.open(filepath, 'rt', encoding='utf8') if filename.endswith('.gz') else open(filepath, encoding="utf-8")
        data = csv.reader(fIn, delimiter=self.delimiter, quoting=self.quoting)
        examples = []
        for id, row in enumerate(data):
            score = float(row[self.score_col_idx])
            if self.normalize_scores:  # Normalize to a 0...1 value
                score = (score - self.min_score) / (self.max_score - self.min_score)

            s1 = row[self.s1_col_idx]
            s2 = row[self.s2_col_idx]
            examples.append(InputExample(guid=filename+str(id), texts=[s1, s2], label=score))

            if max_examples > 0 and len(examples) >= max_examples:
                break

        return examples
