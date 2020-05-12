"""
 Code adapted from https://github.com/UKPLab/sentence-transformers
"""
from sklearn.metrics.pairwise import paired_cosine_distances, paired_euclidean_distances, paired_manhattan_distances
import torch
from torch.utils.data import DataLoader
from torch import Tensor, device
import logging
from tqdm import tqdm
import os
import csv
from scipy.stats import pearsonr, spearmanr
import numpy as np
from enum import Enum
from ner_bert_crf import *

def batch_to_device(batch, target_device: device):
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

class SimilarityFunction(Enum):
    COSINE = 0
    EUCLIDEAN = 1
    MANHATTAN = 2
    DOT_PRODUCT = 3

class SentenceEvaluator:
    """
    Base class for all evaluators

    Extend this class and implement __call__ for custom evaluators.
    """

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        """
        This is called during training to evaluate the model.
        It returns a score for the evaluation with a higher score indicating a better result.

        :param model:
            the model to evaluate
        :param output_path:
            path where predictions and metrics are written to
        :param epoch
            the epoch where the evaluation takes place.
            This is used for the file prefixes.
            If this is -1, then we assume evaluation on test data.
        :param steps
            the steps in the current epoch at time of the evaluation.
            This is used for the file prefixes.
            If this is -1, then we assume evaluation at the end of the epoch.
        :return: a score for the evaluation with a higher score indicating a better result
        """
        pass

class EmbeddingSimilarityEvaluator(SentenceEvaluator):
    """
    Evaluate a model based on the similarity of the embeddings by calculating the Spearman and Pearson rank correlation
    in comparison to the gold standard labels.
    The metrics are the cosine similarity as well as euclidean and Manhattan distance
    The returned score is the Spearman correlation with a specified metric.

    The results are written in a CSV. If a CSV already exists, then values are appended.
    """


    def __init__(self, dataloader: DataLoader, main_similarity: SimilarityFunction = None, name: str = '', show_progress_bar: bool = None):
        """
        Constructs an evaluator based for the dataset

        The labels need to indicate the similarity between the sentences.

        :param dataloader:
            the data for the evaluation
        :param main_similarity:
            the similarity metric that will be used for the returned score
        """
        self.dataloader = dataloader
        self.main_similarity = main_similarity
        self.name = name
        if name:
            name = "_"+name

        if show_progress_bar is None:
            show_progress_bar = (logging.getLogger().getEffectiveLevel() == logging.INFO or logging.getLogger().getEffectiveLevel() == logging.DEBUG)
        self.show_progress_bar = show_progress_bar

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.csv_file = "similarity_evaluation"+name+"_results.csv"
        self.csv_headers = ["epoch", "steps", "cosine_pearson", "cosine_spearman", "euclidean_pearson", "euclidean_spearman", "manhattan_pearson", "manhattan_spearman", "dot_pearson", "dot_spearman"]

    def __call__(self, model: 'SequentialSentenceEmbedder', output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        model.eval()
        embeddings1 = []
        embeddings2 = []
        labels = []

        if epoch != -1:
            if steps == -1:
                out_txt = f" after epoch {epoch}:"
            else:
                out_txt = f" in epoch {epoch} after {steps} steps:"
        else:
            out_txt = ":"

        logging.info("Evaluation the model on "+self.name+" dataset"+out_txt)

        def get_sentence_features(subtokens, pad_seq_length):
            max_seq_length = 510
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

        self.dataloader.collate_fn = smart_batching_collate

        iterator = self.dataloader
        #if self.show_progress_bar:
        #    iterator = tqdm(iterator, desc="Convert Evaluating")

        for step, batch in enumerate(iterator):
            features, label_ids = batch_to_device(batch, self.device)
            with torch.no_grad():
                sentence_embeddings = model.module._get_sentence_embeddings(features)
                emb1, emb2 = [sentence_embeddings[key].to("cpu").numpy() for key in sentence_embeddings]

            labels.extend(label_ids.to("cpu").numpy())
            embeddings1.extend(emb1)
            embeddings2.extend(emb2)

        try:
            cosine_scores = 1 - (paired_cosine_distances(embeddings1, embeddings2))
        except Exception as e:
            print(embeddings1)
            print(embeddings2)
            raise(e)

        manhattan_distances = -paired_manhattan_distances(embeddings1, embeddings2)
        euclidean_distances = -paired_euclidean_distances(embeddings1, embeddings2)
        dot_products = [np.dot(emb1, emb2) for emb1, emb2 in zip(embeddings1, embeddings2)]


        eval_pearson_cosine, _ = pearsonr(labels, cosine_scores)
        eval_spearman_cosine, _ = spearmanr(labels, cosine_scores)

        eval_pearson_manhattan, _ = pearsonr(labels, manhattan_distances)
        eval_spearman_manhattan, _ = spearmanr(labels, manhattan_distances)

        eval_pearson_euclidean, _ = pearsonr(labels, euclidean_distances)
        eval_spearman_euclidean, _ = spearmanr(labels, euclidean_distances)

        eval_pearson_dot, _ = pearsonr(labels, dot_products)
        eval_spearman_dot, _ = spearmanr(labels, dot_products)

        logging.info("Cosine-Similarity :\tPearson: {:.4f}\tSpearman: {:.4f}".format(
            eval_pearson_cosine, eval_spearman_cosine))
        logging.info("Manhattan-Distance:\tPearson: {:.4f}\tSpearman: {:.4f}".format(
            eval_pearson_manhattan, eval_spearman_manhattan))
        logging.info("Euclidean-Distance:\tPearson: {:.4f}\tSpearman: {:.4f}".format(
            eval_pearson_euclidean, eval_spearman_euclidean))
        logging.info("Dot-Product-Similarity:\tPearson: {:.4f}\tSpearman: {:.4f}".format(
            eval_pearson_dot, eval_spearman_dot))

        if output_path is not None:
            csv_path = os.path.join(output_path, self.csv_file)
            output_file_exists = os.path.isfile(csv_path)
            with open(csv_path, mode="a" if output_file_exists else 'w', encoding="utf-8") as f:
                writer = csv.writer(f)
                if not output_file_exists:
                    writer.writerow(self.csv_headers)

                writer.writerow([epoch, steps, eval_pearson_cosine, eval_spearman_cosine, eval_pearson_euclidean,
                                 eval_spearman_euclidean, eval_pearson_manhattan, eval_spearman_manhattan, eval_pearson_dot, eval_spearman_dot])


        if self.main_similarity == SimilarityFunction.COSINE:
            return eval_spearman_cosine
        elif self.main_similarity == SimilarityFunction.EUCLIDEAN:
            return eval_spearman_euclidean
        elif self.main_similarity == SimilarityFunction.MANHATTAN:
            return eval_spearman_manhattan
        elif self.main_similarity == SimilarityFunction.DOT_PRODUCT:
            return eval_spearman_dot
        elif self.main_similarity is None:
            return eval_spearman_cosine, eval_spearman_manhattan, eval_spearman_euclidean, eval_spearman_dot
        else:
            raise ValueError("Unknown main_similarity value")

class TripletEvaluator(SentenceEvaluator):
    """
    Evaluate a model based on a triplet: (sentence, positive_example, negative_example). Checks if distance(sentence,positive_example) < distance(sentence, negative_example).
    """
    def __init__(self, dataloader: DataLoader, main_distance_function: SimilarityFunction = None, name: str =''):
        """
        Constructs an evaluator based for the dataset


        :param dataloader:
            the data for the evaluation
        :param main_similarity:
            the similarity metric that will be used for the returned score
        """
        self.dataloader = dataloader
        self.main_distance_function = main_distance_function
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.name = name
        if name:
            name = "_"+name

        self.csv_file: str = "triplet_evaluation"+name+"_results.csv"
        self.csv_headers = ["epoch", "steps", "accuracy_cosinus", "accuracy_manhatten", "accuracy_euclidean"]

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        model.eval()

        if epoch != -1:
            if steps == -1:
                out_txt = " after epoch {}:".format(epoch)
            else:
                out_txt = " in epoch {} after {} steps:".format(epoch, steps)
        else:
            out_txt = ":"

        logging.info("Evaluation the model on "+self.name+" dataset"+out_txt)

        num_triplets = 0
        num_correct_cos_triplets, num_correct_manhatten_triplets, num_correct_euclidean_triplets = 0, 0, 0

        def get_sentence_features(subtokens, pad_seq_length):
            max_seq_length = 510
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

        self.dataloader.collate_fn = smart_batching_collate
        for step, batch in enumerate(self.dataloader):
            features, label_ids = batch_to_device(batch, self.device)
            with torch.no_grad():
                sentence_embeddings = model.module._get_sentence_embeddings(features)
                emb1, emb2, emb3 = [sentence_embeddings[key].to("cpu").numpy() for key in sentence_embeddings]

            #Cosine distance
            pos_cos_distance = paired_cosine_distances(emb1, emb2)
            neg_cos_distances = paired_cosine_distances(emb1, emb3)

            # Manhatten
            pos_manhatten_distance = paired_manhattan_distances(emb1, emb2)
            neg_manhatten_distances = paired_manhattan_distances(emb1, emb3)

            # Euclidean
            pos_euclidean_distance = paired_euclidean_distances(emb1, emb2)
            neg_euclidean_distances = paired_euclidean_distances(emb1, emb3)

            for idx in range(len(pos_cos_distance)):
                num_triplets += 1

                if pos_cos_distance[idx] < neg_cos_distances[idx]:
                    num_correct_cos_triplets += 1

                if pos_manhatten_distance[idx] < neg_manhatten_distances[idx]:
                    num_correct_manhatten_triplets += 1

                if pos_euclidean_distance[idx] < neg_euclidean_distances[idx]:
                    num_correct_euclidean_triplets += 1

        accuracy_cos = num_correct_cos_triplets / num_triplets
        accuracy_manhatten = num_correct_manhatten_triplets / num_triplets
        accuracy_euclidean = num_correct_euclidean_triplets / num_triplets

        logging.info("Accuracy Cosine Distance:\t{:.4f}".format(accuracy_cos))
        logging.info("Accuracy Manhatten Distance:\t{:.4f}".format(accuracy_manhatten))
        logging.info("Accuracy Euclidean Distance:\t{:.4f}\n".format(accuracy_euclidean))

        if output_path is not None:
            csv_path = os.path.join(output_path, self.csv_file)
            if not os.path.isfile(csv_path):
                with open(csv_path, mode="w", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(self.csv_headers)
                    writer.writerow([epoch, steps, accuracy_cos, accuracy_manhatten, accuracy_euclidean])

            else:
                with open(csv_path, mode="a", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch, steps, accuracy_cos, accuracy_manhatten, accuracy_euclidean])

        if self.main_distance_function == SimilarityFunction.COSINE:
            return accuracy_cos
        if self.main_distance_function == SimilarityFunction.MANHATTAN:
            return accuracy_manhatten
        if self.main_distance_function == SimilarityFunction.EUCLIDEAN:
            return accuracy_euclidean

        return accuracy_cos, accuracy_manhatten, accuracy_euclidean
