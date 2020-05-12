"""
 Code adapted from https://github.com/UKPLab/sentence-transformers
"""

import torch
from torch import nn, Tensor
from typing import Union, Tuple, List, Iterable, Dict
import torch.nn.functional as F
from enum import Enum
import logging

class TripletDistanceMetric(Enum):
    """
    The metric for the triplet loss
    """
    COSINE = lambda x, y: 1 - F.cosine_similarity(x, y)
    EUCLIDEAN = lambda x, y: F.pairwise_distance(x, y, p=2)
    MANHATTAN = lambda x, y: F.pairwise_distance(x, y, p=1)

class TripletLoss(nn.Module):
    def __init__(self, distance_metric=TripletDistanceMetric.EUCLIDEAN, triplet_margin=1):
        super(TripletLoss, self).__init__()
        self.distance_metric = distance_metric
        self.triplet_margin = triplet_margin

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        reps = [sentence_features[sentence_feature] for sentence_feature in sentence_features]
        rep_anchor, rep_pos, rep_neg = reps
        distance_pos = self.distance_metric(rep_anchor, rep_pos)
        distance_neg = self.distance_metric(rep_anchor, rep_neg)

        losses = F.relu(distance_pos - distance_neg + self.triplet_margin)
        return losses.mean()

class SoftmaxLoss(nn.Module):
    def __init__(self,
                 sentence_embedding_dimension: int,
                 num_labels: int,
                 concatenation_sent_rep: bool = True,
                 concatenation_sent_difference: bool = True,
                 concatenation_sent_multiplication: bool = False):
        super(SoftmaxLoss, self).__init__()
        self.num_labels = num_labels
        self.concatenation_sent_rep = concatenation_sent_rep
        self.concatenation_sent_difference = concatenation_sent_difference
        self.concatenation_sent_multiplication = concatenation_sent_multiplication

        num_vectors_concatenated = 0
        if concatenation_sent_rep:
            num_vectors_concatenated += 2
        if concatenation_sent_difference:
            num_vectors_concatenated += 1
        if concatenation_sent_multiplication:
            num_vectors_concatenated += 1
        logging.info("Softmax loss: #Vectors concatenated: {}".format(num_vectors_concatenated))
        self.classifier = nn.Linear(num_vectors_concatenated * sentence_embedding_dimension, num_labels)

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        reps = [sentence_features[sentence_feature] for sentence_feature in sentence_features]
        rep_a, rep_b = reps

        vectors_concat = []
        if self.concatenation_sent_rep:
            vectors_concat.append(rep_a)
            vectors_concat.append(rep_b)

        if self.concatenation_sent_difference:
            vectors_concat.append(torch.abs(rep_a - rep_b))

        if self.concatenation_sent_multiplication:
            vectors_concat.append(rep_a * rep_b)

        features = torch.cat(vectors_concat, 1)

        output = self.classifier(features)
        loss_fct = nn.CrossEntropyLoss()

        if labels is not None:
            loss = loss_fct(output, labels.view(-1))
            return loss
        else:
            return reps, output

class CosineSimilarityLoss(nn.Module):
    def __init__(self):
        super(CosineSimilarityLoss, self).__init__()


    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        reps = [sentence_features[sentence_feature] for sentence_feature in sentence_features]
        rep_a, rep_b = reps

        output = torch.cosine_similarity(rep_a, rep_b)
        loss_fct = nn.MSELoss()

        if labels is not None:
            loss = loss_fct(output, labels.view(-1))
            return loss
        else:
            return reps, output
