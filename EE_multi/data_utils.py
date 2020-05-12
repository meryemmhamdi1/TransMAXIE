import numpy as np
import torch
from torch.utils import data
import json

from consts import NONE, PAD, CLS, SEP, TRIGGERS, ARGUMENTS, ENTITIES, POSTAGS
from utils import build_vocab
from get_arguments import *

# init vocab
all_triggers, trigger2idx, idx2trigger = build_vocab(TRIGGERS)
all_entities, entity2idx, idx2entity = build_vocab(ENTITIES)
all_postags, postag2idx, idx2postag = build_vocab(POSTAGS, BIO_tagging=False)
all_arguments, argument2idx, idx2argument = build_vocab(ARGUMENTS, BIO_tagging=False)

hp = get_args()
tokenizer = MODELS_dict[hp.trans_model][1].from_pretrained(MODELS_dict[hp.trans_model][2])
cls_token_id = tokenizer.convert_tokens_to_ids([tokenizer.cls_token])[0]
sep_token_id = tokenizer.convert_tokens_to_ids([tokenizer.sep_token])[0]
max_seq_length = 512

CUTOFF = 50


class ACE2005Dataset(data.Dataset):
    def __init__(self, fpath):
        self.sent_li, self.adjm_li,  self.entities_li, self.postags_li, self.triggers_li, self.arguments_li, self.examples  = [], [], [], [], [], [], []

        with open(fpath, 'r') as f:
            data = json.load(f)
            for item in data:
                words = item['words']
                self.examples.append(words)
                entities = [[NONE] for _ in range(len(words))]
                triggers = [NONE] * len(words)
                if 'pos-tags' in item:
                    postags = item['pos-tags']
                else:
                    postags = ["NN"] * len(words)
                arguments = {
                    'candidates': [
                        # ex. (5, 6, "entity_type_str"), ...
                    ],
                    'events': {
                        # ex. (1, 3, "trigger_type_str"): [(5, 6, "argument_role_idx"), ...]
                    },
                }

                for entity_mention in item['golden-entity-mentions']:
                    arguments['candidates'].append((entity_mention['start'], entity_mention['end'], entity_mention['entity-type']))
                    if entity_mention['end'] < len(words):
                        end_entity = entity_mention['end']
                    else:
                        end_entity = len(words)
                    for i in range(entity_mention['start'], end_entity):
                        entity_type = entity_mention['entity-type']
                        if i == entity_mention['start']:
                            entity_type = 'B-{}'.format(entity_type)
                        else:
                            entity_type = 'I-{}'.format(entity_type)
                       
                        if len(entities[i]) == 1 and entities[i][0] == NONE:
                            entities[i][0] = entity_type
                        else:
                            entities[i].append(entity_type)

                for event_mention in item['golden-event-mentions']:
                    if event_mention['trigger']['end'] < len(words):
                        end_trig = event_mention['trigger']['end']
                    else:
                        end_trig = len(words)
                    for i in range(event_mention['trigger']['start'], end_trig):
                        trigger_type = event_mention['event_type']
                        if i == event_mention['trigger']['start']:
                            triggers[i] = 'B-{}'.format(trigger_type)
                        else:
                            triggers[i] = 'I-{}'.format(trigger_type)

                    event_key = (event_mention['trigger']['start'], event_mention['trigger']['end'], event_mention['event_type'])
                    arguments['events'][event_key] = []
                    for argument in event_mention['arguments']:
                        role = argument['role']
                        if role.startswith('Time'):
                            role = role.split('-')[0]
                        arguments['events'][event_key].append((argument['start'], argument['end'], argument2idx[role]))

                self.sent_li.append([CLS] + words + [SEP])
                self.entities_li.append([[PAD]] + entities + [[PAD]])
                self.postags_li.append([PAD] + postags + [PAD])
                self.triggers_li.append(triggers)
                self.adjm_li.append([])
                self.arguments_li.append(arguments)

    def __len__(self):
        return len(self.sent_li)

    def __getitem__(self, idx):
        words, entities, postags, triggers, arguments, adjm = self.sent_li[idx], self.entities_li[idx], self.postags_li[idx], self.triggers_li[idx], self.arguments_li[idx], self.adjm_li[idx]

        # We give credits only to the first piece.
        tokens_x, entities_x, postags_x, is_heads = [], [], [], []
        for w, e, p in zip(words, entities, postags):
            tokens = tokenizer.tokenize(w) if w not in [CLS, SEP] else [w]
            tokens_xx = tokenizer.convert_tokens_to_ids(tokens)

            if w in [CLS, SEP]:
                is_head = [0]
            else:
                is_head = [1] + [0] * (len(tokens) - 1)

            p = [p] + [PAD] * (len(tokens) - 1)
            e = [e] + [[PAD]] * (len(tokens) - 1)  # <PAD>: no decision
            p_ids = []
            for postag in p:
                if postag not in postag2idx:
                    postag2idx.update({postag: len(postag2idx)})

                p_ids.append(postag2idx[postag])

            p = p_ids #[postag2idx[postag] for postag in p]

            e = [[entity2idx[entity] for entity in entities] for entities in e]

            tokens_x.extend(tokens_xx), postags_x.extend(p), entities_x.extend(e), is_heads.extend(is_head)

        triggers_y = [trigger2idx[t] for t in triggers]
        head_indexes = []
        for i in range(len(is_heads)):
            if is_heads[i]:
                head_indexes.append(i)

        seqlen = len(tokens_x)

        return tokens_x, entities_x, postags_x, triggers_y, arguments, seqlen, head_indexes, words, triggers, adjm

    def get_samples_weight(self):
        samples_weight = []
        for triggers in self.triggers_li:
            not_none = False
            for trigger in triggers:
                if trigger != NONE:
                    not_none = True
                    break
            if not_none:
                samples_weight.append(5.0)
            else:
                samples_weight.append(1.0)
        return np.array(samples_weight)
    
    def longest(self):
        return max([len(x) for x in self.examples])


def generateAdjMatrix(edgeJsonList, length):
    sparseAdjMatrixPos = [[], [], []]
    sparseAdjMatrixValues = []

    def addedge(type_, from_, to_, value_):
        sparseAdjMatrixPos[0].append(type_)
        sparseAdjMatrixPos[1].append(from_)
        sparseAdjMatrixPos[2].append(to_)
        sparseAdjMatrixValues.append(value_)

    for edgeJson in edgeJsonList:
        fromIndex = edgeJson["source_index"]
        toIndex = edgeJson["target_index"]
        etype = edgeJson["edge_label"]
        if etype == "root" or fromIndex == -1 or toIndex == -1 or fromIndex >= CUTOFF or toIndex >= CUTOFF:
            continue
        addedge(0, fromIndex, toIndex, 1.0)
        addedge(1, toIndex, fromIndex, 1.0)

    for i in range(length):
        addedge(2, i, i, 1.0)

    return sparseAdjMatrixPos, sparseAdjMatrixValues


class BETTERDataset(data.Dataset):
    def __init__(self, fpath):
        self.sent_li, self.entities_li, self.postags_li, self.triggers_li, self.arguments_li, self.examples, self.adjm_li = [], [], [], [], [], [], []
        hp = get_args()

        if hp.use_full_arg:
            span_key = "full_span"
        else:
            span_key = "head_span"

        if hp.use_full_trig:
            trig_span_key = "full_span"
        else:
            trig_span_key = "head_span"

        with open(fpath, 'r') as f:
            parsed = json.load(f)
            count = 0
            for doc in parsed:
                count = count + 1
                sent_ids = []
                arguments_dict = {}
                triggers_dict = {}
                for sent in parsed[doc]["sentences"]:
                    words = sent["words"]
                    entities = [[NONE] for _ in range(len(words))] #[NONE] * len(words)
                    triggers = [NONE] * len(words)

                    postags = sent["pos_tags"]
                    adjm = generateAdjMatrix(sent["dependencies"], len(words))
                    arguments = {
                        'candidates': [
                            # ex. (5, 6, "entity_type_str"), ...
                        ],
                        'events': {
                            # ex. (1, 3, "trigger_type_str"): [(5, 6, "argument_role_idx"), ...]
                        },
                    }

                    sent_id = sent["sent_id"]

                    if "arabic" not in fpath:
                        for entity_mention in sent["mentions"]:
                            head_start_off = entity_mention[span_key]['start_token']
                            head_end_off = entity_mention[span_key]['end_token'] + 1
                            entity_type = entity_mention['entity_type']

                            arguments['candidates'].append((head_start_off, head_end_off, entity_type))

                            if entity_type is not None:
                                for i in range(head_start_off, head_end_off):
                                    if i == head_start_off:
                                        entity_type_bio = 'B-' + entity_type
                                    else:
                                        entity_type_bio = 'I-' + entity_type

                                    if len(entities[i]) == 1 and entities[i][0] == NONE:
                                        entities[i][0] = entity_type_bio
                                    else:
                                        entities[i].append(entity_type_bio)

                    else:
                        start_off_count = 0
                        for _ in sent["words"]:
                            head_start_off = start_off_count
                            head_end_off = start_off_count + 1
                            entity_type = "None"

                            arguments['candidates'].append((head_start_off, head_end_off, entity_type))

                    self.sent_li.append([CLS] + words + [SEP])
                    self.entities_li.append([[PAD]] + entities + [[PAD]]) #([PAD] + entities + [PAD])
                    self.postags_li.append([PAD] + postags + [PAD])
                    self.examples.append(words)
                    self.adjm_li.append(adjm)
                    sent_ids.append(sent_id)

                    triggers_dict.update({sent_id: triggers})
                    arguments_dict.update({sent_id: arguments})

                for sent_id in sent_ids:
                    sent_id = str(sent_id)
                    if sent_id in parsed[doc]["abstract_events"]:
                        for event in parsed[doc]["abstract_events"][sent_id]:
                            quad_class1 = event["properties"]["helpful-harmful"]
                            quad_class2 = event["properties"]["material-verbal"]

                            if hp.use_quad:
                                both_classes = quad_class1 + "_" + quad_class2
                            else:
                                both_classes = "unk_unk"

                            full_span = event["anchors"]["spans"][0]["grounded_span"][trig_span_key]

                            full_trig_start = full_span["start_token"]
                            full_trig_end = full_span["end_token"] + 1

                            for i in range(full_trig_start, full_trig_end):
                                if i == full_trig_start:
                                    triggers_dict[int(sent_id)][i] = 'B-{}'.format(both_classes)
                                else:
                                    triggers_dict[int(sent_id)][i] = 'I-{}'.format(both_classes)

                            event_key = (full_trig_start, full_trig_end, both_classes)
                            arguments_dict[int(sent_id)]["events"][event_key] = []
                            for argument in event["arguments"]:
                                arg_role = argument["role"]
                                head_span = argument["span_set"]["spans"][0]["grounded_span"][span_key]
                                head_arg_start = head_span["start_token"]
                                head_arg_end = head_span["end_token"] + 1

                                arguments_dict[int(sent_id)]["events"][event_key].append\
                                    ((head_arg_start, head_arg_end, argument2idx[arg_role]))

                    self.triggers_li.append(triggers_dict[int(sent_id)])
                    self.arguments_li.append(arguments_dict[int(sent_id)])

    def __len__(self):
        return len(self.sent_li)

    def __getitem__(self, idx):
        words, entities, postags, triggers, arguments, adjm = self.sent_li[idx], self.entities_li[idx], self.postags_li[idx], self.triggers_li[idx], self.arguments_li[idx], self.adjm_li[idx]

        # We give credits only to the first piece.
        tokens_x, entities_x, postags_x, is_heads = [], [], [], []
        for w, e, p in zip(words, entities, postags):
            tokens = tokenizer.tokenize(w) if w not in [CLS, SEP] else [w]
            tokens_xx = tokenizer.convert_tokens_to_ids(tokens)

            if w in [CLS, SEP]:
                is_head = [0]
            else:
                is_head = [1] + [0] * (len(tokens) - 1)

            p = [p] + [PAD] * (len(tokens) - 1)
            e = [[PAD]] * (len(tokens) - 1)  # <PAD>: no decision #[e] + [PAD] * (len(tokens) - 1)
            p_ids = []
            for postag in p:
                if postag not in postag2idx:
                    postag2idx.update({postag: len(postag2idx)})

                p_ids.append(postag2idx[postag])

            p = p_ids #[postag2idx[postag] for postag in p]
            e = [[entity2idx[entity] for entity in entities] for entities in e] #[entity2idx[entity] for entity in e]

            tokens_x.extend(tokens_xx), postags_x.extend(p), entities_x.extend(e), is_heads.extend(is_head)

        triggers_y = [trigger2idx[t] for t in triggers]
        head_indexes = []
        for i in range(len(is_heads)):
            if is_heads[i]:
                head_indexes.append(i)

        seqlen = len(tokens_x)

        return tokens_x, entities_x, postags_x, triggers_y, arguments, seqlen, head_indexes, words, triggers, adjm

    def get_samples_weight(self):
        samples_weight = []
        for triggers in self.triggers_li:
            not_none = False
            for trigger in triggers:
                if trigger != NONE:
                    not_none = True
                    break
            if not_none:
                samples_weight.append(5.0)
            else:
                samples_weight.append(1.0)
        return np.array(samples_weight)

    def longest(self):
        return max([len(x) for x in self.examples])


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


def pad(batch):
    tokens_x_2d, entities_x_3d, postags_x_2d, triggers_y_2d, arguments_2d, seqlens_1d, head_indexes_2d, words_2d, triggers_2d, adjm = list(map(list, zip(*batch)))
    maxlen = np.array(seqlens_1d).max()

    for i in range(len(tokens_x_2d)):
        tokens_x_2d[i] = tokens_x_2d[i] + [0] * (maxlen - len(tokens_x_2d[i])) # TODO padding with [PAD] which index is 3
        postags_x_2d[i] = postags_x_2d[i] + [0] * (maxlen - len(postags_x_2d[i]))
        head_indexes_2d[i] = head_indexes_2d[i] + [0] * (maxlen - len(head_indexes_2d[i]))
        triggers_y_2d[i] = triggers_y_2d[i] + [trigger2idx[PAD]] * (maxlen - len(triggers_y_2d[i]))
        entities_x_3d[i] = entities_x_3d[i] + [[entity2idx[PAD]] for _ in range(maxlen - len(entities_x_3d[i]))] #[entity2idx[PAD]] * (maxlen - len(entities_x_3d[i]))

    return tokens_x_2d, entities_x_3d, postags_x_2d, \
           triggers_y_2d, arguments_2d, \
           seqlens_1d, head_indexes_2d, \
           words_2d, triggers_2d, adjm
