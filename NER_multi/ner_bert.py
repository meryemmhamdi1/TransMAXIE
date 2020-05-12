# %%
import GPUtil
import csv
import numpy as np
import os
import time
import torch
import torch.nn as nn
from pytorch_pretrained_bert.modeling import BertModel, BertForTokenClassification, BertLayerNorm
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear
from pytorch_pretrained_bert.tokenization import BertTokenizer
from torch.utils import data

from data_utils import *
from evaluators import *
from get_arguments import *
from losses import *
from readers import *
from model import *


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


def load_instrinsic_data(test_lang="ar", train_lang="en"):
    batch_size = 8  # hp.batch_size
    xintr_reader = XTripletReader(xintr_path, s1_col_idx=0, s2_col_idx=1, s3_col_idx=2, delimiter=',',
                                  quoting=csv.QUOTE_MINIMAL, has_header=True)
    sts_reader = STSBDataReader(stsbpath)

    print("Loading Intrinsic Training Data from Cross-Lingual Triplet train portion")
    train_data = SentencesDataset(
        examples=xintr_reader.get_examples(test_lang + '-' + train_lang + '-train.csv', start=0, max_examples=500),
        model=tokenizer, show_progress_bar=True)
    train_dataloader = data.DataLoader(train_data, shuffle=True, batch_size=batch_size)
    # del train_data
    # gc.collect()
    train_dataloader.collate_fn = smart_batching_collate

    # --------------------------------------------------------------------------------
    print("Loading Intrinsic Data from Cross-Lingual Triplet test portion")
    trip_test_data = SentencesDataset(
        examples=xintr_reader.get_examples(test_lang + '-' + train_lang + '-test.csv', max_examples=500),
        model=tokenizer, show_progress_bar=True)
    trip_test_dataloader = data.DataLoader(trip_test_data, shuffle=True, batch_size=batch_size)
    trip_test_dataloader.collate_fn = smart_batching_collate
    test_triplet_evaluator = TripletEvaluator(trip_test_dataloader, main_distance_function=None, name="test")

    print("Loading Intrinsic Data from STS mono train_lang portion")
    sts_mono_train_data = SentencesDataset(
        examples=sts_reader.get_examples('STS.' + train_lang + '-' + train_lang + '.txt'), model=tokenizer,
        show_progress_bar=True)
    sts_mono_train_dataloader = data.DataLoader(sts_mono_train_data, shuffle=False, batch_size=batch_size)
    sts_mono_train_dataloader.collate_fn = smart_batching_collate
    sts_mono_train_evaluator = EmbeddingSimilarityEvaluator(sts_mono_train_dataloader,
                                                            name='sts_' + train_lang + '_mono')

    print("Loading Intrinsic Data from STS mono test_lang portion")
    sts_mono_test_data = SentencesDataset(
        examples=sts_reader.get_examples('STS.' + test_lang + '-' + test_lang + '.txt'), model=tokenizer,
        show_progress_bar=True)
    sts_mono_test_dataloader = data.DataLoader(sts_mono_test_data, shuffle=False, batch_size=batch_size)
    sts_mono_test_dataloader.collate_fn = smart_batching_collate
    sts_mono_test_evaluator = EmbeddingSimilarityEvaluator(sts_mono_test_dataloader, name="sts_" + test_lang + "_mono")

    print("Loading Intrinsic Data from STS bilingual train/test lang portion")
    sts_bil_data = SentencesDataset(examples=sts_reader.get_examples('STS.' + train_lang + '-' + test_lang + '.txt'),
                                    model=tokenizer, show_progress_bar=True)
    sts_bil_dataloader = data.DataLoader(sts_bil_data, shuffle=False, batch_size=batch_size)
    sts_bil_dataloader.collate_fn = smart_batching_collate
    sts_bil_evaluator = EmbeddingSimilarityEvaluator(sts_bil_dataloader,
                                                     name='sts_' + train_lang + '-' + test_lang + '_bil')

    return train_dataloader, trip_test_dataloader, test_triplet_evaluator, sts_mono_train_evaluator, sts_mono_test_evaluator, sts_bil_evaluator


def evaluate(model, predict_dataloader, batch_size, epoch_th, dataset_name):
    # print("***** Running prediction *****")
    model.eval()
    all_preds = []
    all_labels = []
    total = 0
    correct = 0
    start = time.time()
    with torch.no_grad():
        for batch in predict_dataloader:
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, predict_mask, label_ids = batch
            out_scores = model(input_ids, segment_ids, input_mask)
            # out_scores = out_scores.detach().cpu().numpy()
            _, predicted = torch.max(out_scores, -1)
            valid_predicted = torch.masked_select(predicted, predict_mask)
            valid_label_ids = torch.masked_select(label_ids, predict_mask)
            # print(len(valid_label_ids),len(valid_predicted),len(valid_label_ids)==len(valid_predicted))
            all_preds.extend(valid_predicted.tolist())
            all_labels.extend(valid_label_ids.tolist())
            total += len(valid_label_ids)
            correct += valid_predicted.eq(valid_label_ids).sum().item()

    test_acc = correct / total
    precision, recall, f1 = f1_score(np.array(all_labels), np.array(all_preds))
    end = time.time()
    print('Epoch:%d, Acc:%.2f, Precision: %.2f, Recall: %.2f, F1: %.2f on %s, Spend: %.3f minutes for evaluation' \
          % (epoch_th, 100. * test_acc, 100. * precision, 100. * recall, 100. * f1, dataset_name, (end - start) / 60.0))
    print('--------------------------------------------------------------')
    return test_acc, f1


def f1_score(y_true, y_pred):
    '''
    0,1,2,3 are [CLS],[SEP],[X],O
    '''
    ignore_id = 3

    num_proposed = len(y_pred[y_pred > ignore_id])
    num_correct = (np.logical_and(y_true == y_pred, y_true > ignore_id)).sum()
    num_gold = len(y_true[y_true > ignore_id])

    try:
        precision = num_correct / num_proposed
    except ZeroDivisionError:
        precision = 1.0

    try:
        recall = num_correct / num_gold
    except ZeroDivisionError:
        recall = 1.0

    try:
        f1 = 2 * precision * recall / (precision + recall)
    except ZeroDivisionError:
        if precision * recall == 0:
            f1 = 1.0
        else:
            f1 = 0

    return precision, recall, f1


if __name__ == "__main__":
    """ Device Related """
    cuda_yes = torch.cuda.is_available()
    print('Cuda is available?', cuda_yes)
    device = torch.device("cuda" if cuda_yes else "cpu")

    # random.seed(44)
    np.random.seed(44)
    torch.manual_seed(44)
    if cuda_yes:
        torch.cuda.manual_seed_all(44)

    hp = get_args()

    tokenizer = BertTokenizer.from_pretrained(hp.bert_model_scale, do_lower_case=hp.do_lower_case)
    cls_token_id = tokenizer.convert_tokens_to_ids(["[CLS]"])[0]
    sep_token_id = tokenizer.convert_tokens_to_ids(["[SEP]"])[0]

    """ Preparing Dataset """
    conllProcessor = CoNLLDataProcessor()
    label_list = conllProcessor.get_labels()
    label_map = conllProcessor.get_label_map()
    train_examples = conllProcessor.get_train_examples(hp.en_data_dir)
    dev_examples = conllProcessor.get_dev_examples(hp.en_data_dir)
    test_examples = conllProcessor.get_test_examples(hp.en_data_dir)

    total_train_steps = int(len(train_examples) / hp.batch_size / hp.gradient_accumulation_steps * hp.total_train_epochs)

    print("***** Running training *****")
    print("  Num examples = %d" % len(train_examples))
    print("  Batch size = %d" % hp.batch_size)
    print("  Num steps = %d" % total_train_steps)

    train_dataset = NerDataset(train_examples, tokenizer, label_map, hp.max_seq_length)
    dev_dataset = NerDataset(dev_examples, tokenizer, label_map, hp.max_seq_length)
    test_dataset = NerDataset(test_examples, tokenizer, label_map, hp.max_seq_length)

    train_dataloader = data.DataLoader(dataset=train_dataset,
                                       batch_size=hp.batch_size,
                                       shuffle=True,
                                       num_workers=4,
                                       collate_fn=NerDataset.pad)

    dev_dataloader = data.DataLoader(dataset=dev_dataset,
                                     batch_size=hp.batch_size,
                                     shuffle=False,
                                     num_workers=4,
                                     collate_fn=NerDataset.pad)

    test_dataloader = data.DataLoader(dataset=test_dataset,
                                      batch_size=hp.batch_size,
                                      shuffle=False,
                                      num_workers=4,
                                      collate_fn=NerDataset.pad)

    print('*** Use only BertForTokenClassification ***')

    if hp.load_checkpoint and os.path.exists(output_dir + '/ner_bert_checkpoint.pt'):
        checkpoint = torch.load(output_dir + '/ner_bert_checkpoint.pt', map_location='cpu')
        start_epoch = checkpoint['epoch'] + 1
        valid_acc_prev = checkpoint['valid_acc']
        valid_f1_prev = checkpoint['valid_f1']
        model = BertForTokenClassification.from_pretrained(hp.bert_model_scale, state_dict=checkpoint['model_state'], num_labels=len(label_list))
        print('Loaded the pretrain NER_BERT model, epoch:', checkpoint['epoch'], 'valid acc:',
              checkpoint['valid_acc'], 'valid f1:', checkpoint['valid_f1'])
    else:
        start_epoch = 0
        valid_acc_prev = 0
        valid_f1_prev = 0
        model = BertForTokenClassification.from_pretrained(hp.bert_model_scale, num_labels=len(label_list))

    model.to(device)

    # Prepare optimizer
    named_params = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in named_params if not any(nd in n for nd in no_decay)],
         'weight_decay': hp.weight_decay_finetune},
        {'params': [p for n, p in named_params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = BertAdam(optimizer_grouped_parameters, lr=hp.learning_rate0, warmup=hp.warmup_proportion,
                         t_total=total_train_steps) #optim.Adam(model.parameters(), lr=learning_rate0)

    global_step_th = int(len(train_examples) / hp.batch_size / hp.gradient_accumulation_steps * start_epoch)
    for epoch in range(start_epoch, hp.total_train_epochs):
        tr_loss = 0
        train_start = time.time()
        model.train()
        optimizer.zero_grad()
        # for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(device) for t in batch)

            input_ids, input_mask, segment_ids, predict_mask, label_ids = batch

            loss = model(input_ids, segment_ids, input_mask, label_ids)

            if hp.gradient_accumulation_steps > 1:
                loss = loss / hp.gradient_accumulation_steps

            loss.backward()
            tr_loss += loss.item()

            if (step + 1) % hp.gradient_accumulation_steps == 0:
                # modify learning rate with special warm up BERT uses
                lr_this_step = hp.learning_rate0 * warmup_linear(global_step_th / total_train_steps, hp.warmup_proportion)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_this_step
                optimizer.step()
                optimizer.zero_grad()
                global_step_th += 1

            print("Epoch:{}-{}/{}, CrossEntropyLoss: {} ".format(epoch, step, len(train_dataloader), loss.item()))

        print('--------------------------------------------------------------')
        print("Epoch:{} completed, Total training's Loss: {}, Spend: {}m".format(epoch, tr_loss, (time.time()) / 60.0))
        valid_acc, valid_f1 = evaluate(model, dev_dataloader, hp.batch_size, epoch, 'Valid_set')
        # Save a checkpoint
        if valid_f1 > valid_f1_prev:
            # model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
            torch.save({'epoch': epoch, 'model_state': model.state_dict(), 'valid_acc': valid_acc,
                        'valid_f1': valid_f1, 'max_seq_length': hp.max_seq_length, 'lower_case': hp.do_lower_case},
                       os.path.join(output_dir, 'ner_bert_checkpoint.pt'))
            valid_f1_prev = valid_f1

    evaluate(model, test_dataloader, hp.batch_size, hp.total_train_epochs - 1, 'Test_set')

    # %%
    '''
    Test_set prediction using the best epoch of NER_BERT model
    '''
    checkpoint = torch.load(hp.output_dir + '/ner_bert_checkpoint.pt', map_location='cpu')
    epoch = checkpoint['epoch']
    valid_acc_prev = checkpoint['valid_acc']
    valid_f1_prev = checkpoint['valid_f1']
    model = BertForTokenClassification.from_pretrained(hp.bert_model_scale, state_dict=checkpoint['model_state'], num_labels=len(label_list))
    # if os.path.exists(output_dir+'/ner_bert_crf_checkpoint.pt'):
    model.to(device)
    print('Loaded the pretrain NER_BERT model, epoch:', checkpoint['epoch'], 'valid acc:',
          checkpoint['valid_acc'], 'valid f1:', checkpoint['valid_f1'])

    model.to(device)
    # evaluate(model, train_dataloader, batch_size, total_train_epochs-1, 'Train_set')
    evaluate(model, test_dataloader, hp.batch_size, epoch, 'Test_set')
