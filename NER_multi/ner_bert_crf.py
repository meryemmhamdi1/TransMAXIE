# %%
import os
cwd = os.getcwd()
import sys
from pathlib import Path
sys.path.append(os.path.join(Path(os.getcwd()).parent, "OnlineAlignment"))
from evaluators import *
import csv
import numpy as np
import os
import time
import torch
import torch.nn as nn
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear
from torch.utils import data
from tqdm import tqdm
import gc

from data_utils import *
from get_arguments import *
from losses import *
from readers import *
from model import *
from torch.utils.tensorboard import SummaryWriter
import ast

hp = get_args()
tokenizer = MODELS_dict[hp.trans_model][1].from_pretrained(MODELS_dict[hp.trans_model][2])
cls_token_id = tokenizer.convert_tokens_to_ids(["[CLS]"])[0]
sep_token_id = tokenizer.convert_tokens_to_ids(["[SEP]"])[0]

max_seq_length = 512
lang_dict = {"en": 0, "ar": 1, "de": 2, "es": 3, "zh": 4}


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
    if lang not in ["de", "zh"]:
        sts_mono_test_data = SentencesDataset(
            examples=sts_reader.get_examples('STS.' + test_lang + '-' + test_lang + '.txt'), model=tokenizer,
            show_progress_bar=True)
        sts_mono_test_dataloader = data.DataLoader(sts_mono_test_data, shuffle=False, batch_size=batch_size)
        sts_mono_test_dataloader.collate_fn = smart_batching_collate
        sts_mono_test_evaluator = EmbeddingSimilarityEvaluator(sts_mono_test_dataloader, name="sts_" + test_lang + "_mono")
    else:
        sts_mono_test_evaluator = None

    print("Loading Intrinsic Data from STS bilingual train/test lang portion")
    if lang != "zh":
        sts_bil_data = SentencesDataset(examples=sts_reader.get_examples('STS.' + train_lang + '-' + test_lang + '.txt'),
                                        model=tokenizer, show_progress_bar=True)
        sts_bil_dataloader = data.DataLoader(sts_bil_data, shuffle=False, batch_size=batch_size)
        sts_bil_dataloader.collate_fn = smart_batching_collate
        sts_bil_evaluator = EmbeddingSimilarityEvaluator(sts_bil_dataloader,
                                                         name='sts_' + train_lang + '-' + test_lang + '_bil')
    else:
        sts_bil_evaluator = None

    return sts_mono_train_evaluator, sts_mono_test_evaluator, sts_bil_evaluator


def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x / warmup
    return 1.0 - x


def evaluate(lang, model, predict_dataloader, test_trip_dataloader, use_multi_task, batch_size, epoch_th, dataset_name):
    print("Evaluating ...")
    # print("***** Running prediction *****")
    model.eval()
    all_preds = []
    all_labels = []
    total = 0
    correct = 0
    start = time.time()
    with torch.no_grad():
        i = 0
        for batch in tqdm(predict_dataloader):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, predict_mask, label_ids = batch
            if use_multi_task:
                test_iter_iterator = iter(test_trip_dataloader)
                try:
                    intr_data = next(test_iter_iterator)
                except StopIteration:
                    test_iter_iterator = iter(test_trip_dataloader)
                    intr_data = next(test_iter_iterator)

                features, labels = batch_to_device(intr_data, model.module.device)
            else:
                features = input_ids
                labels = input_ids

            neg_log_likelihood, intrinsic_loss, _ , predicted_label_seq_ids = model(torch.tensor([lang_dict[lang]]).
                                                                                    to(model.module.device), features,
                                                                                    labels, input_ids, segment_ids,
                                                                                    input_mask, label_ids)

            valid_predicted = torch.masked_select(predicted_label_seq_ids, predict_mask)
            valid_label_ids = torch.masked_select(label_ids, predict_mask)

            all_preds.extend(valid_predicted.tolist())
            all_labels.extend(valid_label_ids.tolist())
            total += len(valid_label_ids)
            correct += valid_predicted.eq(valid_label_ids).sum().item()
            i = i + 1

    test_acc = correct / total
    precision, recall, f1 = f1_score(np.array(all_labels), np.array(all_preds))
    end = time.time()
    print('Epoch:%d, Acc:%.2f, Precision: %.2f, Recall: %.2f, F1: %.2f on %s, Spend:%.3f minutes for evaluation' \
          % (epoch_th, 100. * test_acc, 100. * precision, 100. * recall, 100. * f1, dataset_name, (end - start) / 60.0))
    print('--------------------------------------------------------------')
    return test_acc, f1, precision, recall


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
    if cuda_yes:
        import GPUtil

    device = torch.device("cuda" if cuda_yes else "cpu")

    # random.seed(44)
    np.random.seed(44)
    torch.manual_seed(44)
    if cuda_yes:
        torch.cuda.manual_seed_all(44)

    hp = get_args()

    """ Preparing English Dataset """

    label_list = ['X', '[CLS]', '[SEP]']
    with open(hp.data_dir+"bio_labels.txt") as file:
        labels = [line.rstrip() for line in file.readlines()]

    label_list.extend(labels)

    nerProcessor = NERDataProcessor(label_list)
    label_map = nerProcessor.get_label_map()

    train_langs = hp.train_langs.split(",")
    test_langs = hp.test_langs.split(",")

    train_examples = []
    dev_examples = []
    test_examples = {}
    for lang in train_langs:
        train_examples.extend(nerProcessor.get_train_examples(os.path.join(hp.data_dir, lang)))
        dev_examples.extend(nerProcessor.get_dev_examples(os.path.join(hp.data_dir, lang)))

    total_train_steps = int(len(train_examples) / hp.batch_size / hp.gradient_accumulation_steps * hp.total_train_epochs)

    train_dataset = NerDataset(train_examples, tokenizer, label_map, hp.max_seq_length)
    train_dataloader = data.DataLoader(dataset=train_dataset,
                                       batch_size=hp.batch_size,
                                       shuffle=True,
                                       num_workers=4,
                                       collate_fn=NerDataset.pad)

    print("***** Loaded train and dev sets *****")
    print("  Num training examples = %d" % len(train_examples))
    print("  Num dev examples = %d" % len(dev_examples))
    print("  Batch size = %d" % hp.batch_size)
    print("  Num steps = %d" % total_train_steps)

    dev_dataset = NerDataset(dev_examples, tokenizer, label_map, hp.max_seq_length)
    dev_dataloader = data.DataLoader(dataset=dev_dataset,
                                     batch_size=1,
                                     shuffle=False,
                                     num_workers=4,
                                     collate_fn=NerDataset.pad)

    for lang in test_langs:
        test_examples.update({lang: nerProcessor.get_test_examples(os.path.join(hp.data_dir, lang))})

    test_dataset = {}
    test_dataloader = {}
    for lang in test_langs:
        test_dataset.update({lang: NerDataset(test_examples[lang], tokenizer, label_map, hp.max_seq_length)})

        test_dataloader.update({lang: data.DataLoader(dataset=test_dataset[lang],
                                                      batch_size=1,
                                                      shuffle=False,
                                                      num_workers=4,
                                                      collate_fn=NerDataset.pad)})

    print("***** Loaded test sets *****")
    for lang in test_langs:
        print("Num test examples in language %s = %d" % (lang, len(test_examples[lang])))

    start_label_id = nerProcessor.get_start_label_id()
    stop_label_id = nerProcessor.get_stop_label_id()

    pre_model = MODELS_dict[hp.trans_model][0].from_pretrained(MODELS_dict[hp.trans_model][2])

    if hp.use_alignment:
        alignment_files = ast.literal_eval(hp.alignment_dict)
    else:
        alignment_files = None

    model = TRANSFORMER_CRF_NER(alignment_files, pre_model, hp.use_multi_task, hp.pooling_choice, start_label_id,
                                stop_label_id, len(label_list), hp.max_seq_length, hp.batch_size, device)

    device_ids = [i for i in range(torch.cuda.device_count())]
    model = nn.DataParallel(model, device_ids=device_ids) #DDP(model, device_ids=device_ids)
    model.to(device)

    # %%
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
        if not os.path.exists(output_dir + "/triplets/"):
            os.makedirs(output_dir + "/triplets/")

        if not os.path.exists(output_dir + "/stsb/"):
            os.makedirs(output_dir + "/stsb/")

    print("output_dir:", output_dir)

    if hp.load_checkpoint and os.path.exists(output_dir + '/ner_bert_crf_checkpoint.pt'):
        checkpoint = torch.load(output_dir + '/ner_bert_crf_checkpoint.pt', map_location='cpu')
        start_epoch = checkpoint['epoch'] + 1
        valid_acc_prev = checkpoint['valid_acc']
        valid_f1_prev = checkpoint['valid_f1']
        pretrained_dict = checkpoint['model_state']
        net_state_dict = model.state_dict()
        pretrained_dict_selected = {k: v for k, v in pretrained_dict.items() if k in net_state_dict}
        net_state_dict.update(pretrained_dict_selected)
        model.load_state_dict(net_state_dict)
        print('Loaded the pretrain NER_BERT_CRF model, epoch:', checkpoint['epoch'], 'valid acc:',
              checkpoint['valid_acc'], 'valid f1:', checkpoint['valid_f1'])
    else:
        start_epoch = 0
        valid_acc_prev = 0
        valid_f1_prev = 0

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    new_param = ['transitions', 'hidden2label.weight', 'hidden2label.bias']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay) \
                    and not any(nd in n for nd in new_param)], 'weight_decay': hp.weight_decay_finetune},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay) \
                    and not any(nd in n for nd in new_param)], 'weight_decay': 0.0},
        {'params': [p for n, p in param_optimizer if n in ('transitions', 'hidden2label.weight')] \
            , 'lr': hp.lr0_crf_fc, 'weight_decay': hp.weight_decay_crf_fc},
        {'params': [p for n, p in param_optimizer if n == 'hidden2label.bias'] \
            , 'lr': hp.lr0_crf_fc, 'weight_decay': 0.0}
    ]
    optimizer = BertAdam(optimizer_grouped_parameters, lr=hp.learning_rate0, warmup=hp.warmup_proportion,
                         t_total=total_train_steps)

    global_step_th = int(len(train_examples) / hp.batch_size / hp.gradient_accumulation_steps * start_epoch)

    if hp.use_multi_task:
        train_trip_dataloader, test_trip_dataloader, test_trip_evaluator = load_instrinsic_triplet_data(
            hp.batch_size_intr, hp.xintr_path, test_langs)

        train_iter_iterator = iter(train_trip_dataloader)

        sts_mono_train_evaluator = {}
        sts_mono_test_evaluator = {}
        sts_bil_evaluator = {}
        for lang in test_langs:
            sts_mono_train_evaluator.update({lang: []})
            sts_mono_test_evaluator.update({lang: []})
            sts_bil_evaluator.update({lang: []})
            sts_mono_train_evaluator[lang], sts_mono_test_evaluator[lang], sts_bil_evaluator[lang] = \
                load_instrinsic_xsts_data(hp.batch_size_intr, hp.stsb_path, lang, train_lang="en")

    writer = SummaryWriter(os.path.join(output_dir, 'runs'))

    if not hp.use_multi_task:
        test_trip_dataloader = None

    if hp.do_train:
        no_improv = 0
        for epoch in range(start_epoch, hp.total_train_epochs):
            gc.collect()

            tr_loss = 0
            train_start = time.time()
            model.train()
            optimizer.zero_grad()
            # for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            for step, batch in enumerate(train_dataloader):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, predict_mask, label_ids = batch

                if hp.use_multi_task:
                    try:
                        intr_data = next(train_iter_iterator)
                    except StopIteration:
                        train_iter_iterator = iter(train_trip_dataloader)
                        intr_data = next(train_iter_iterator)

                    features, labels = batch_to_device(intr_data, model.module.device)
                else:
                    features = input_ids
                    labels = input_ids

                neg_log_likelihood, intrinsic_loss, _ , predicted_label_seq_ids = model(torch.tensor([lang_dict["en"]]).
                                                                                        to(model.module.device),
                                                                                        features, labels, input_ids,
                                                                                        segment_ids, input_mask,
                                                                                        label_ids)

                if hp.gradient_accumulation_steps > 1:
                    neg_log_likelihood = neg_log_likelihood / hp.gradient_accumulation_steps

                if hp.use_multi_task:
                    #if epoch <= 5:
                    tr_loss = neg_log_likelihood.mean() + intrinsic_loss.mean()
                    writer.add_scalar('intrinsic_loss', intrinsic_loss.mean(), epoch*step)
                else:
                    tr_loss = neg_log_likelihood.mean()

                tr_loss.backward()
                if (step + 1) % hp.gradient_accumulation_steps == 0:
                    # modify learning rate with special warm up BERT uses
                    lr_this_step = hp.learning_rate0 * warmup_linear(global_step_th / total_train_steps,
                                                                     hp.warmup_proportion)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    for p in model.parameters():
                        if p.grad is not None:
                            del p.grad  # free some memory
                    torch.cuda.empty_cache()
                    global_step_th += 1

                writer.add_scalar('tr_loss', tr_loss, epoch*step)
                writer.add_scalar('neg_log_likelihood', neg_log_likelihood.mean(), epoch*step)

                print("Epoch:{}-{}/{}, Negative loglikelihood: {} ".format(epoch, step, len(train_dataloader),
                                                                           neg_log_likelihood.mean()))

            print('--------------------------------------------------------------')
            print("Epoch:{} completed, Total training's Loss: {}, Spend: {}m".format(epoch, tr_loss,
                                                                                     (time.time() - train_start) / 60.0))

            valid_acc, valid_f1, valid_prec, valid_rec = evaluate(train_langs[0], model, dev_dataloader,
                                                                  test_trip_dataloader, hp.use_multi_task,
                                                                  hp.batch_size, epoch, 'Valid_set')

            print("VALIDATION on {} acc: {}, f1: {}, prec: {}, rec: {}".
                  format(train_langs[0], valid_acc, valid_f1, valid_prec, valid_rec))
            writer.add_scalar('valid_acc', valid_acc, epoch)
            writer.add_scalar('en_valid_f1', valid_f1, epoch)
            writer.add_scalar('en_valid_prec', valid_prec, epoch)
            writer.add_scalar('en_valid_rec', valid_rec, epoch)

            if epoch % 3 == 0:
                for lang in test_langs:
                    test_acc, test_f1, test_prec, test_rec = evaluate(lang, model, test_dataloader[lang],
                                                                      test_trip_dataloader, hp.use_multi_task,
                                                                      hp.batch_size, epoch, lang+'_Test_set')

                    print("TESTING on ", lang,  " acc:", test_acc, " f1:", test_f1,  " prec:", test_prec, " rec:", test_rec)

                    writer.add_scalar('test_'+lang+'_acc', test_acc, epoch)
                    writer.add_scalar('test_'+lang+'_f1', test_f1, epoch)
                    writer.add_scalar('test_'+lang+'_prec', test_prec, epoch)
                    writer.add_scalar('test_'+lang+'_rec', test_rec, epoch)

            if hp.use_multi_task:
                """
                print("================ Evaluation intrinsically the model on test triplet ================")
                acc_cos_test_trip, acc_manhatten_test_trip, acc_euclidean_test_trip = test_trip_evaluator(model, output_path= output_dir + "/triplets/", epoch=epoch, steps=-1)
    
                print(">>>>Metrics on test triplet: acc_cos_test_trip:", acc_cos_test_trip,
                      " acc_manhatten_test_trip:", acc_manhatten_test_trip, " acc_euclidean_test_trip:", acc_euclidean_test_trip)
                writer.add_scalar('acc_cos_test_trip', acc_cos_test_trip, epoch)
                writer.add_scalar('acc_manhatten_test_trip', acc_manhatten_test_trip, epoch)
                writer.add_scalar('acc_euclidean_test_trip', acc_euclidean_test_trip, epoch)
                
                """

                print("================ Evaluation intrinsically the model on sts en_mono ================")
                spearman_cosine_mono_train, spearman_manhattan_mono_train, spearman_euclidean_mono_train, \
                spearman_dot_mono_train = sts_mono_train_evaluator["en"](model, output_path=output_dir + "/stsb/", epoch=epoch,
                                                                         steps=-1)

                print(">>>>Metrics on en-en sts : spearman_cosine_mono_train:", spearman_cosine_mono_train,
                      " spearman_manhattan_mono_train:", spearman_manhattan_mono_train,
                      " spearman_euclidean_mono_train:", spearman_euclidean_mono_train, " spearman_dot_mono_train:",
                      spearman_dot_mono_train)

                writer.add_scalar('spearman_cosine_mono_en_train', spearman_cosine_mono_train, epoch)
                writer.add_scalar('spearman_manhattan_mono_en_train', spearman_manhattan_mono_train, epoch)
                writer.add_scalar('spearman_euclidean_mono_en_train', spearman_euclidean_mono_train, epoch)
                writer.add_scalar('spearman_dot_mono_en_train', spearman_dot_mono_train, epoch)

                for lang in test_langs:

                    if lang not in ["de", "zh"]:
                        print("================ Evaluation intrinsically the model on sts " + lang + "_mono ================")
                        spearman_cosine_mono_test, spearman_manhattan_mono_test, spearman_euclidean_mono_test, \
                        spearman_dot_mono_test = sts_mono_test_evaluator[lang](model, output_path=output_dir + "/stsb/",
                                                                               epoch=epoch, steps=-1)

                        print(">>>>Metrics on ar-ar sts : spearman_cosine_mono_test:", spearman_cosine_mono_test,
                              " spearman_manhattan_mono_test:", spearman_manhattan_mono_test,
                              " spearman_euclidean_mono_test:", spearman_euclidean_mono_test, " spearman_dot_mono_test:",
                              spearman_dot_mono_test)

                        writer.add_scalar('spearman_cosine_mono_test'+lang, spearman_cosine_mono_test, epoch)
                        writer.add_scalar('spearman_manhattan_mono_test'+lang, spearman_manhattan_mono_test, epoch)
                        writer.add_scalar('spearman_euclidean_mono_test'+lang, spearman_euclidean_mono_test, epoch)
                        writer.add_scalar('spearman_dot_mono_test'+lang, spearman_dot_mono_test, epoch)

                    print("================ Evaluation intrinsically the model on sts  en-" + lang +
                          "_bilingual ================")

                    if lang != "zh":
                        spearman_cosine_bil, spearman_manhattan_bil, spearman_euclidean_bil, spearman_dot_bil = \
                            sts_bil_evaluator[lang](model, output_path=output_dir + "/stsb/", epoch=epoch, steps=-1)

                        print(">>>>Metrics on en-ar sts : spearman_cosine_bil:", spearman_cosine_bil,
                              " spearman_manhattan_bil:", spearman_manhattan_bil,
                              " spearman_euclidean_bil:", spearman_euclidean_bil, " spearman_dot_bil:", spearman_dot_bil)

                        writer.add_scalar('spearman_cosine_bil'+lang, spearman_cosine_bil, epoch)
                        writer.add_scalar('spearman_manhattan_bil'+lang, spearman_manhattan_bil, epoch)
                        writer.add_scalar('spearman_euclidean_bil'+lang, spearman_euclidean_bil, epoch)
                        writer.add_scalar('spearman_dot_bil'+lang, spearman_dot_bil, epoch)

            # Save a checkpoint
            if valid_f1 > valid_f1_prev:
                torch.save({'epoch': epoch, 'model_state': model.state_dict(), 'valid_acc': valid_acc,
                            'valid_f1': valid_f1, 'valid_prec': valid_prec, 'valid_rec': valid_rec,
                            'max_seq_length': hp.max_seq_length, 'lower_case': hp.do_lower_case},
                           os.path.join(output_dir, 'ner_bert_crf_checkpoint.pt'))
                valid_f1_prev = valid_f1
            else:
                no_improv += 1

            if no_improv == hp.early_stop_ep:
                print("No Improvement Early Stopping!")
                break

    if hp.do_predict:
        # %%
        '''
        Test_set evaluation and printing prediction using the best epoch of NER_BERT_CRF model
        '''
        checkpoint = torch.load(output_dir + '/ner_bert_crf_checkpoint.pt', map_location='cpu')
        epoch = checkpoint['epoch']
        valid_acc_prev = checkpoint['valid_acc']
        valid_f1_prev = checkpoint['valid_f1']
        pretrained_dict = checkpoint['model_state']
        net_state_dict = model.state_dict()
        pretrained_dict_selected = {k: v for k, v in pretrained_dict.items() if k in net_state_dict}
        net_state_dict.update(pretrained_dict_selected)
        model.load_state_dict(net_state_dict)
        print('Loaded the pretrain  NER_BERT_CRF  model, epoch:', checkpoint['epoch'], 'valid acc:',
              checkpoint['valid_acc'], 'valid f1:', checkpoint['valid_f1'])

        model.to(device)

        # %%
        model.eval()

        with torch.no_grad():
            for lang in test_dataset:
                demon_dataloader = data.DataLoader(dataset=test_dataset[lang],
                                                   batch_size=1,
                                                   shuffle=False,
                                                   num_workers=4,
                                                   collate_fn=NerDataset.pad)
                for batch in demon_dataloader:
                    batch = tuple(t.to(device) for t in batch)
                    input_ids, input_mask, segment_ids, predict_mask, label_ids = batch
                    if hp.use_multi_task:
                        test_iter_iterator = iter(test_trip_dataloader)
                        try:
                            intr_data = next(test_iter_iterator)
                        except StopIteration:
                            test_iter_iterator = iter(test_trip_dataloader)
                            intr_data = next(test_iter_iterator)

                        features, labels = batch_to_device(intr_data, model.module.device)
                    else:
                        features = input_ids
                        labels = input_ids

                    neg_log_likelihood, intrinsic_loss, _, predicted_label_seq_ids = model(
                        torch.tensor([lang_dict[lang]]).to(model.module.device), features, labels, input_ids,
                        segment_ids, input_mask, label_ids)

                    valid_predicted = torch.masked_select(predicted_label_seq_ids, predict_mask)
                    print("Printing predictions for %s >>>> " % lang)
                    for i in range(1):
                        print("Predictions:", predicted_label_seq_ids[i])
                        print("True Labels:", label_ids[i])
                        print("predict_mask[i].cpu().numpy():", predict_mask[i].cpu().numpy())
                        new_ids = predicted_label_seq_ids[i].cpu().numpy()[predict_mask[i].cpu().numpy()]
                        new_ids_true = label_ids[i].cpu().numpy()[predict_mask[i].cpu().numpy()]
                        print("new_ids:", new_ids)
                        print("new_ids_true:", new_ids)
                        print(test_examples[lang][i].words)
                        print(test_examples[lang][i].labels)
                        print(list(map(lambda i: label_list[i], new_ids)))
                        print(list(map(lambda i: label_list[i], new_ids_true)))

                    break
