import os
cwd = os.getcwd()
import sys
from pathlib import Path
sys.path.append(os.path.join(Path(os.getcwd()).parent, "OnlineAlignment"))

from OnlineAlignment.evaluators import *

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data

from model import Net

from data_utils import *
from eval import eval
from get_arguments import *
from torch.utils.tensorboard import SummaryWriter
from OnlineAlignment.readers import *
from OnlineAlignment.losses import *
import csv
import numpy as np
import random
import ast

lang_dict = {"en": 0, "ar": 1, "es": 2, "zh": 3}

def load_instrinsic_data(hp):
    batch_size = 8#hp.batch_size
    xintr_reader = XTripletReader(hp.xintrpath, s1_col_idx=0, s2_col_idx=1, s3_col_idx=2, delimiter=',', quoting=csv.QUOTE_MINIMAL, has_header=True)
    sts_reader = STSBDataReader(hp.stsbpath)

    print("Loading Intrinsic Training Data from Cross-Lingual Triplet train portion")
    train_data = SentencesDataset(examples=xintr_reader.get_examples(hp.test_lang+'-'+hp.train_lang+'-train.csv', start=0, max_examples=50), model=tokenizer, show_progress_bar=True)
    train_dataloader = data.DataLoader(train_data, shuffle=True, batch_size=batch_size)
    #del train_data
    #gc.collect()
    train_dataloader.collate_fn = smart_batching_collate

    # --------------------------------------------------------------------------------
    print("Loading Intrinsic Data from Cross-Lingual Triplet test portion")
    trip_test_data = SentencesDataset(examples=xintr_reader.get_examples(hp.test_lang+'-'+hp.train_lang+'-test.csv', max_examples=50), model=tokenizer, show_progress_bar=True)
    trip_test_dataloader = data.DataLoader(trip_test_data, shuffle=True, batch_size=batch_size)
    trip_test_dataloader.collate_fn = smart_batching_collate
    test_triplet_evaluator = TripletEvaluator(trip_test_dataloader, main_distance_function=None, name="test")

    print("Loading Intrinsic Data from STS mono train_lang portion")
    sts_mono_train_data = SentencesDataset(examples=sts_reader.get_examples('STS.'+hp.train_lang+'-'+hp.train_lang+'.txt'), model=tokenizer, show_progress_bar=True)
    sts_mono_train_dataloader = data.DataLoader(sts_mono_train_data, shuffle=False, batch_size=batch_size)
    sts_mono_train_dataloader.collate_fn = smart_batching_collate
    sts_mono_train_evaluator = EmbeddingSimilarityEvaluator(sts_mono_train_dataloader, name='sts_'+hp.train_lang+'_mono')

    print("Loading Intrinsic Data from STS mono test_lang portion")
    sts_mono_test_data = SentencesDataset(examples=sts_reader.get_examples('STS.'+hp.test_lang+'-'+hp.test_lang+'.txt'), model=tokenizer, show_progress_bar=True)
    sts_mono_test_dataloader = data.DataLoader(sts_mono_test_data, shuffle=False, batch_size=batch_size)
    sts_mono_test_dataloader.collate_fn = smart_batching_collate
    sts_mono_test_evaluator = EmbeddingSimilarityEvaluator(sts_mono_test_dataloader, name="sts_"+hp.test_lang+"_mono")

    print("Loading Intrinsic Data from STS bilingual train/test lang portion")
    sts_bil_data = SentencesDataset(examples=sts_reader.get_examples('STS.'+hp.train_lang+'-'+hp.test_lang+'.txt'), model=tokenizer, show_progress_bar=True)
    sts_bil_dataloader = data.DataLoader(sts_bil_data, shuffle=False, batch_size=batch_size)
    sts_bil_dataloader.collate_fn = smart_batching_collate
    sts_bil_evaluator = EmbeddingSimilarityEvaluator(sts_bil_dataloader, name='sts_'+hp.train_lang+'-'+hp.test_lang+'_bil')

    return train_dataloader, trip_test_dataloader, test_triplet_evaluator, sts_mono_train_evaluator, sts_mono_test_evaluator, sts_bil_evaluator

def train(model, iterator, use_multi_task, intr_ratio, train_dataloader, optimizer, loss_choice, criterion, epoch, writer):
    use_multi_task = False
    model.train()
    train_iterator = iter(train_dataloader)
    # Choice of loss method
    if loss_choice == "triplet":
        intrinsic_loss = TripletLoss() # uses Euclidean as a default distance metric with margin of 1
    elif loss_choice == "class":
        intrinsic_loss = SoftmaxLoss(model.hidden_size, 3) # NLI has 3 classes in general
    elif loss_choice == "cossim":
        intrinsic_loss = CosineSimilarityLoss()

    for i, batch in enumerate(iterator):
        tokens_x_2d, entities_x_3d, postags_x_2d, triggers_y_2d, arguments_2d, seqlens_1d, head_indexes_2d, words_2d, triggers_2d, adjm = batch

        optimizer.zero_grad()
        for p in model.parameters():
            if p.grad is not None:
                del p.grad  # free some memory
        torch.cuda.empty_cache()


        tokens_x_2d = torch.LongTensor(tokens_x_2d).to(model.module.device)
        postags_x_2d = torch.LongTensor(postags_x_2d).to(model.module.device)
        print("triggers_y_2d:", triggers_y_2d)
        triggers_y_2d = torch.LongTensor(triggers_y_2d).to(model.module.device)
        head_indexes_2d = torch.LongTensor(head_indexes_2d).to(model.module.device)
        entities_x_3d = torch.LongTensor(entities_x_3d).to(model.module.device)

        if use_multi_task:
            # intrinsic batch
            try:
                intr_data = next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_dataloader)
                intr_data = next(train_iterator)

            features, labels = batch_to_device(intr_data, model.device)
        else:
            features = tokens_x_2d
            labels = tokens_x_2d

        trigger_logits, triggers_y_2d, trigger_hat_2d, argument_hidden, argument_keys = model.module.predict_triggers(lang="en", tokens_x_2d=tokens_x_2d, entities_x_3d=entities_x_3d,
                                                                                                                      postags_x_2d=postags_x_2d, head_indexes_2d=head_indexes_2d,
                                                                                                                      triggers_y_2d=triggers_y_2d, arguments_2d=arguments_2d, adjm=adjm)
        if model.module.use_crf_trig:
            xlen = [max(x) for x in head_indexes_2d]
            batch_size = tokens_x_2d.shape[0]
            SEQ_LEN = tokens_x_2d.shape[1]
            mask = np.zeros(shape=[batch_size, SEQ_LEN], dtype=np.bool)
            for k in range(len(xlen)):
                # Slicing, changing position and removing
                mask[k, :xlen[k]] = True
            mask = torch.BoolTensor(mask).to(model.module.device)

            trigger_loss = model.module.tri_CRF1.neg_log_likelihood_loss(feats=trigger_logits, mask=mask, tags=triggers_y_2d)
        else:
            trigger_logits = trigger_logits.view(-1, trigger_logits.shape[-1])
            trigger_loss = criterion(trigger_logits, triggers_y_2d.view(-1))

        if len(argument_keys) > 0:
            argument_logits, arguments_y_1d, argument_hat_1d, argument_hat_2d = model.module.predict_arguments(argument_hidden, argument_keys, arguments_2d)
            argument_loss = criterion(argument_logits, arguments_y_1d)
            event_loss = trigger_loss + 2 * argument_loss
            if i == 0:
                print("=====sanity check for arguments======")
                print('arguments_y_1d:', arguments_y_1d)
                print("arguments_2d[0]:", arguments_2d[0]['events'])
                print("argument_hat_2d[0]:", argument_hat_2d[0]['events'])
                print("=======================")
        else:
            argument_loss = 100
            event_loss = trigger_loss

        loss = event_loss
        

    
        """
        GPUtil.showUtilization()

        print("features.device:", features.get_device())
        print("labels.device:", labels.get_device())
        print("tokens_x_2d.device:", tokens_x_2d.get_device())
        print("entities_x_3d.device:", entities_x_3d.get_device())
        print("postags_x_2d.device:", postags_x_2d.get_device())
        print("head_indexes_2d.device:", head_indexes_2d.get_device())
        print("head_indexes_2d.device:", head_indexes_2d.get_device())
        """

        #
        model.arguments_2d = arguments_2d
        model.adjm = adjm
        intrinsic_loss, triggers_y_2d, trigger_hat_2d, trigger_loss, argument_loss, event_loss \
            = model(features, labels, lang=torch.tensor([lang_dict["en"]]).to(model.device), tokens_x_2d=tokens_x_2d,
                    entities_x_3d=entities_x_3d, postags_x_2d=postags_x_2d, head_indexes_2d=head_indexes_2d,
                    triggers_y_2d=triggers_y_2d)

        if use_multi_task:
            if random.choice([0,1]) == 0:
                loss = event_loss
            else:
                loss = intrinsic_loss
        else:
            loss = event_loss

        if i == 0:
            print("=====sanity check======")
            print("tokens_x_2d[0]:", tokenizer.convert_ids_to_tokens(tokens_x_2d[0])[:seqlens_1d[0]])
            print("entities_x_3d[0]:", entities_x_3d[0][:seqlens_1d[0]])
            print("postags_x_2d[0]:", postags_x_2d[0][:seqlens_1d[0]])
            print("head_indexes_2d[0]:", head_indexes_2d[0][:seqlens_1d[0]])
            print("triggers_2d[0]:", triggers_2d[0])
            print("triggers_y_2d[0]:", triggers_y_2d.cpu().numpy().tolist()[0][:seqlens_1d[0]])
            print('trigger_hat_2d[0]:', trigger_hat_2d.cpu().numpy().tolist()[0][:seqlens_1d[0]])
            print("seqlens_1d[0]:", seqlens_1d[0])
            print("arguments_2d[0]:", arguments_2d[0])
            print("=======================")

        writer.add_scalar('loss', loss, epoch * len(iterator) + i)
        writer.add_scalar('trig_loss', trigger_loss, epoch * len(iterator) + i)
        writer.add_scalar('arg_loss', argument_loss, epoch * len(iterator) + i)
        writer.add_scalar('intrinsic_loss', argument_loss, epoch * len(iterator) + i)

        if i % 10 == 0:  # monitoring
            print("step: {}, loss: {}".format(i, loss.item()))

        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        with torch.autograd.set_detect_anomaly(True):
            loss.backward()

        torch.cuda.empty_cache()
        optimizer.step()


if __name__ == "__main__":

    hp = get_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    pre_model = MODELS_dict[hp.trans_model][0].from_pretrained(MODELS_dict[hp.trans_model][2])

    print("Loading Intrinsic dataset")
    train_trip_dataloader, test_trip_dataloader, test_trip_evaluator, sts_train_mono_evaluator, sts_test_mono_evaluator, sts_bil_evaluator = load_instrinsic_data(hp)

    print("Loading BETTER dataset")
    train_dataset = ACE2005Dataset(hp.trainset)
    dev_dataset = ACE2005Dataset(hp.devset)
    test_dataset = ACE2005Dataset(hp.testset)
    zh_test_dataset = ACE2005Dataset(hp.zhtestset)
    es_test_dataset = ACE2005Dataset(hp.estestset)

    psemb_size = max([train_dataset.longest(), dev_dataset.longest(), test_dataset.longest()]) + 2

    print("len(postag2idx):", len(postag2idx))

    if hp.use_alignment:
        alignment_files = ast.literal_eval(hp.alignment_dict)
    else:
        alignment_files = None

    model = Net(
        use_multi_tasking=hp.use_multi_task,
        aligning_files=alignment_files,
        pre_model=pre_model,
        pooling_choice=hp.pooling_choice,
        use_pos=hp.use_pos,
        use_ent=hp.use_ent,
        use_gcn=hp.use_gcn,
        use_pred_trig=hp.use_pred_trig,
        use_crf_trig=hp.use_crf_trig,
        use_crf_arg=hp.use_crf_arg,
        device=device,
        trigger_size=len(all_triggers),
        entity_size=len(all_entities),
        all_postags=len(postag2idx),
        psemb_size=psemb_size,
        argument_size=len(all_arguments)
    )
    #if device == 'cuda':
    #    model = model.cuda()

    model = nn.DataParallel(model, device_ids=[0,1])
    model.to(device)


    samples_weight = train_dataset.get_samples_weight()
    sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(samples_weight))

    train_iter = data.DataLoader(dataset=train_dataset,
                                 batch_size=hp.batch_size,
                                 shuffle=False,
                                 sampler=sampler,
                                 num_workers=4,
                                 collate_fn=pad)

    dev_iter = data.DataLoader(dataset=dev_dataset,
                               batch_size=hp.batch_size,
                               shuffle=False,
                               num_workers=4,
                               collate_fn=pad)

    test_iter = data.DataLoader(dataset=test_dataset,
                                batch_size=1,
                                shuffle=False,
                                num_workers=4,
                                collate_fn=pad)

    zh_test_iter = data.DataLoader(dataset=zh_test_dataset,
                                   batch_size=1,
                                   shuffle=False,
                                   num_workers=4,
                                   collate_fn=pad)

    es_test_iter = data.DataLoader(dataset=es_test_dataset,
                                   batch_size=1,
                                   shuffle=False,
                                   num_workers=4,
                                   collate_fn=pad)

    optimizer = optim.Adam(model.parameters(), lr=hp.lr)
    # optimizer = optim.Adadelta(model.parameters(), lr=1.0, weight_decay=1e-2)

    criterion = nn.CrossEntropyLoss(ignore_index=0)

    model_path = get_model_name(hp)

    print("Results saved to model_path:", model_path)

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    if not os.path.exists(model_path + hp.logdir):
        os.makedirs(model_path + hp.logdir)

    if not os.path.exists(model_path + "/triplets/"):
        os.makedirs(model_path + "/triplets/")

    if not os.path.exists(model_path + "/stsb/"):
        os.makedirs(model_path + "/stsb/")

    writer = SummaryWriter(model_path + 'runs')

    eval_progress_f1 = 0
    epoch_wout_progress = 0
    print("hp.use_multi_task:", hp.use_multi_task)
    for epoch in range(1, hp.n_epochs + 1):
        train(model, train_iter, hp.use_multi_task, hp.intr_ratio, train_trip_dataloader, optimizer, "triplet", criterion, epoch, writer)
        batch_size = hp.batch_size
        #xintr_reader = XTripletReader(hp.xintrpath, s1_col_idx=0, s2_col_idx=1, s3_col_idx=2, delimiter=',', quoting=csv.QUOTE_MINIMAL, has_header=True)
        #train_data = SentencesDataset(examples=xintr_reader.get_examples(hp.test_lang+'-'+hp.train_lang+'-train.csv', start=epoch*250, max_examples=epoch*250+250), model=tokenizer, show_progress_bar=True)
        #train_trip_dataloader = data.DataLoader(train_data, shuffle=True, batch_size=batch_size)
        #train_trip_dataloader.collate_fn = smart_batching_collate
        fname = os.path.join(model_path + hp.logdir, str(epoch))
        print(f"=========eval dev at epoch={epoch}=========")
        metric_dev, metric_dev_dict  = eval("en", model, dev_iter, fname + '_dev')

        writer.add_scalar('dev_trig_ident_p', metric_dev_dict["trigger"]["ident"]["p"], epoch)
        writer.add_scalar('dev_trig_ident_r', metric_dev_dict["trigger"]["ident"]["r"], epoch)
        writer.add_scalar('dev_trig_ident_f1', metric_dev_dict["trigger"]["ident"]["f1"], epoch)

        writer.add_scalar('dev_trig_class_p', metric_dev_dict["trigger"]["class"]["p"], epoch)
        writer.add_scalar('dev_trig_class_r', metric_dev_dict["trigger"]["class"]["r"], epoch)
        writer.add_scalar('dev_trig_class_f1', metric_dev_dict["trigger"]["class"]["f1"], epoch)

        writer.add_scalar('dev_arg_ident_p', metric_dev_dict["argument"]["ident"]["p"], epoch)
        writer.add_scalar('dev_arg_ident_r', metric_dev_dict["argument"]["ident"]["r"], epoch)
        writer.add_scalar('dev_arg_ident_f1', metric_dev_dict["argument"]["ident"]["f1"], epoch)

        writer.add_scalar('dev_arg_class_p', metric_dev_dict["argument"]["class"]["p"], epoch)
        writer.add_scalar('dev_arg_class_r', metric_dev_dict["argument"]["class"]["r"], epoch)
        writer.add_scalar('dev_arg_class_f1', metric_dev_dict["argument"]["class"]["f1"], epoch)

        print(f"=========eval test at epoch={epoch}=========")
        metric_test, metric_test_dict = eval("en", model, test_iter, fname + '_test')

        writer.add_scalar('test_trig_ident_p', metric_test_dict["trigger"]["ident"]["p"], epoch)
        writer.add_scalar('test_trig_ident_r', metric_test_dict["trigger"]["ident"]["r"], epoch)
        writer.add_scalar('test_trig_ident_f1', metric_test_dict["trigger"]["ident"]["f1"], epoch)

        writer.add_scalar('test_trig_class_p', metric_test_dict["trigger"]["class"]["p"], epoch)
        writer.add_scalar('test_trig_class_r', metric_test_dict["trigger"]["class"]["r"], epoch)
        writer.add_scalar('test_trig_class_f1', metric_test_dict["trigger"]["class"]["f1"], epoch)

        writer.add_scalar('test_arg_ident_p', metric_test_dict["argument"]["ident"]["p"], epoch)
        writer.add_scalar('test_arg_ident_r', metric_test_dict["argument"]["ident"]["r"], epoch)
        writer.add_scalar('test_arg_ident_f1', metric_test_dict["argument"]["ident"]["f1"], epoch)

        writer.add_scalar('test_arg_class_p', metric_test_dict["argument"]["class"]["p"], epoch)
        writer.add_scalar('test_arg_class_r', metric_test_dict["argument"]["class"]["r"], epoch)
        writer.add_scalar('test_arg_class_f1', metric_test_dict["argument"]["class"]["f1"], epoch)


        print(f"=========CHINESE eval test at epoch={epoch}=========")
        zh_metric_test, zh_metric_test_dict = eval("zh", model, zh_test_iter, fname + '_zh_test')

        writer.add_scalar('test_trig_ident_p_zh', zh_metric_test_dict["trigger"]["ident"]["p"], epoch)
        writer.add_scalar('test_trig_ident_r_zh', zh_metric_test_dict["trigger"]["ident"]["r"], epoch)
        writer.add_scalar('test_trig_ident_f1_zh', zh_metric_test_dict["trigger"]["ident"]["f1"], epoch)

        writer.add_scalar('test_trig_class_p_zh', zh_metric_test_dict["trigger"]["class"]["p"], epoch)
        writer.add_scalar('test_trig_class_r_zh', zh_metric_test_dict["trigger"]["class"]["r"], epoch)
        writer.add_scalar('test_trig_class_f1_zh', zh_metric_test_dict["trigger"]["class"]["f1"], epoch)

        writer.add_scalar('test_arg_ident_p_zh', zh_metric_test_dict["argument"]["ident"]["p"], epoch)
        writer.add_scalar('test_arg_ident_r_zh', zh_metric_test_dict["argument"]["ident"]["r"], epoch)
        writer.add_scalar('test_arg_ident_f1_zh', zh_metric_test_dict["argument"]["ident"]["f1"], epoch)

        writer.add_scalar('test_arg_class_p_zh', zh_metric_test_dict["argument"]["class"]["p"], epoch)
        writer.add_scalar('test_arg_class_r_zh', zh_metric_test_dict["argument"]["class"]["r"], epoch)
        writer.add_scalar('test_arg_class_f1_zh', zh_metric_test_dict["argument"]["class"]["f1"], epoch)

        print(f"=========SPANISH eval test at epoch={epoch}=========")
        es_metric_test, es_metric_test_dict = eval("es", model, zh_test_iter, fname + '_es_test')

        writer.add_scalar('test_trig_ident_p_es', es_metric_test_dict["trigger"]["ident"]["p"], epoch)
        writer.add_scalar('test_trig_ident_r_es', es_metric_test_dict["trigger"]["ident"]["r"], epoch)
        writer.add_scalar('test_trig_ident_f1_es', es_metric_test_dict["trigger"]["ident"]["f1"], epoch)

        writer.add_scalar('test_trig_class_p_es', es_metric_test_dict["trigger"]["class"]["p"], epoch)
        writer.add_scalar('test_trig_class_r_es', es_metric_test_dict["trigger"]["class"]["r"], epoch)
        writer.add_scalar('test_trig_class_f1_es', es_metric_test_dict["trigger"]["class"]["f1"], epoch)

        writer.add_scalar('test_arg_ident_p_es', es_metric_test_dict["argument"]["ident"]["p"], epoch)
        writer.add_scalar('test_arg_ident_r_es', es_metric_test_dict["argument"]["ident"]["r"], epoch)
        writer.add_scalar('test_arg_ident_f1_es', es_metric_test_dict["argument"]["ident"]["f1"], epoch)

        writer.add_scalar('test_arg_class_p_es', es_metric_test_dict["argument"]["class"]["p"], epoch)
        writer.add_scalar('test_arg_class_r_es', es_metric_test_dict["argument"]["class"]["r"], epoch)
        writer.add_scalar('test_arg_class_f1_es', es_metric_test_dict["argument"]["class"]["f1"], epoch)

        ### Evaluation on test portion of triplet dataset
        #print("================ Evaluation intrinsically the model on test triplet ================")
        #acc_cos_test_trip, acc_manhatten_test_trip, acc_euclidean_test_trip = test_trip_evaluator(model, output_path=model_path+"/triplets/", epoch=epoch, steps=-1)

        #print(">>>>Metrics on test triplet: acc_cos_test_trip:", acc_cos_test_trip,
        #      " acc_manhatten_test_trip:", acc_manhatten_test_trip, " acc_euclidean_test_trip:", acc_euclidean_test_trip)
        #writer.add_scalar('acc_cos_test_trip', acc_cos_test_trip, epoch)
        #writer.add_scalar('acc_manhatten_test_trip', acc_manhatten_test_trip, epoch)
        #writer.add_scalar('acc_euclidean_test_trip', acc_euclidean_test_trip, epoch)

        ### Evaluation on STS Benchmark
        print("================ Evaluation intrinsically the model on sts " + hp.train_lang + "_mono ================")
        spearman_cosine_mono_train, spearman_manhattan_mono_train, spearman_euclidean_mono_train, spearman_dot_mono_train = sts_train_mono_evaluator(model, output_path=model_path+"/stsb/", epoch=epoch, steps=-1)

        print(">>>>Metrics on en-en sts : spearman_cosine_mono_train:", spearman_cosine_mono_train,
              " spearman_manhattan_mono_train:", spearman_manhattan_mono_train,
              " spearman_euclidean_mono_train:", spearman_euclidean_mono_train, " spearman_dot_mono_train:", spearman_dot_mono_train)

        writer.add_scalar('spearman_cosine_mono_train', spearman_cosine_mono_train, epoch)
        writer.add_scalar('spearman_manhattan_mono_train', spearman_manhattan_mono_train, epoch)
        writer.add_scalar('spearman_euclidean_mono_train', spearman_euclidean_mono_train, epoch)
        writer.add_scalar('spearman_dot_mono_train', spearman_dot_mono_train, epoch)

        print("================ Evaluation intrinsically the model on sts " + hp.test_lang + "_mono ================")
        spearman_cosine_mono_test, spearman_manhattan_mono_test, spearman_euclidean_mono_test, spearman_dot_mono_test = sts_test_mono_evaluator(model, output_path=model_path+"/stsb/", epoch=epoch, steps=-1)

        print(">>>>Metrics on ar-ar sts : spearman_cosine_mono_test:", spearman_cosine_mono_test,
              " spearman_manhattan_mono_test:", spearman_manhattan_mono_test, " spearman_euclidean_mono_test:",
              spearman_euclidean_mono_test, " spearman_dot_mono_test:", spearman_dot_mono_test)

        writer.add_scalar('spearman_cosine_mono_test', spearman_cosine_mono_test, epoch)
        writer.add_scalar('spearman_manhattan_mono_test', spearman_manhattan_mono_test, epoch)
        writer.add_scalar('spearman_euclidean_mono_test', spearman_euclidean_mono_test, epoch)
        writer.add_scalar('spearman_dot_mono_test', spearman_dot_mono_test, epoch)

        print("================ Evaluation intrinsically the model on sts "+hp.train_lang+"-"+hp.test_lang+ "_bilingual ================")
        spearman_cosine_bil, spearman_manhattan_bil, spearman_euclidean_bil, spearman_dot_bil = sts_bil_evaluator(model, output_path=model_path+"/stsb/", epoch=epoch, steps=-1)

        print(">>>>Metrics on en-ar sts : spearman_cosine_bil:", spearman_cosine_bil, " spearman_manhattan_bil:",
              spearman_manhattan_bil," spearman_euclidean_bil:", spearman_euclidean_bil, " spearman_dot_bil:", spearman_dot_bil)

        writer.add_scalar('spearman_cosine_bil', spearman_cosine_bil, epoch)
        writer.add_scalar('spearman_manhattan_bil', spearman_manhattan_bil, epoch)
        writer.add_scalar('spearman_euclidean_bil', spearman_euclidean_bil, epoch)
        writer.add_scalar('spearman_dot_bil', spearman_dot_bil, epoch)

        if metric_test_dict["argument"]["class"]["f1"] > eval_progress_f1:
            epoch_wout_progress = 0
            eval_progress_f1 = metric_test_dict["argument"]["class"]["f1"]
            torch.save(model, model_path + "latest_model.pt")
        else:
            epoch_wout_progress += 1

        if epoch_wout_progress == hp.early_stop:
            print("Stopped training => no progress")
            break
