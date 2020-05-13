import os
cwd = os.getcwd()
import sys
from pathlib import Path
sys.path.append(os.path.join(Path(os.getcwd()).parent, "OnlineAlignment"))

from evaluators import *

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data

from model import Net

from data_utils import *
from eval import eval
from get_arguments import *
from torch.utils.tensorboard import SummaryWriter
from readers import *
from losses import *
import csv
import random
import ast

lang_dict = {"en": 0, "ar": 1, "es": 2, "zh": 3}

def load_instrinsic_triplet_data(batch_size, xintr_path, test_langs, train_lang="en", train_max_examples=50000,
                                 test_max_examples=50000):
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

def train_parallel(train_lang, model, iterator, use_multi_task, intr_ratio, train_dataloader, optimizer, loss_choice, criterion, epoch, writer):
    model.train()

    for i, batch in enumerate(iterator):
        tokens_x_2d, entities_x_3d, postags_x_2d, triggers_y_2d, arguments_2d, seqlens_1d, head_indexes_2d, words_2d, triggers_2d, adjm = batch

        optimizer.zero_grad()
        for p in model.parameters():
            if p.grad is not None:
                del p.grad  # free some memory
        torch.cuda.empty_cache()

        tokens_x_2d = torch.LongTensor(tokens_x_2d).to(model.module.device)
        postags_x_2d = torch.LongTensor(postags_x_2d).to(model.module.device)
        head_indexes_2d = torch.LongTensor(head_indexes_2d).to(model.module.device)
        entities_x_3d = torch.LongTensor(entities_x_3d).to(model.module.device)
        triggers_y_2d = torch.LongTensor(triggers_y_2d).to(model.module.device)

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

        model.arguments_2d = arguments_2d
        model.adjm = adjm
        intrinsic_loss, triggers_y_2d, trigger_hat_2d, trigger_loss, argument_loss, event_loss \
            = model(features, labels, lang=torch.tensor([lang_dict[train_lang]]).to(model.device), tokens_x_2d=tokens_x_2d,
                    entities_x_3d=entities_x_3d, postags_x_2d=postags_x_2d, head_indexes_2d=head_indexes_2d,
                    triggers_y_2d=triggers_y_2d)

        if use_multi_task:
            #if random.choice([0,1]) == 0:
            #    loss = event_loss
            #else:
            #    loss = intrinsic_loss
            loss = intr_ratio * intrinsic_loss + event_loss
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

def train(train_lang, model, iterator, use_multi_task, intr_ratio, train_dataloader, optimizer, loss_choice, criterion, epoch, writer):
    model.train()

    for i, batch in enumerate(iterator):
        tokens_x_2d, entities_x_3d, postags_x_2d, triggers_y_2d, arguments_2d, seqlens_1d, head_indexes_2d, words_2d, triggers_2d, adjm = batch

        optimizer.zero_grad()
        for p in model.parameters():
            if p.grad is not None:
                del p.grad  # free some memory
        torch.cuda.empty_cache()


        tokens_x_2d = torch.LongTensor(tokens_x_2d).to(model.module.device)
        postags_x_2d = torch.LongTensor(postags_x_2d).to(model.module.device)
        head_indexes_2d = torch.LongTensor(head_indexes_2d).to(model.module.device)
        entities_x_3d = torch.LongTensor(entities_x_3d).to(model.module.device)
        triggers_y_2d = torch.LongTensor(triggers_y_2d).to(model.module.device)

        if use_multi_task:
            # intrinsic batch
            try:
                intr_data = next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_dataloader)
                intr_data = next(train_iterator)

            features, labels = batch_to_device(intr_data, model.device)

            sentence_embeddings =  model.module._get_sentence_embeddings(features)

            intrinsic_loss =  TripletLoss()(sentence_embeddings, labels)

        model.module.arguments_2d = arguments_2d
        trigger_loss, trigger_logits, triggers_y_2d, trigger_hat_2d, argument_hidden, argument_keys = model.module._predict_triggers(lang=train_lang, tokens_x_2d=tokens_x_2d, entities_x_3d=entities_x_3d,
                                                                                                                      postags_x_2d=postags_x_2d, head_indexes_2d=head_indexes_2d,
                                                                                                                      triggers_y_2d=triggers_y_2d)

        if len(argument_keys) > 0:
            argument_logits, arguments_y_1d, argument_hat_1d, argument_hat_2d = model.module._predict_arguments(argument_hidden, argument_keys)
            argument_loss = criterion(argument_logits, arguments_y_1d)
            event_loss = trigger_loss + 2 * argument_loss
            if i == 0:
                print("=====sanity check for arguments======")
                print('arguments_y_1d:', arguments_y_1d)
                print("arguments_2d[0]:", arguments_2d[0]['events'])
                print("argument_hat_2d[0]:", argument_hat_2d[0]['events'])
                print("=======================")
        else:
            event_loss = trigger_loss


        if use_multi_task:
            #if random.choice([0,1]) == 0:
            #    loss = event_loss
            #else:
            #    loss = intrinsic_loss
            loss = intr_ratio * intrinsic_loss + event_loss
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
    """ Device Related """
    cuda_yes = torch.cuda.is_available()
    print('Cuda is available?', cuda_yes)
    if cuda_yes:
        import GPUtil

    device = torch.device("cuda" if cuda_yes else "cpu")

    hp = get_args()
    test_langs = hp.test_langs.split(",")
    train_lang = hp.train_langs.split(",")[0]

    print("Loading Events dataset")
    if hp.schema_type == "BETTER":
        EventDataset = BETTERDataset
    else:
        EventDataset = EREDataset

    ## Train Dataset (only train on language for now)
    train_dataset = EventDataset(os.path.join(os.path.join(hp.data_dir, train_lang), "train.json"))
    samples_weight = train_dataset.get_samples_weight()
    sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(samples_weight))
    train_iter = data.DataLoader(dataset=train_dataset,
                                 batch_size=hp.batch_size,
                                 shuffle=True,
                                 sampler=sampler,
                                 num_workers=4,
                                 collate_fn=pad)

    # Dev Dataset (only dev on language for now)
    dev_dataset = EventDataset(os.path.join(os.path.join(hp.data_dir, train_lang), "dev.json"))
    dev_iter = data.DataLoader(dataset=dev_dataset,
                               batch_size=hp.batch_size,
                               shuffle=False,
                               num_workers=4,
                               collate_fn=pad)
    test_dataset = {}
    for lang in test_langs:
        test_dataset.update({lang: EventDataset(os.path.join(os.path.join(hp.data_dir, lang), "test.json"))})

    test_iter = {}
    for lang in test_langs:
        test_iter.update({lang: data.DataLoader(dataset=test_dataset,
                                    batch_size=1,
                                    shuffle=False,
                                    num_workers=4,
                                    collate_fn=pad)})

    print("Loading Intrinsic dataset")
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
    else:
        train_trip_dataloader = None

    psemb_size = max([train_dataset.longest(), dev_dataset.longest()] + [test_dataset[lang].longest()for lang in test_dataset]) + 2

    pre_model = MODELS_dict[hp.trans_model][0].from_pretrained(MODELS_dict[hp.trans_model][2])

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

    writer = SummaryWriter(os.path.join(model_path, 'runs'))

    eval_progress_f1 = 0
    epoch_wout_progress = 0
    for epoch in range(1, hp.n_epochs + 1):
        train(train_lang, model, train_iter, hp.use_multi_task, hp.intr_ratio, train_trip_dataloader, optimizer, "triplet", criterion, epoch, writer)
        batch_size = hp.batch_size
        fname = os.path.join(model_path + hp.logdir, str(epoch))
        print(f"=========eval dev at epoch={epoch}=========")
        metric_dev, metric_dev_dict  = eval(train_lang, model, dev_iter, fname + '_dev')

        for task in ["trig", "arg"]:
            for task_type in ["ident", "class"]:
                writer.add_scalar('dev_'+task+'_'+task_type+'_p', metric_dev_dict[task][task_type]["p"], epoch)
                writer.add_scalar('dev_'+task+'_'+task_type+'_r', metric_dev_dict[task][task_type]["r"], epoch)
                writer.add_scalar('dev_'+task+'_'+task_type+'_f1', metric_dev_dict[task][task_type]["f1"], epoch)

        for lang in test_langs:
            print("=========eval test on %s at epoch=%d=========" % (lang, epoch))
            for task in ["trig", "arg"]:
                for task_type in ["ident", "class"]:
                        metric_test, metric_test_dict = eval(lang, model, test_iter, fname + '_test')

                        writer.add_scalar('test_'+task+'_'+task_type+"_"+lang+'_p', metric_test_dict[task][task_type]["p"], epoch)
                        writer.add_scalar('test_'+task+'_'+task_type+"_"+lang+'_r', metric_test_dict[task][task_type]["r"], epoch)
                        writer.add_scalar('test_'+task+'_'+task_type+"_"+lang+'_f1', metric_test_dict[task][task_type]["f1"], epoch)


        ### Evaluation on STS Benchmark
        if hp.use_multi_task:
            ### Evaluation on test portion of triplet dataset
            #print("================ Evaluation intrinsically the model on test triplet ================")
            #acc_cos_test_trip, acc_manhatten_test_trip, acc_euclidean_test_trip = test_trip_evaluator(model, output_path=model_path+"/triplets/", epoch=epoch, steps=-1)

            #print(">>>>Metrics on test triplet: acc_cos_test_trip:", acc_cos_test_trip,
            #      " acc_manhatten_test_trip:", acc_manhatten_test_trip, " acc_euclidean_test_trip:", acc_euclidean_test_trip)
            #writer.add_scalar('acc_cos_test_trip', acc_cos_test_trip, epoch)
            #writer.add_scalar('acc_manhatten_test_trip', acc_manhatten_test_trip, epoch)
            #writer.add_scalar('acc_euclidean_test_trip', acc_euclidean_test_trip, epoch)

            print("================ Evaluation intrinsically the model on sts en_mono ================")
            spearman_cosine_mono_train, spearman_manhattan_mono_train, spearman_euclidean_mono_train, \
            spearman_dot_mono_train = sts_mono_train_evaluator["en"](model, output_path=model_path + "/stsb/", epoch=epoch,
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
                    spearman_dot_mono_test = sts_mono_test_evaluator[lang](model, output_path=model_path + "/stsb/",
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
                        sts_bil_evaluator[lang](model, output_path=model_path + "/stsb/", epoch=epoch, steps=-1)

                    print(">>>>Metrics on en-ar sts : spearman_cosine_bil:", spearman_cosine_bil,
                          " spearman_manhattan_bil:", spearman_manhattan_bil,
                          " spearman_euclidean_bil:", spearman_euclidean_bil, " spearman_dot_bil:", spearman_dot_bil)

                    writer.add_scalar('spearman_cosine_bil'+lang, spearman_cosine_bil, epoch)
                    writer.add_scalar('spearman_manhattan_bil'+lang, spearman_manhattan_bil, epoch)
                    writer.add_scalar('spearman_euclidean_bil'+lang, spearman_euclidean_bil, epoch)
                    writer.add_scalar('spearman_dot_bil'+lang, spearman_dot_bil, epoch)

        if metric_test_dict["argument"]["class"]["f1"] > eval_progress_f1:
            epoch_wout_progress = 0
            eval_progress_f1 = metric_test_dict["argument"]["class"]["f1"]
            torch.save(model, model_path + "latest_model.pt")
        else:
            epoch_wout_progress += 1

        if epoch_wout_progress == hp.early_stop:
            print("Stopped training => no progress")
            break
