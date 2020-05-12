"""
Training from scratch cross-lingual offline triplet loss + autoencoder
"""
from evaluators import *
from losses import *
from readers import *
import torch
import numpy as np
import csv
from get_arguments import *
from torch.utils import data
import torch.nn as nn
import os
import gc
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import GPUtil

hp = get_args()
max_seq_length = 512
tokenizer = MODELS_dict[hp.trans_model][1].from_pretrained(MODELS_dict[hp.trans_model][2])
cls_token_id = tokenizer.convert_tokens_to_ids(["[CLS]"])[0]
sep_token_id = tokenizer.convert_tokens_to_ids(["[SEP]"])[0]
dict_lang = {"arabic": "ar", "english": "en", "german": "de", "spanish": "es"}


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


def load_instrinsic_data(xintrpath, stsbpath, test_lang="arabic", train_lang="english"):
    batch_size = 32  # hp.batch_size
    xintr_reader = XTripletReader(xintrpath, s1_col_idx=0, s2_col_idx=1, s3_col_idx=2, delimiter=',',
                                  quoting=csv.QUOTE_MINIMAL, has_header=True)
    sts_reader = STSBDataReader(stsbpath)

    print("Loading Intrinsic Training Data from Cross-Lingual Triplet train portion")
    train_data = SentencesDataset(examples=xintr_reader.get_examples(test_lang + '-' + train_lang + '-train.csv', start=0), model=tokenizer, show_progress_bar=True)
    train_dataloader = data.DataLoader(train_data, shuffle=True, batch_size=batch_size)
    # del train_data
    # gc.collect()
    train_dataloader.collate_fn = smart_batching_collate

    # --------------------------------------------------------------------------------
    print("Loading Intrinsic Data from Cross-Lingual Triplet test portion")
    batch_size = 1
    trip_test_data = SentencesDataset(
        examples=xintr_reader.get_examples(test_lang + '-' + train_lang + '-test.csv', start=0, max_examples=20000),
        model=tokenizer, show_progress_bar=True)
    trip_test_dataloader = data.DataLoader(trip_test_data, shuffle=True, batch_size=batch_size)
    trip_test_dataloader.collate_fn = smart_batching_collate
    test_triplet_evaluator = TripletEvaluator(trip_test_dataloader, main_distance_function=None, name="test")

    print("Loading Intrinsic Data from STS mono train_lang portion")
    sts_mono_train_data = SentencesDataset(
        examples=sts_reader.get_examples('STS.' + dict_lang[train_lang] + '-' + dict_lang[train_lang] + '.txt'), model=tokenizer,
        show_progress_bar=True)
    sts_mono_train_dataloader = data.DataLoader(sts_mono_train_data, shuffle=False, batch_size=batch_size)
    sts_mono_train_dataloader.collate_fn = smart_batching_collate
    sts_mono_train_evaluator = EmbeddingSimilarityEvaluator(sts_mono_train_dataloader,
                                                            name='sts_' + train_lang + '_mono')

    print("Loading Intrinsic Data from STS mono test_lang portion")
    sts_mono_test_data = SentencesDataset(
        examples=sts_reader.get_examples('STS.' + dict_lang[test_lang] + '-' + dict_lang[test_lang] + '.txt'), model=tokenizer,
        show_progress_bar=True)
    sts_mono_test_dataloader = data.DataLoader(sts_mono_test_data, shuffle=False, batch_size=batch_size)
    sts_mono_test_dataloader.collate_fn = smart_batching_collate
    sts_mono_test_evaluator = EmbeddingSimilarityEvaluator(sts_mono_test_dataloader, name="sts_" + test_lang + "_mono")

    print("Loading Intrinsic Data from STS bilingual train/test lang portion")
    sts_bil_data = SentencesDataset(examples=sts_reader.get_examples('STS.' + dict_lang[train_lang] + '-' + dict_lang[test_lang] + '.txt'),
                                    model=tokenizer, show_progress_bar=True)
    sts_bil_dataloader = data.DataLoader(sts_bil_data, shuffle=False, batch_size=batch_size)
    sts_bil_dataloader.collate_fn = smart_batching_collate
    sts_bil_evaluator = EmbeddingSimilarityEvaluator(sts_bil_dataloader,
                                                     name='sts_' + dict_lang[train_lang] + '-' + dict_lang[test_lang] + '_bil')

    return train_dataloader, trip_test_dataloader, test_triplet_evaluator, sts_mono_train_evaluator, sts_mono_test_evaluator, sts_bil_evaluator


class MultilingualTransformerTripletAutoencoder(nn.Module):
    def __init__(self, pre_model, pooling_choice):
        super(MultilingualTransformerTripletAutoencoder, self).__init__()
        self.pre_model = pre_model

        strategies = pooling_choice.split(",")
        self.pooling_mode_cls_token = False
        self.pooling_mode_mean_tokens = False
        self.pooling_mode_max_tokens = False
        self.pooling_mode_mean_sqrt_len_tokens = False

        self.hidden_size = 768

        pooling_mode_multiplier = 0
        for strategy in strategies:
            pooling_mode_multiplier += 1
            if strategy == "cls":
                self.pooling_mode_cls_token = True
            elif strategy == "mean":
                self.pooling_mode_mean_tokens = True
            elif strategy == "max":
                self.pooling_mode_max_tokens = True
            elif strategy == "mean_sqrt":
                self.pooling_mode_mean_sqrt_len_tokens = True

        self.pooling_output_dimension = (pooling_mode_multiplier * self.hidden_size)

    def _get_sentence_embeddings(self, training, sentence_features):
        sentence_embeddings = {}
        for i, features in enumerate(sentence_features):
            if training:
                self.pre_model.train()
                lm_output, _ = self.pre_model(input_ids=features["input_ids"],
                                              token_type_ids=features["token_type_ids"],
                                              attention_mask=features['input_mask'])#, output_all_encoded_layers=False)

            else:
                self.pre_model.eval()
                with torch.no_grad():
                    lm_output, _ = self.pre_model(input_ids=features["input_ids"], token_type_ids=features["token_type_ids"],
                                                  attention_mask=features['input_mask'])#, output_all_encoded_layers=False)

            cls_token = lm_output[:, 0, :]

            ## 2. Pooling layer
            output_vectors = []
            if self.pooling_mode_cls_token:  # CLS Pooling
                output_vectors.append(cls_token)

            if self.pooling_mode_max_tokens:  # Max Pooling
                input_mask_expanded = features['input_mask'].unsqueeze(-1).expand(lm_output.size()).float()
                lm_output[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
                max_over_time = torch.max(lm_output, 1)[0]
                output_vectors.append(max_over_time)

            if self.pooling_mode_mean_tokens or self.pooling_mode_mean_sqrt_len_tokens:
                input_mask_expanded = features['input_mask'].unsqueeze(-1).expand(lm_output.size()).float()
                sum_embeddings = torch.sum(lm_output * input_mask_expanded, 1)

                # If tokens are weighted (by WordWeights layer), feature 'token_weights_sum' will be present
                if 'token_weights_sum' in features:
                    sum_mask = features['token_weights_sum'].unsqueeze(-1).expand(sum_embeddings.size())
                else:
                    sum_mask = input_mask_expanded.sum(1)

                sum_mask = torch.clamp(sum_mask, min=1e-9)

                if self.pooling_mode_mean_tokens:
                    output_vectors.append(sum_embeddings / sum_mask)
                if self.pooling_mode_mean_sqrt_len_tokens:
                    output_vectors.append(sum_embeddings / torch.sqrt(sum_mask))

            output_vector = torch.cat(output_vectors, 1)

            sentence_embeddings.update({i: output_vector})

        return sentence_embeddings

    def forward(self, training, sentence_features, labels):
        sentence_embeddings = self._get_sentence_embeddings(training, sentence_features)
        intrinsic_loss = TripletLoss()(sentence_embeddings, labels)
        # Autoencoder loss objective too


        return intrinsic_loss


if __name__=="__main__":
    # Device
    cuda_yes = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_yes else "cpu")
    device_ids = [i for i in range(torch.cuda.device_count())]

    # random.seed(44)
    np.random.seed(44)
    torch.manual_seed(44)
    if cuda_yes:
        torch.cuda.manual_seed_all(44)

    hp = get_args()

    output_dir = "./X_" + hp.trans_model + "_" + hp.test_lang + "-" + hp.train_lang + "_" + hp.pooling_choice + "/"
    # output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not os.path.exists(output_dir + "triplets/"):
        os.makedirs(output_dir + "triplets/")

    if not os.path.exists(output_dir + "stsb/"):
        os.makedirs(output_dir + "stsb/")

    # Loading dataset
    train_trip_dataloader, test_trip_dataloader, test_trip_evaluator, sts_train_mono_evaluator, sts_test_mono_evaluator, sts_bil_evaluator = load_instrinsic_data(hp.xintrpath, hp.stsbpath)

    # Loading the model
    pre_model = MODELS_dict[hp.trans_model][0].from_pretrained(MODELS_dict[hp.trans_model][2])

    # Define the model and distribute over many gpus
    model = MultilingualTransformerTripletAutoencoder(pre_model, hp.pooling_choice)
    device_ids = [i for i in range(torch.cuda.device_count())]
    model = nn.DataParallel(model, device_ids=device_ids) #DDP(model, device_ids=device_ids)
    model.to(device)

    if os.path.exists(output_dir + '/checkpoint.pt'):
        checkpoint = torch.load(output_dir + '/checkpoint.pt', map_location='cpu')
        start_epoch = checkpoint['epoch'] + 1
        net_state_dict = model.state_dict()
        pretrained_dict = checkpoint['model_state']
        pretrained_dict_selected = {k: v for k, v in pretrained_dict.items() if k in net_state_dict}
        net_state_dict.update(pretrained_dict_selected)
        model.load_state_dict(net_state_dict)
    else:
        start_epoch = 0

    # Iterate over epochs/ batches
    writer = SummaryWriter(output_dir + 'runs')
    optimizer = optim.Adam(model.parameters(), lr=hp.lr)
    for epoch in range(start_epoch, hp.n_epochs):
        gc.collect()

        model.train()
        optimizer.zero_grad()
        for step, batch in enumerate(train_trip_dataloader):
            """
            train_iter_iterator = iter(train_trip_dataloader)
            try:
                intr_data = next(train_iter_iterator)
            except StopIteration:
                train_iter_iterator = iter(train_trip_dataloader)
                intr_data = next(train_iter_iterator)
            """
            features, labels = batch_to_device(batch, device)

            # fit the model
            intrinsic_loss = model(True, features, labels)

            print("intrinsic_loss:", intrinsic_loss)

            tr_loss = intrinsic_loss.mean()
            tr_loss.backward()
            GPUtil.showUtilization()

            optimizer.step()
            optimizer.zero_grad()
            for p in model.parameters():
                if p.grad is not None:
                    del p.grad  # free some memory
            torch.cuda.empty_cache()

            writer.add_scalar('tr_loss', tr_loss, epoch*step)

            print("Epoch:{}-{}/{}, Intrinsic loss: {} ".format(epoch, step, len(train_trip_dataloader),
                                                               intrinsic_loss.mean()))

        #print("================ Evaluation intrinsically the model on test triplet ================")
        #acc_cos_test_trip, acc_manhatten_test_trip, acc_euclidean_test_trip = test_trip_evaluator(model, output_path= output_dir + "triplets/", epoch=epoch, steps=-1)

        #print(">>>>Metrics on test triplet: acc_cos_test_trip:", acc_cos_test_trip,
        #      " acc_manhatten_test_trip:", acc_manhatten_test_trip, " acc_euclidean_test_trip:", acc_euclidean_test_trip)
        #writer.add_scalar('acc_cos_test_trip', acc_cos_test_trip, epoch)
        #writer.add_scalar('acc_manhatten_test_trip', acc_manhatten_test_trip, epoch)
        #writer.add_scalar('acc_euclidean_test_trip', acc_euclidean_test_trip, epoch)

        print("================ Evaluation intrinsically the model on sts " + hp.train_lang + "_mono ================")
        spearman_cosine_mono_train, spearman_manhattan_mono_train, spearman_euclidean_mono_train, spearman_dot_mono_train = sts_train_mono_evaluator(model, output_path=output_dir + "stsb/", epoch=epoch, steps=-1)

        print(">>>>Metrics on en-en sts : spearman_cosine_mono_train:", spearman_cosine_mono_train,
              " spearman_manhattan_mono_train:", spearman_manhattan_mono_train,
              " spearman_euclidean_mono_train:", spearman_euclidean_mono_train, " spearman_dot_mono_train:", spearman_dot_mono_train)

        writer.add_scalar('spearman_cosine_mono_train', spearman_cosine_mono_train, epoch)
        writer.add_scalar('spearman_manhattan_mono_train', spearman_manhattan_mono_train, epoch)
        writer.add_scalar('spearman_euclidean_mono_train', spearman_euclidean_mono_train, epoch)
        writer.add_scalar('spearman_dot_mono_train', spearman_dot_mono_train, epoch)


        print("================ Evaluation intrinsically the model on sts " + hp.test_lang + "_mono ================")
        spearman_cosine_mono_test, spearman_manhattan_mono_test, spearman_euclidean_mono_test, spearman_dot_mono_test = sts_test_mono_evaluator(model, output_path=output_dir + "stsb/", epoch=epoch, steps=-1)

        print(">>>>Metrics on ar-ar sts : spearman_cosine_mono_test:", spearman_cosine_mono_test,
              " spearman_manhattan_mono_test:", spearman_manhattan_mono_test,
              " spearman_euclidean_mono_test:", spearman_euclidean_mono_test, " spearman_dot_mono_test:", spearman_dot_mono_test)

        writer.add_scalar('spearman_cosine_mono_test', spearman_cosine_mono_test, epoch)
        writer.add_scalar('spearman_manhattan_mono_test', spearman_manhattan_mono_test, epoch)
        writer.add_scalar('spearman_euclidean_mono_test', spearman_euclidean_mono_test, epoch)
        writer.add_scalar('spearman_dot_mono_test', spearman_dot_mono_test, epoch)

        print("================ Evaluation intrinsically the model on sts " + hp.train_lang + "-" + hp.test_lang + "_bilingual ================")
        spearman_cosine_bil, spearman_manhattan_bil, spearman_euclidean_bil, spearman_dot_bil = sts_bil_evaluator(model, output_path=output_dir + "stsb/", epoch=epoch, steps=-1)

        print(">>>>Metrics on en-ar sts : spearman_cosine_bil:", spearman_cosine_bil,
              " spearman_manhattan_bil:", spearman_manhattan_bil,
              " spearman_euclidean_bil:", spearman_euclidean_bil, " spearman_dot_bil:", spearman_dot_bil)

        writer.add_scalar('spearman_cosine_bil', spearman_cosine_bil, epoch)
        writer.add_scalar('spearman_manhattan_bil', spearman_manhattan_bil, epoch)
        writer.add_scalar('spearman_euclidean_bil', spearman_euclidean_bil, epoch)
        writer.add_scalar('spearman_dot_bil', spearman_dot_bil, epoch)



