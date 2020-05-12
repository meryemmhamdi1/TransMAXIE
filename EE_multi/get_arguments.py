import argparse
from transformers import GPT2Model, GPT2Tokenizer, BertModel, BertTokenizer, OpenAIGPTModel, OpenAIGPTTokenizer, \
    CTRLModel, CTRLTokenizer, TransfoXLModel, TransfoXLTokenizer, XLNetModel, XLNetTokenizer, XLMModel, XLMTokenizer, \
    DistilBertModel, DistilBertTokenizer, RobertaModel, RobertaTokenizer, XLMRobertaModel, XLMRobertaTokenizer, \
    AlbertModel, AlbertTokenizer

MODELS_dict = {"BertLarge": (BertModel, BertTokenizer, 'bert-large-uncased'),
               "BertBaseCased": (BertModel, BertTokenizer, 'bert-base-cased'),
               "BertBaseMultilingualCased": (BertModel, BertTokenizer, 'bert-base-multilingual-cased'),
               "Gpt": (OpenAIGPTModel, OpenAIGPTTokenizer, 'openai-gpt'),
               "Gpt2": (GPT2Model, GPT2Tokenizer, 'gpt2'),
               "Ctrl": (CTRLModel, CTRLTokenizer, 'ctrl'),
               "TransfoXL": (TransfoXLModel, TransfoXLTokenizer, 'transfo-xl-wt103'),
               "Xlnet_base": (XLNetModel, XLNetTokenizer, 'xlnet-base-cased'),
               "Xlnet_large": (XLNetModel, XLNetTokenizer, 'xlnet-large-cased'),
               "XLM": (XLMModel, XLMTokenizer, 'xlm-mlm-enfr-1024'),
               "DistilBert_base": (DistilBertModel, DistilBertTokenizer, 'distilbert-base-uncased'),
               "DistilBert_large": (DistilBertModel, DistilBertTokenizer, 'distilbert-large-cased'),
               "Roberta_base": (RobertaModel, RobertaTokenizer, 'roberta-base'),
               "Roberta_large": (RobertaModel, RobertaTokenizer, 'roberta-large'),
               "XLMRoberta_base": (XLMRobertaModel, XLMRobertaTokenizer, 'xlm-roberta-base'),
               "XLMRoberta_large": (XLMRobertaModel, XLMRobertaTokenizer, 'xlm-roberta-large'),
               "ALBERT-base-v1": (AlbertModel, AlbertTokenizer, 'albert-base-v1'),
               "ALBERT-large-v1": (AlbertModel, AlbertTokenizer, 'albert-large-v1'),
               "ALBERT-xlarge-v1": (AlbertModel, AlbertTokenizer, 'albert-xlarge-v1'),
               "ALBERT-xxlarge-v1": (AlbertModel, AlbertTokenizer, 'albert-xxlarge-v1'),
               "ALBERT-base-v2": (AlbertModel, AlbertTokenizer, 'albert-base-v2'),
               "ALBERT-large-v2": (AlbertModel, AlbertTokenizer, 'albert-large-v2'),
               "ALBERT-xlarge-v2": (AlbertModel, AlbertTokenizer, 'albert-xlarge-v2'),
               "ALBERT-xxlarge-v2": (AlbertModel, AlbertTokenizer, 'albert-xxlarge-v2'),
               }


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--lr", type=float, default=0.00002)
    parser.add_argument("--n_epochs", type=int, default=50)
    parser.add_argument("--logdir", type=str, default="logdir")

    parser.add_argument("--use-quad", default=False, action='store_true') #False
    parser.add_argument("--use-full-trig", default=False, action='store_true') # False
    parser.add_argument("--use-full-arg", default=False, action='store_true') #False

    parser.add_argument("--use-pos", default=False, action='store_true') #False
    parser.add_argument("--use-ent", default=False, action='store_true') #False
    parser.add_argument("--use-gcn", default=False, action='store_true') #False
    parser.add_argument("--use-pred-trig", default=False, action='store_true') #False

    parser.add_argument("--trans-model", type=str, default="BertBaseMultilingualCased")
    parser.add_argument("--train-lang", type=str, default="english")
    parser.add_argument("--test-lang", type=str, default="spanish")
    parser.add_argument("--use-crf-trig", default=False, action='store_true')  # True
    parser.add_argument("--use-crf-arg", default=False, action='store_true') # False
    parser.add_argument("--early-stop", type=int, default=10)

    ## Multi-Tasking arguments
    parser.add_argument("--use-multi-task", default=False, action='store_true') # False
    parser.add_argument("--pooling-choice", type=str, default="mean")
    parser.add_argument("--intr-ratio", type=float, default=0.5)
    parser.add_argument("--xintrpath", type=str, default="/nas/clear/users/meryem/Datasets/XSTS")
    parser.add_argument("--stsbpath", type=str, default="/nas/clear/users/meryem/sentence-transformers/STS2017-extended")
    parser.add_argument("--use-alignment", type=bool, default=False)
    parser.add_argument("--schema-type", type=str, default="BETTER")
    parser.add_argument("--alignment-dict", type=str, default="{'en':'', 'ar':'/nas/clear/users/meryem/CLBT/gd.en-ar.trial-model/best_mapping.pkl'}")
    # SVD CLBT "{'en':'', 'ar':'/nas/clear/users/meryem/CLBT/svd.en-ar.trial-model/best_mapping.pkl'}"
    # Schuster "{'en':'', 'ar':'/nas/clear/users/meryem/MUSE/dumped/debug/3b7ahmtdbd/best_mapping.pth'}"

    #parser.add_argument("--trainset", type=str, default="/nas/clear/users/meryem/Datasets/Events/BETTER/internal/abstract-8d-inclusive.train-train.internal.json")
    #parser.add_argument("--devset", type=str, default="/nas/clear/users/meryem/Datasets/Events/BETTER/internal/abstract-8d-inclusive.train-test.internal.json")
    #parser.add_argument("--testset", type=str, default="/nas/clear/users/meryem/Datasets/Events/BETTER/internal/abstract-8d-inclusive.train-dev.internal.json")
    #parser.add_argument("--artestset", type=str, default="/nas/clear/data/better_annotation/0224/02_augmented/original-arabic.ten-sentences.augmented.ar.json")
    parser.add_argument("--trainset", type=str, default="/nas/clear/users/meryem/Datasets/Events/ERE/Preprocessed/JMEE/English/train.json")
    parser.add_argument("--devset", type=str, default="/nas/clear/users/meryem/Datasets/Events/ERE/Preprocessed/JMEE/English/dev.json")
    parser.add_argument("--testset", type=str, default="/nas/clear/users/meryem/Datasets/Events/ERE/Preprocessed/JMEE/English/test.json")
    parser.add_argument("--zhtestset", type=str, default="/nas/clear/users/meryem/Datasets/Events/ERE/Preprocessed/JMEE/Chinese/test.json")
    parser.add_argument("--estestset", type=str, default="/nas/clear/users/meryem/Datasets/Events/ERE/Preprocessed/JMEE/Spanish/test.json")


    return parser.parse_args()


def get_model_name(hp):
    if hp.use_alignment:
        model_save_path = ""
        if "CLBT/gd" in hp.alignment_dict:
            model_save_path += "CLBT_gd"
        elif "CLBT/svd" in hp.alignment_dict:
            model_save_path += "CLBT_svd"
        elif "MUSE" in hp.alignment_dict:
            model_save_path += "Schuster"
    else:
        model_save_path = ""

    if hp.use_quad:
        model_save_path += "wquad+"
    else:
        model_save_path += "noquad+"

    if hp.use_full_trig:
        model_save_path += "ftrig+"
    else:
        model_save_path += "hdtrig+"

    if hp.use_full_arg:
        model_save_path += "farg+"
    else:
        model_save_path += "hdarg+"

    if hp.use_pos:
        model_save_path += "wpos+"
    else:
        model_save_path += "nopos+"

    if hp.use_ent:
        model_save_path += "went+"
    else:
        model_save_path += "noent+"

    if hp.use_gcn:
        model_save_path += "wgcn+"
    else:
        model_save_path += "nogcn+"

    if hp.use_pred_trig:
        model_save_path += "predtrig+"
    else:
        model_save_path += "goldtrig+"

    if hp.use_crf_trig:
        model_save_path += "wcrf+"
    else:
        model_save_path += "nocrf+"

    model_save_path += hp.trans_model

    if hp.use_multi_task:
        model_save_path += "wmtALTERNATE+"
    else:
        model_save_path += "nomt+"

    model_save_path += hp.pooling_choice

    return "Models/" + model_save_path + "/"


if __name__ == "__main__":
    hp = get_args()
    model_save_path = get_model_name(hp)

    print(model_save_path)
