import argparse
import os
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
    parser.add_argument("--train-langs", type=str, default="en")
    parser.add_argument("--test-langs", type=str, default="en,ar")
    parser.add_argument("--use-crf-trig", default=False, action='store_true')  # True
    parser.add_argument("--use-crf-arg", default=False, action='store_true') # False
    parser.add_argument("--early-stop", type=int, default=10)

    ## Multi-Tasking arguments
    parser.add_argument("--use-multi-task", default=False, action='store_true') # False
    parser.add_argument("--pooling-choice", type=str, default="mean")
    parser.add_argument("--intr-ratio", type=float, default=0.5)
    parser.add_argument("--xintrpath", type=str, default="")
    parser.add_argument("--stsbpath", type=str, default="")
    parser.add_argument("--use-alignment", type=bool, default=False)
    parser.add_argument("--alignment-choice", type=str, default="gd")
    parser.add_argument("--schema-type", type=str, default="BETTER")
    parser.add_argument("--output-dir", type=str)
    parser.add_argument("--data-dir", type=str)
    parser.add_argument("--alignment-dict", type=str)

    return parser.parse_args()


def get_model_name(hp):
    model_save_path = os.path.join(hp.output_dir, hp.schema_type)

    model_save_path = os.path.join(model_save_path, "train-" + hp.train_langs+"_" + "test-" + hp.test_langs)

    if hp.use_alignment:
        model_save_path = os.path.join(model_save_path, hp.alignment_choice)

    options = ""
    if hp.schema_type == "BETTER":
        if hp.use_quad :
            options += "wquad_"
        else:
            options += "noquad_"

        if hp.use_full_trig:
            options += "ftrig_"
        else:
            options += "hdtrig_"

        if hp.use_full_arg:
            options += "farg"
        else:
            options += "hdarg"

        model_save_path = os.path.join(model_save_path, options)

    options = ""
    if hp.use_pos:
        options += "wpos_"
    else:
        options += "nopos_"

    if hp.use_ent:
        options += "went_"
    else:
        options += "noent_"

    if hp.use_gcn:
        options += "wgcn_"
    else:
        options += "nogcn_"

    if hp.use_crf_trig:
        options += "wcrf"
    else:
        options += "nocrf"

    if hp.use_pred_trig:
        options += "predtrig_"
    else:
        options += "goldtrig_"

    model_save_path = os.path.join(model_save_path, options)
    model_save_path = os.path.join(model_save_path, hp.trans_model)

    if hp.use_multi_task:
        options = "wmtALTERNATE"
    else:
        options = "nomt"

    model_save_path = os.path.join(model_save_path, options)

    model_save_path = os.path.join(model_save_path, hp.pooling_choice)

    return model_save_path


if __name__ == "__main__":
    hp = get_args()
    model_save_path = get_model_name(hp)

    print(model_save_path)
