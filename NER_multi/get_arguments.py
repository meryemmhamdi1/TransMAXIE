"""


"""
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
    parser.add_argument("--xintr-path", type=str, help="Directory to train/test triplet datasets")
    parser.add_argument("--stsb-path", type=str, help="XSTS evaluation path")
    parser.add_argument("--data-dir", type=str, help="Path of directory for multilingual NER dataset (e.g. Wikiann)")
    parser.add_argument("--output-dir", type=str, help="Path to output directory")

    parser.add_argument("--do-train", default=False, action='store_true',
                        help="Whether to run training.")

    parser.add_argument("--do-eval", default=False, action='store_true',
                        help="Whether to run eval on the dev set.") # True

    parser.add_argument("--do-predict", default=False, action='store_true',
                        help="Whether to run the model in inference mode on the test set.")  # True

    parser.add_argument("--load-checkpoint", default=False, action='store_true',
                        help="Whether load checkpoint file before train model")  # True

    parser.add_argument("--load-max_seq_length", default=False, action='store_true',
                        help="the max sequence of vocabulary that BERT was trained on")  # True

    parser.add_argument("--do-lower-case", default=False, action='store_true')  # False

    parser.add_argument("--use-multi-task", default=False, action='store_true')

    parser.add_argument("--use-alignment", default=False, action='store_true')

    ## Hyperparameters

    parser.add_argument("--max-seq-length", type=int, default=180)
    parser.add_argument("--batch-size", type=int, default=16, help="batch size for main task")
    parser.add_argument("--batch-size-intr", type=int, default=16, help="batch size for intrinsic task")
    parser.add_argument("--learning-rate0", type=float, default=5e-5)
    parser.add_argument("--lr0-crf-fc", type=float, default=8e-5)
    parser.add_argument("--weight-decay-finetune", type=float, default=1e-5)
    parser.add_argument("--weight-decay-crf-fc", type=float, default=5e-6)
    parser.add_argument("--total-train-epochs", type=int, default=15)
    parser.add_argument("--early-stop-ep", type=int, default=3)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--warmup-proportion", type=float, default=0.1)

    parser.add_argument("--pooling-choice", type=str, default="mean")
    parser.add_argument("--trans-model", type=str, default="BertBaseMultilingualCased")

    parser.add_argument("--alignment-choice",  type=str, default="")

    parser.add_argument("--alignment-dict", type=str,
                        help="list of alignment files for each language as string dictionary")

    parser.add_argument("--train-langs", type=str, default="en",
                        help="source training language => use 'en' only for zero-shot on English")

    parser.add_argument("--test-langs", type=str, default="en,ar,es,de,zh",
                        help="Testing languages delimited by commas")

    return parser.parse_args()
