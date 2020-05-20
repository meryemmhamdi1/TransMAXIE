import torchtext


def read_data(corpus_file, datafields, tokenizer):
    max_seq = 0
    count = 0
    with open(corpus_file, encoding='utf-8') as f:
        examples = []
        words = []
        tokens = ['[CLS]']
        predict_mask = [-1]
        postags = []
        heads = []
        deprels = []
        for line in f:
            if line[0] == '#': # Skip comments.
                continue
            line = line.strip()
            if not line:
                # Blank line for the end of a sentence.
                tokens.append('[SEP]')
                predict_mask.append(-1)
                input_ids = tokenizer.convert_tokens_to_ids(tokens)
                segment_ids = [0] * len(input_ids)
                input_mask = [1] * len(input_ids)

                examples.append(torchtext.data.Example.fromlist([words, postags, heads, deprels, input_ids,
                                                                 segment_ids, input_mask, predict_mask], datafields))
                words = []
                postags = []
                heads = []
                deprels = []
                if len(tokens) > max_seq:
                    max_seq = len(tokens)

                if len(tokens) > 512:
                    count += 1
                tokens = ['[CLS]']
                predict_mask = [-1]
            else:
                columns = line.split('\t')
                # Skip dummy tokens used in ellipsis constructions, and multiword tokens.
                if '.' in columns[0] or '-' in columns[0]:
                    continue
                word = columns[1]
                words.append(word)

                sub_words = tokenizer.tokenize(text=word)
                if not sub_words:
                    sub_words = ['[UNK]']

                tokens.extend(sub_words)

                for j in range(len(sub_words)):
                    if j == 0:
                        predict_mask.append(1)
                    else:
                        predict_mask.append(-1)

                postags.append(columns[4])
                heads.append(int(columns[6]))
                deprels.append(columns[7])

        #print("max_seq:", max_seq)
        #print("count:", count)
        return torchtext.data.Dataset(examples, datafields)

