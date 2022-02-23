import json
import random
import re
from glob import glob
import functools

import jieba
import tensorflow as tf
import opencc

import tokenization

converter = opencc.OpenCC('t2s.json')
INVALID_VOCAB_IDS = set([0,100,101,102,103,104,105])
resentencesp = re.compile('([﹒﹔﹖﹗．；。！？]["’”」』]{0,2}|：(?=["‘“「『]{1,2}|$))')

# GECToR
PAD = "@@PADDING@@"
UNK = "@@UNKNOWN@@"
START_TOKEN = "$START"
SEQ_DELIMETERS = {"tokens": " ",
                  "labels": "SEPL|||SEPR",
                  "operations": "SEPL__SEPR"}

def split_sentence(sentence):
    s = sentence
    slist = []
    for i in resentencesp.split(s):
        if resentencesp.match(i) and slist:
            slist[-1] += i
        elif i:
            slist.append(i)
    return slist


class BaseDataset:
    def __init__(self,file_name, max_seq_len, tokenizer):
        self.file_name = glob(file_name)
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer

        self.valid_tokens = [k for k,v in tokenizer.vocab.items() if v not in INVALID_VOCAB_IDS and v<=13317]
        self.num_valid_tokens = len(self.valid_tokens)

        self.window_size = 10

        tf.logging.info('load pinyin confusion')
        self.pinyin_confusion = json.load(open('assets/pinyin_confusion.json'))
        tf.logging.info('load shape confusion')
        self.shape_confusion = json.load(open('assets/shape_confusion.json'))
        tf.logging.info('load word confusion')
        words = json.load(open('assets/word_confusion.json'))
        self.word_confusion = {}
        for k,v in words.items():
            for w in v:
                tmp = []
                for ww in v:
                    if ww != w and len(ww) == len(w):
                        tmp.append(ww)
                if tmp:
                    self.word_confusion[w] = tmp

        tf.logging.info('load dimsim confusion')
        self.dimsim_confusion = json.load(open('assets/dimsim_confusion.json'))

        tf.logging.info('load confusion done')

    def random_mask(self,x):
        tokens = self.tokenizer.tokenize(x)
        return tokens, [self.valid_tokens[random.randint(0,self.num_valid_tokens-1)] for i in range(len(tokens))]

    def word_mask(self,x):
        if len(x) > 1:
            if x in self.word_confusion:
                mask = random.choice(self.word_confusion[x])
                return self.tokenizer.tokenize(x),self.tokenizer.tokenize(mask)
            elif x in self.dimsim_confusion:
                mask = random.choice(self.dimsim_confusion[x])
                return self.tokenizer.tokenize(x),self.tokenizer.tokenize(mask)
        return self.char_mask(x)

    def char_mask(self,x):
        tokens = self.tokenizer.tokenize(x)
        mask_tokens = []
        for t in tokens:
            if len(t) == 1 and self.tokenizer.basic_tokenizer._is_chinese_char(ord(t)):
                if random.random() <= 0.9:
                    x = random.choice(self.pinyin_confusion.get(t,[t]))
                else:
                    x = random.choice(self.shape_confusion.get(t,[t]))
                mask_tokens.append(x)
            else:
                mask_tokens.append(t)
        return tokens,mask_tokens


class ConfusionDataset(BaseDataset):

    def get_mask_method(self):
        prob = random.random()
        if prob <= 0.15:
            return lambda x: [self.tokenizer.tokenize(x)] * 2,1
        elif 0.15 < prob <= 0.25:
            return self.random_mask,2
        elif 0.25 < prob <= 0.6:
            return self.word_mask,3
        else:
            return self.char_mask,4

    def confusion_mask(self, tokens):
        src_tokens = [None] * len(tokens)
        tgt_tokens = [None] * len(tokens)
        mask_positions = [None] * len(tokens)
        idxs = list(range(len(tokens)))
        random.shuffle(idxs)
        for i in idxs:
            fn,v = self.get_mask_method()
            if len(tokens[i]) == 1 and tokenization._is_punctuation(tokens[i]):
                if v != 1 or random.random() < 0.9:
                    continue
            raw_token,mask_token = fn(tokens[i])
            if mask_token is None:
                continue
            src_tokens[i] = mask_token
            tgt_tokens[i] = raw_token
            mask_positions[i] = [1] * len(mask_token)
            break

        for i,t in enumerate(tokens):
            if src_tokens[i] is None:
                t = self.tokenizer.tokenize(t)
                src_tokens[i] = t
                tgt_tokens[i] = t
                mask_positions[i] = [0] * len(t)

        return [y for x in src_tokens for y in x],[y for x in tgt_tokens for y in x],[y for x in mask_positions for y in x]

    def generator(self):
        random.shuffle(self.file_name)
        tf.logging.info(f'# files: {self.file_name}')
        for fname in self.file_name:
            for line in open(fname):
                try:
                    x = json.loads(line)
                except:
                    continue
                text = x['text'].lower()
                text = converter.convert(text)
                if len(text) < 10:
                    continue
                sents = split_sentence(text)

                srcs = []
                tgts = []
                masks = []

                for text in sents:
                    tokens = [t.strip() for t in jieba.lcut(text) if t.strip()]
                    src_tokens = []
                    tgt_tokens = []
                    mask = []
                    for i in range(0,len(tokens),self.window_size):
                        src,tgt,_mask = self.confusion_mask(tokens[i:i+self.window_size])
                        src_tokens.extend(src)
                        tgt_tokens.extend(tgt)
                        mask.extend(_mask)
                    if len(src_tokens) != len(tgt_tokens) and len(tgt_tokens) != len(mask):
                        continue
                    srcs.append(src_tokens)
                    tgts.append(tgt_tokens)
                    masks.append(mask)

            

                def convert(input_ids, label_ids, mask_ids):

                    input_ids = input_ids[:self.max_seq_len - 2]
                    label_ids = label_ids[:self.max_seq_len - 2]
                    mask_ids = mask_ids[:self.max_seq_len - 2]

                    input_ids = self.tokenizer.convert_tokens_to_ids(['[CLS]'] + input_ids + ['[SEP]'])
                    input_mask = [1] * len(input_ids)
                    segment_ids = [0] * len(input_ids)
                    label_ids = self.tokenizer.convert_tokens_to_ids(['[CLS]'] + label_ids + ['[SEP]'])
                    lmask = [0] + mask_ids + [0]

                    while len(input_ids) < self.max_seq_len:
                        input_ids.append(0)
                        input_mask.append(0)
                        segment_ids.append(0)
                        lmask.append(0)
                        label_ids.append(0)
                    
                    return {
                            "input_ids": input_ids,
                            "input_mask": input_mask,
                            'segment_ids': segment_ids,
                            'lmask': lmask,
                            'label_ids': label_ids
                        }

                input_ids = []
                label_ids = []
                mask_ids = []
                for src_tokens,tgt_tokens,mask in zip(srcs,tgts,masks):
                    # print(len(input_ids),len(src_tokens))
                    if len(input_ids) + len(src_tokens) > self.max_seq_len - 2:
                        if len(input_ids) > 10:
                            yield convert(input_ids,label_ids,mask_ids)
                        input_ids = []
                        label_ids = []
                        mask_ids = []
                    else:
                        input_ids.extend(src_tokens)
                        label_ids.extend(tgt_tokens)
                        mask_ids.extend(mask)
                
                if len(input_ids) >= 10:
                    yield convert(input_ids,label_ids,mask_ids)

    def make_finetune_dataset(self, batch_size=32):
        def _generator():
            for fname in self.file_name:
                for line in open(fname):
                    try:
                        x = json.loads(line)
                    except:
                        continue
                    src,tgt = x['src'],x['tgt']
                    src = converter.convert(src)
                    tgt = converter.convert(tgt)
                    src,tgt = self.tokenizer.tokenize(src),self.tokenizer.tokenize(tgt)
                    if len(src) != len(tgt):
                        continue
                    src = self.tokenizer.convert_tokens_to_ids(src)[:self.max_seq_len-2]
                    tgt = self.tokenizer.convert_tokens_to_ids(tgt)[:self.max_seq_len-2]
                    input_ids = [101] + src + [102]
                    input_mask = [1] * len(input_ids)
                    segment_ids = [0] * len(input_ids)
                    label_ids = [101] + tgt + [102]
                    lmask = [0] + [1] * len(src) + [0]

                    while len(input_ids) < self.max_seq_len:
                        input_ids.append(0)
                        input_mask.append(0)
                        segment_ids.append(0)
                        lmask.append(0)
                        label_ids.append(0)

                    
                    
                    yield {
                            "input_ids": input_ids,
                            "input_mask": input_mask,
                            'segment_ids': segment_ids,
                            'lmask': lmask,
                            'label_ids': label_ids
                        }

        dataset = tf.data.Dataset.from_generator(
            _generator,
            output_types={
                "input_ids": tf.int32,
                "input_mask": tf.int32,
                'segment_ids': tf.int32,
                'lmask': tf.int32,
                'label_ids': tf.int32
            },
            output_shapes={
                "input_ids": tf.TensorShape([self.max_seq_len]),
                "input_mask": tf.TensorShape([self.max_seq_len]),
                'segment_ids': tf.TensorShape([self.max_seq_len]),
                'lmask': tf.TensorShape([self.max_seq_len]),
                'label_ids': tf.TensorShape([self.max_seq_len])
            }
        )
        dataset = dataset.repeat().shuffle(buffer_size=1000)
        dataset = dataset.batch(batch_size).prefetch(50)
        return dataset

    def make_train_dataset(self,batch_size=32):
        dataset = tf.data.Dataset.from_generator(
            self.generator,
            output_types={
                "input_ids": tf.int32,
                "input_mask": tf.int32,
                'segment_ids': tf.int32,
                'lmask': tf.int32,
                'label_ids': tf.int32
            },
            output_shapes={
                "input_ids": tf.TensorShape([self.max_seq_len]),
                "input_mask": tf.TensorShape([self.max_seq_len]),
                'segment_ids': tf.TensorShape([self.max_seq_len]),
                'lmask': tf.TensorShape([self.max_seq_len]),
                'label_ids': tf.TensorShape([self.max_seq_len])
            }
        )
        dataset = dataset.repeat().shuffle(buffer_size=1000)
        dataset = dataset.batch(batch_size).prefetch(50)
        return dataset


class GECToRDataset:
    def __init__(self,file_name, max_seq_len, tokenizer):
        self.file_name = glob(file_name)
        self.max_seq_len = max_seq_len
        self._delimeters = SEQ_DELIMETERS
        self.tokenizer = tokenizer

        self.label2id = tokenization.load_vocab('assets/labels.txt')
        self.dlabel2id = {'CORRECT':0,'INCORRECT':1}

    def convert(self, tokens, labels, dtags):
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        segment_ids = [0] * len(input_ids)

        label_ids = [self.label2id.get(t,self.label2id[UNK]) for t in labels]
        dtag_ids = [self.dlabel2id[t] for t in dtags]

        while len(input_ids) < self.max_seq_len:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            label_ids.append(0)
            dtag_ids.append(0)
        
        return {
                "input_ids": input_ids,
                "input_mask": input_mask,
                'segment_ids': segment_ids,
                'dtags': dtag_ids,
                'label_ids': label_ids
            }

    def generator(self):
        random.shuffle(self.file_name)
        tf.logging.info(f'# files: {self.file_name}')
        for fname in self.file_name:
            for line in open(fname):
                line = line.strip('\n')
                tokens_and_tags = [pair.rsplit(self._delimeters['labels'], 1)
                                   for pair in line.split(self._delimeters['tokens'])]
                
                tokens = [token for token, tag in tokens_and_tags]
                tags = [tag for token, tag in tokens_and_tags]

                tokens = tokens[:self.max_seq_len]
                tags = tags[:self.max_seq_len]

                op_del = self._delimeters['operations']

                labels = [x.split(op_del) for x in tags]
                # keep one 
                labels = [x[0] for x in labels]
                detect_tags = ["CORRECT" if label == "$KEEP" else "INCORRECT" for label in labels]

                yield self.convert(tokens,labels,detect_tags)

    def make_train_dataset(self,batch_size=32):
        dataset = tf.data.Dataset.from_generator(
            self.generator,
            output_types={
                "input_ids": tf.int32,
                "input_mask": tf.int32,
                'segment_ids': tf.int32,
                'dtags': tf.int32,
                'label_ids': tf.int32
            },
            output_shapes={
                "input_ids": tf.TensorShape([self.max_seq_len]),
                "input_mask": tf.TensorShape([self.max_seq_len]),
                'segment_ids': tf.TensorShape([self.max_seq_len]),
                'dtags': tf.TensorShape([self.max_seq_len]),
                'label_ids': tf.TensorShape([self.max_seq_len])
            }
        )
        dataset = dataset.repeat().shuffle(buffer_size=5000)
        dataset = dataset.batch(batch_size).prefetch(50)
        return dataset



