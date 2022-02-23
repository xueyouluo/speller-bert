import tensorflow as tf 
from gector import GECToRModel
import json
import tokenization
import modeling
import numpy as np
from tqdm import tqdm
import diff_match_patch as dmp_module
from pyltp import SentenceSplitter
import opencc

converter = opencc.OpenCC('t2s.json')
dmp = dmp_module.diff_match_patch()
START_TOKEN = '$START'

def convert_single_example(text,tokenizer,max_seq_len):
    rtokens = tokenizer.tokenize(text)
    tokens = rtokens[:max_seq_len - 1]
    text_len = len(tokens)
    input_ids = tokenizer.convert_tokens_to_ids([START_TOKEN] + tokens)
    input_mask = [1] * len(input_ids)
    segment_ids = [0] * len(input_ids)
    return rtokens,text_len,np.asarray([input_ids]), np.asarray([input_mask]),np.asarray([segment_ids])

def convert_tokens(tokens,tokenizer):
    input_ids = tokenizer.convert_tokens_to_ids([START_TOKEN] + tokens)
    input_mask = [1] * len(input_ids)
    segment_ids = [0] * len(input_ids)
    return np.asarray([input_ids]), np.asarray([input_mask]),np.asarray([segment_ids])



max_seq_len = 128
model_save_path = '/data/xueyou/data/speller/gector/finetune/model_fix_data/gector'
tokenizer = tokenization.FullTokenizer('assets/vocab.txt')
bert_config = modeling.BertConfig.from_json_file('/data/xueyou/data/bert_pretrain/chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_config.json')

input_ids = tf.placeholder(tf.int32,[None,None])
input_mask = tf.placeholder(tf.int32,[None,None])
segment_ids = tf.placeholder(tf.int32,[None,None])
label2id = tokenization.load_vocab('assets/labels.txt')
id2label = {v:k for k,v in label2id.items()}
model = GECToRModel(bert_config,num_labels_classes=len(label2id))
class_probabilities_labels,class_probabilities_d, _, _,_ = model.create_model(
    input_ids=input_ids,
    input_mask=input_mask,
    segment_ids=segment_ids,
    is_training=False
)
tf_config = tf.ConfigProto(log_device_placement=False)
tf_config.gpu_options.allow_growth = True

sess = tf.Session(config=tf_config)
init = tf.global_variables_initializer()
sess.run(init)

tvars = tf.trainable_variables()
saver = tf.train.Saver(var_list=tvars)
saver.restore(sess, model_save_path)

min_error_probability = 0.0
min_probability = 0.0
num_pass=2



def get_target_sent_by_edits(source_tokens, edits):
    target_tokens = source_tokens[:]
    shift_idx = 0
    for edit in edits:
        start, end, label, _ = edit
        target_pos = start + shift_idx
        source_token = target_tokens[target_pos] \
            if len(target_tokens) > target_pos >= 0 else ''
        if label == "":
            del target_tokens[target_pos]
            shift_idx -= 1
        elif start == end:
            word = label.replace("$APPEND_", "")
            target_tokens[target_pos: target_pos] = [word]
            shift_idx += 1
        elif start == end - 1:
            word = label.replace("$REPLACE_", "")
            target_tokens[target_pos] = word

    return target_tokens


def get_token_action(token, index, prob, sugg_token):
    """Get lost of suggested actions for token."""
    # cases when we don't need to do anything
    if prob < min_probability or sugg_token in ['@@UNKNOWN@@','@@PADDING@@' '$KEEP']:
        return None

    if sugg_token.startswith('$REPLACE_') or sugg_token.startswith('$TRANSFORM_') or sugg_token == '$DELETE':
        start_pos = index
        end_pos = index + 1
    elif sugg_token.startswith("$APPEND_") or sugg_token.startswith("$MERGE_"):
        start_pos = index + 1
        end_pos = index + 1

    if sugg_token == "$DELETE":
        sugg_token_clear = ""
    elif sugg_token.startswith('$TRANSFORM_') or sugg_token.startswith("$MERGE_"):
        sugg_token_clear = sugg_token[:]
    else:
        sugg_token_clear = sugg_token[sugg_token.index('_') + 1:]

    return start_pos - 1, end_pos - 1, sugg_token_clear, prob

def postprocess(tokens, idxs, error_prob, probabilities):
    length = min(len(tokens), max_seq_len)
    # print(tokens)
    # print(idxs)
    # print(error_prob)
    # print(probabilities)
    edits = []
    noop_index = 0

    # skip whole sentences if there no errors
    if max(idxs) == 0:
        return tokens

    # skip whole sentence if probability of correctness is not high
    if error_prob < min_error_probability:
        return tokens

    for i in range(length + 1):
        # because of START token
        if i == 0:
            token = START_TOKEN
        else:
            token = tokens[i - 1]
        # skip if there is no error
        if idxs[i] == noop_index:
            continue

        sugg_token = id2label[idxs[i]]
        action = get_token_action(token, i, probabilities[i],
                                        sugg_token)
        if not action:
            continue

        edits.append(action)
    
    return get_target_sent_by_edits(tokens,edits)


def predict_file():
    texts = open('/data/xueyou/data/speller/gector/finetune/eval.src').readlines()
    predicts = []
    for text in tqdm(texts):
        tokens,text_len,_input_ids,_input_mask,_segment_ids = convert_single_example(text,tokenizer,max_seq_len)
        _prob_labels,_prob_d = sess.run([class_probabilities_labels,class_probabilities_d],feed_dict={input_ids:_input_ids,input_mask:_input_mask,segment_ids:_segment_ids})
        preds = np.argmax(_prob_labels[0], axis=-1)[:text_len + 1]
        error_prob = max([p[1] for p in _prob_d[0][:text_len + 1]])
        raw_text_len = text_len 
        pp = [id2label[p] for p in preds]     
        fp = postprocess(tokens[:text_len],preds,error_prob,[_prob_labels[0][i][x] for i,x in enumerate(preds)])
        for i in range(num_pass - 1):
            text_len = len(fp)
            _input_ids,_input_mask,_segment_ids = convert_tokens(fp,tokenizer)
            _prob_labels,_prob_d = sess.run([class_probabilities_labels,class_probabilities_d],feed_dict={input_ids:_input_ids,input_mask:_input_mask,segment_ids:_segment_ids})
            preds = np.argmax(_prob_labels[0], axis=-1)[:text_len + 1]
            error_prob = max([p[1] for p in _prob_d[0][:text_len + 1]])
            pp = [id2label[p] for p in preds]   
            pre_fp = fp  
            fp = postprocess(pre_fp[:text_len],preds,error_prob,[_prob_labels[0][i][x] for i,x in enumerate(preds)])
            if fp == pre_fp:
                break
        pred = ' '.join(fp + tokens[raw_text_len:])
        predicts.append(pred)
    with open('/data/xueyou/data/speller/gector/finetune/predict.txt','w') as f:
        for x in predicts:
            f.write(x + '\n')


def check():
    while True:
        text = input('输入：')
        tokens,text_len,_input_ids,_input_mask,_segment_ids = convert_single_example(text,tokenizer,max_seq_len)
        _prob_labels,_prob_d = sess.run([class_probabilities_labels,class_probabilities_d],feed_dict={input_ids:_input_ids,input_mask:_input_mask,segment_ids:_segment_ids})
        preds = np.argmax(_prob_labels[0], axis=-1)[:text_len + 1]
        error_prob = max([p[1] for p in _prob_d[0][:text_len + 1]])
        pp = [id2label[p] for p in preds] 
        raw_text_len = text_len    
        fp = postprocess(tokens[:text_len],preds,error_prob,[_prob_labels[0][i][x] for i,x in enumerate(preds)])
        for i in range(num_pass - 1):
            text_len = len(fp)
            _input_ids,_input_mask,_segment_ids = convert_tokens(fp,tokenizer)
            _prob_labels,_prob_d = sess.run([class_probabilities_labels,class_probabilities_d],feed_dict={input_ids:_input_ids,input_mask:_input_mask,segment_ids:_segment_ids})
            preds = np.argmax(_prob_labels[0], axis=-1)[:text_len + 1]
            error_prob = max([p[1] for p in _prob_d[0][:text_len + 1]])
            pp = [id2label[p] for p in preds]   
            pre_fp = fp  
            fp = postprocess(pre_fp[:text_len],preds,error_prob,[_prob_labels[0][i][x] for i,x in enumerate(preds)])
            if fp == pre_fp:
                break

        print(' '.join(tokens))
        print(' '.join(fp + tokens[raw_text_len:]))
        
if __name__ == '__main__':
    predict_file()



    

