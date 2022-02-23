import tensorflow as tf 
import cbert
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

def score_f(ans, print_flg=False, only_check=False, out_dir=''):
    fout = open('%s/pred.txt' % out_dir, 'w', encoding="utf-8")
    total_gold_err, total_pred_err, right_pred_err = 0, 0, 0
    total_cnt = 0
    check_right_pred_err = 0
    inputs, golds, preds = ans
    assert len(inputs) == len(golds)
    assert len(golds) == len(preds)
    for ori, god, prd in zip(inputs, golds, preds):
        ori_txt = str(ori)
        god_txt = str(god) #''.join(list(map(str, god)))
        prd_txt = str(prd) #''.join(list(map(str, prd)))
        if print_flg is True:
            print(ori_txt, '\t', god_txt, '\t', prd_txt)
        total_cnt += 1
        if ori_txt == god_txt and ori_txt == prd_txt:
            continue
        if prd_txt != god_txt:
            fout.writelines('%s\t%s\t%s\n' % (ori_txt, god_txt, prd_txt)) 
        if ori != god:
            total_gold_err += 1
        if prd != ori:
            total_pred_err += 1
        if (ori != god) and (prd != ori):
            check_right_pred_err += 1
            if god == prd:
                right_pred_err += 1
    fout.close()

    # print(total_pred_err,total_gold_err,total_cnt,check_right_pred_err)
    print('误报率',(total_pred_err - right_pred_err) / (total_cnt - total_gold_err))

    #check p, r, f
    p = 1. * check_right_pred_err / (total_pred_err + 0.001)
    r = 1. * check_right_pred_err / (total_gold_err + 0.001)
    f = 2 * p * r / (p + r +  1e-13)
    print('token check: p=%.3f, r=%.3f, f=%.3f' % (p, r, f))
    if only_check is True:
        return p, r, f

    #correction p, r, f
    #p = 1. * right_pred_err / (total_pred_err + 0.001)
    pc = 1. * right_pred_err / (check_right_pred_err + 0.001)
    rc = 1. * right_pred_err / (total_gold_err + 0.001)
    fc = 2 * pc * rc / (pc + rc + 1e-13) 
    print('token correction: p=%.3f, r=%.3f, f=%.3f' % (pc, rc, fc))
    return p, r, f


def convert_single_example(text,tokenizer,max_seq_len):
    rtokens = tokenizer.tokenize(text)
    tokens = rtokens[:max_seq_len - 2]
    text_len = len(tokens)
    input_ids = tokenizer.convert_tokens_to_ids(['[CLS]'] + tokens + ['[SEP]'])
    input_mask = [1] * len(input_ids)
    segment_ids = [0] * len(input_ids)
    return rtokens,text_len,np.asarray([input_ids]), np.asarray([input_mask]),np.asarray([segment_ids])



max_seq_len = 128
model_save_path = '/nfs/users/xueyou/data/speller/cbert/models/finetune_plome_cbert/cbert'
tokenizer = tokenization.SimpleTokenizer('assets/vocab.txt')
bert_config = modeling.BertConfig.from_json_file('/data/xueyou/data/bert_pretrain/chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_config.json')

input_ids = tf.placeholder(tf.int32,[None,None])
input_mask = tf.placeholder(tf.int32,[None,None])
segment_ids = tf.placeholder(tf.int32,[None,None])

model = cbert.CBert(bert_config,tie_embedding=True,cbert=True)
logits, probs, _, _ = model.create_model(
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


def eval_corpus_500_score():
    corpus = json.load(open('/nfs/users/xueyou/github/pycorrector/pycorrector/data/eval_corpus.json'))
    TP = 0.0
    FP = 0.0
    FN = 0.0
    TN = 0.0
    total_num = 0
    for data_dict in tqdm(corpus):
        src = text = data_dict.get('text', '').lower().replace(' ','')
        tgt = data_dict.get('correction', '').lower().replace(' ','')

        tokens,text_len,_input_ids,_input_mask,_segment_ids = convert_single_example(text,tokenizer,max_seq_len)
        _pred_probs = sess.run(probs,feed_dict={input_ids:_input_ids,input_mask:_input_mask,segment_ids:_segment_ids})
        preds = np.argmax(_pred_probs[0], axis=-1)[1:text_len + 1]
        pp = tokenizer.convert_ids_to_tokens(preds)
        # 非中文字符不纠错
        pp = [pp[i] if len(pp[i]) == 1 and tokenizer.basic_tokenizer._is_chinese_char(ord(pp[i])) and (len(text[i])==1 and tokenizer.basic_tokenizer._is_chinese_char(ord(text[i]))) else text[i] for i,x in enumerate(pp)]
        pp = [t.strip('##') for t in pp]
        tgt_pred = ''.join(pp)
        src = src[:len(tgt_pred)]
        tgt = tgt[:len(tgt_pred)]
        try:
            assert len(src) == len(tgt) == len(tgt_pred)
        except:
            print(src)
            print(tgt)
            print(tgt_pred)
            raise
        # 负样本
        if src == tgt:
            # 预测也为负
            if tgt == tgt_pred:
                TN += 1
            # 预测为正
            else:
                FP += 1
        # 正样本
        else:
            # 预测也为正
            if tgt == tgt_pred:
                TP += 1
            # 预测为负
            else:
                FN += 1
        total_num += 1
    acc = (TP + TN) / total_num
    precision = TP / (TP + FP) if TP > 0 else 0.0
    recall = TP / (TP + FN) if TP > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
    print(
        f'Sentence Level: acc:{acc:.4f}, precision:{precision:.4f}, recall:{recall:.4f}, f1:{f1:.4f}')

def eval_inner_dataset():
    for fname in [
        'seo_judge_filter.jsonl',
        'sighan13_test.jsonl',
        'sighan14_test.jsonl',
        'sighan15_test.jsonl',
        'tongyin.jsonl',
        'zbsj.jsonl',
        'errors.jsonl',
        'de.jsonl',
        'corpus500.jsonl'
        ]:
        print(fname)
        test = [json.loads(x) for x in open('/nfs/users/xueyou/data/speller/pairs/testset/{}'.format(fname))]
        all_inputs = []
        all_golds = []
        all_preds = []
        for x in tqdm(test):
            text = x['src']
            tokens,text_len,_input_ids,_input_mask,_segment_ids = convert_single_example(text,tokenizer,max_seq_len)
            _pred_probs = sess.run(probs,feed_dict={input_ids:_input_ids,input_mask:_input_mask,segment_ids:_segment_ids})
            preds = np.argmax(_pred_probs[0], axis=-1)[1:text_len + 1]
            rpp = tokenizer.convert_ids_to_tokens(preds)
            tokens = tokens[:text_len]

            fp = []
            for p,t in zip(rpp,tokens):
                p = p.strip('##')
                t = t.strip('##')
                if p == '[UNK]' or t == '[UNK]':
                    if len(p) == 1:
                        fp.append(p)
                    else:
                        fp.append('X')
                elif len(p) != len(t):
                    fp.append(t)
                elif len(t) == 1 and tokenizer.basic_tokenizer._is_chinese_char(ord(t)):
                    if tokenizer.basic_tokenizer._is_chinese_char(ord(p)):
                        fp.append(p)
                    else:
                        fp.append(t)
                else:
                    fp.append(t)


            # pp = [pp[i] if len(tokens[i])==1 and tokenizer.basic_tokenizer._is_chinese_char(ord(tokens[i])) else tokens[i]  for i,x in enumerate(tokens)]
            # rpp = ['X' if p == '[UNK]' else p for i,p in enumerate(pp)]
            # pp = ''.join([x.strip('##') for x in rpp])
            pp = ''.join(fp)
            inputs= list(text)[:len(pp)]
            golds = list(x['tgt'].lower())[:len(pp)]

            try:
                assert len(pp) == len(inputs) == len(golds)
            except:
                print(rpp)
                print(fp)
                print(inputs)
                print(golds)
                print(tokens)
                raise



            all_preds.extend(pp)
            all_golds.extend(golds)
            all_inputs.extend(inputs)

        score_f((all_inputs, all_golds, all_preds),out_dir='/tmp')


def check():
    while True:
        text = input('输入：')

        tokens,text_len,_input_ids,_input_mask,_segment_ids = convert_single_example(text,tokenizer,max_seq_len)
        _pred_probs = sess.run(probs,feed_dict={input_ids:_input_ids,input_mask:_input_mask,segment_ids:_segment_ids})
        preds = np.argmax(_pred_probs[0], axis=-1)[1:text_len + 1]
        pp = tokenizer.convert_ids_to_tokens(preds)
        tokens = tokens[:text_len]
        pp = [pp[i] if pp[i]!='[UNK]' and (len(text[i])==1 and tokenizer.basic_tokenizer._is_chinese_char(ord(text[i]))) else text[i] for i,x in enumerate(pp)]
        pp = [t.strip('##') for t in pp]
        pp = ''.join(pp)
        text = text[:len(pp)]

        diff_pp = ''
        for a,b in dmp.diff_main(text,pp):
            if a == 0:
                diff_pp += b
            if a == -1:
                diff_pp += '\033[1;31m' + b + '\033[0m'
            if a == 1:
                diff_pp += '\033[1;32m' + b + '\033[0m'
        
        print(diff_pp)
        
if __name__ == '__main__':
    eval_corpus_500_score()



    

