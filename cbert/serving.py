from flask import Flask, render_template, request
from flask_cors import CORS
import tensorflow as tf 
import cbert
import json
import tokenization
import modeling
import numpy as np
from tqdm import tqdm
import diff_match_patch as dmp_module
from pyltp import SentenceSplitter

dmp = dmp_module.diff_match_patch()

app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})



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
bert_config = modeling.BertConfig.from_json_file('assets/bert_config.json')

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

def check(text):
    texts = text.split('\n')
    diffs_pp = []
    for part in texts:
        if len(part) < 2:
            diffs_pp.append(part)
            continue
        correct = []
        for text in SentenceSplitter.split(part):
            text = text.replace(' ','')
            tokens,text_len,_input_ids,_input_mask,_segment_ids = convert_single_example(text,tokenizer,max_seq_len)
            _pred_probs = sess.run(probs,feed_dict={input_ids:_input_ids,input_mask:_input_mask,segment_ids:_segment_ids})
            # TODO: 还可以做的优化包括
            # - 加入混淆集预测的限制
            # - 语言模型过滤
            # - 实体过滤
            # - decode改进，主要解决连续字错误的情况，比如topk结果后面接个beam search
            preds = np.argmax(_pred_probs[0], axis=-1)[1:text_len + 1]
            pp = tokenizer.convert_ids_to_tokens(preds)
            # 非中文字符不纠错
            pp = [pp[i] if pp[i]!='[UNK]' and (len(text[i])==1 and tokenizer.basic_tokenizer._is_chinese_char(ord(text[i]))) else text[i] for i,x in enumerate(pp)]
            pp = [t.strip('##') for t in pp]
            pp = ''.join(pp)

            pp = ''.join(pp)
            pp = pp + text[len(pp):]

            diff_pp = ''
            for a,b in dmp.diff_main(text,pp):
                if a == 0:
                    diff_pp += b
                if a == -1:
                    diff_pp += f'<font color="#CB1A50"><b>' + b + '</b></font>'
                if a == 1:
                    diff_pp += f'<font color="#6CCC82"><b>' + b + '</b></font>'
            correct.append(diff_pp)
        diffs_pp.append(''.join(correct))
    return '<br>'.join(diffs_pp)


@app.route('/predict',methods=['POST'])
def predict():
  data = request.get_json(silent=True)
  text = check(data['text'])
  return json.dumps({"text":text})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8505, debug=False)
