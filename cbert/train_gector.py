import tensorflow as tf 
from data import GECToRDataset
import gector
import optimization
import tokenization
import modeling
import time
import numpy as np
import os
import datetime

# 这里为了避免打印重复的日志信息
tf.get_logger().propagate = False


flags = tf.flags

FLAGS = flags.FLAGS


flags.DEFINE_string(
    "bert_config_file", 'assets/bert_config.json',
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string(
    "input_file", None,
    "Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

flags.DEFINE_string(
    "vocab_file", 'assets/vocab.txt',
    "vocab file")

## Other parameters
flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_integer(
    "max_seq_len", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded. Must match data generation.")


flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 32, "Total batch size for eval.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_integer("num_train_steps", 100000, "Number of training steps.")

flags.DEFINE_integer("num_warmup_steps", 10000, "Number of warmup steps.")

flags.DEFINE_integer("save_checkpoints_steps", 5000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("max_eval_steps", 100, "Maximum number of eval steps.")
flags.DEFINE_bool("amp", False, "Whether to enable AMP ops. When false, uses TF32 on A100 and FP32 on V100 GPUS.")
flags.DEFINE_bool("hvd", False, "Whether to use hvd")
flags.DEFINE_bool("finetune", False, "Whether to finetune")
flags.DEFINE_integer("num_accumulation_steps", 1, "how many steps to do gradients accumulation")

def main(_):
    master = True
    if FLAGS.hvd:
        import horovod.tensorflow as hvd
        hvd.init()
        
        if hvd.rank() != 0:
            master = False
    else:
        hvd = None
    tokenizer = tokenization.FullTokenizer(FLAGS.vocab_file)
    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
    data_processor = GECToRDataset(FLAGS.input_file, FLAGS.max_seq_len, tokenizer)
    if FLAGS.finetune:
        iterator = data_processor.make_finetune_dataset(FLAGS.train_batch_size).make_one_shot_iterator()
    else:
        iterator = data_processor.make_train_dataset(FLAGS.train_batch_size).make_one_shot_iterator()
    inputs = iterator.get_next()


    tf_config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    tf_config.gpu_options.allow_growth = True
    if hvd:
        tf_config.gpu_options.visible_device_list = str(hvd.local_rank())

    model = gector.GECToRModel(bert_config,num_labels_classes=len(data_processor.label2id))
    class_probabilities_labels,class_probabilities_d, loss, labels_accuracy,d_accuracy = model.create_model(
        input_ids=inputs['input_ids'],
        input_mask=inputs['input_mask'],
        segment_ids=inputs['segment_ids'],
        labels=inputs['label_ids'],
        d_tags=inputs['dtags'],
        is_training=True
    )
    train_op, learning_rate = optimization.create_optimizer(
        loss=loss,
        init_lr=FLAGS.learning_rate,
        num_train_steps=FLAGS.num_train_steps,
        num_warmup_steps=FLAGS.num_warmup_steps,
        amp=FLAGS.amp,
        hvd=hvd,
        accumulation_step=FLAGS.num_accumulation_steps)


    with tf.Session(config=tf_config) as sess:
        if FLAGS.init_checkpoint:
            tvars = tf.trainable_variables()
            (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars,FLAGS.init_checkpoint)

            tf.train.init_from_checkpoint(FLAGS.init_checkpoint, assignment_map)

            tf.logging.info("**** Trainable Variables ****")
            for var in tvars:
                init_string = ""
                if var.name in initialized_variable_names:
                    init_string = ", *INIT_FROM_CKPT*"
                tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                                init_string)

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        if hvd:
            sess.run(hvd.broadcast_global_variables(0))

        saver = tf.train.Saver(var_list = tf.trainable_variables(),max_to_keep=5)

        loss_values = []
        start_time = time.time()
        model_path = os.path.join(FLAGS.output_dir, 'gector') 
        for step in range(FLAGS.num_train_steps):
            _,train_loss,train_acc,d_acc = sess.run([train_op,loss,labels_accuracy[1],d_accuracy[1]])
            loss_values.append(train_loss)
            if master and step % 50 == 0:
                duration = (time.time() - start_time) / 50
                eta = (FLAGS.num_train_steps - step) * duration
                examples_per_sec = 1 / float(duration) * (FLAGS.train_batch_size * FLAGS.num_accumulation_steps)
                tf.logging.info(f"# Step {step}, ETA {datetime.timedelta(seconds=eta)}, detect acc {d_acc*100:.4f}, train acc {train_acc*100:.4f},  train loss {np.mean(loss_values):.4f},{np.mean(loss_values[-1000:]):.4f},{np.mean(loss_values[-100:]):.4f} ({examples_per_sec:.4f} examples/sec; {duration:.4f} sec/batch)")
                start_time = time.time()
            loss_values = loss_values[-5000:]

            if master and step % 10000 == 0:
                saver.save(sess,model_path,step)

        if master:
            saver.save(sess,model_path)




if __name__ == "__main__":
  flags.mark_flag_as_required("input_file")
  flags.mark_flag_as_required("bert_config_file")
  flags.mark_flag_as_required("output_dir")
  tf.app.run()