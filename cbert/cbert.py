import tensorflow as tf 
import modeling

class CBert:
    def __init__(self, bert_config, tie_embedding=False, add_transform=True, cbert=False):
        self.bert_config = bert_config
        self.tie_embedding = tie_embedding
        self.add_transform = add_transform
        self.cbert=cbert

    def create_model(self, input_ids, input_mask, segment_ids,label_weights=None,labels=None,is_training=True):

        model = modeling.BertModel(
            config = self.bert_config,
            is_training = is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=segment_ids
        )

        input_tensor = model.get_all_encoder_layers()[-1]
        shape_list = modeling.get_shape_list(input_tensor,expected_rank=3)
        hidden_size = self.bert_config.hidden_size

        # 这里需要兼容PLOME的cbert参数，以及传统的BERT的参数
        # cbert的scope为cbert
        # PLOME的cbert scope为loss
        # bert的scope为 cls/predictions，如果需要加载预训练参数需要注意修改
        with tf.variable_scope('cbert' if not self.cbert else 'loss',reuse=tf.AUTO_REUSE):
            if not self.cbert and self.add_transform:
                with tf.variable_scope("transform"):
                    input_tensor = tf.layers.dense(
                        input_tensor,
                        units=hidden_size,
                        activation=modeling.get_activation(self.bert_config.hidden_act),
                        kernel_initializer=modeling.create_initializer(
                            self.bert_config.initializer_range))
                    input_tensor = modeling.layer_norm(input_tensor)

            if not self.cbert and self.tie_embedding:
                output_weights = model.get_embedding_table()
            else:
                output_weights = tf.get_variable(
                    'output_weights', [self.bert_config.vocab_size, hidden_size],
                    initializer=tf.truncated_normal_initializer(stddev=0.02)
                )

            output_bias = tf.get_variable(
                "output_bias", [self.bert_config.vocab_size], initializer=tf.zeros_initializer())

            input_tensor = tf.reshape(input_tensor,[-1,hidden_size])
            logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
            flat_logits = tf.nn.bias_add(logits, output_bias)
            logits = tf.reshape(flat_logits,shape_list[:2] + [self.bert_config.vocab_size])
            probs = tf.nn.softmax(logits, axis=-1)

            loss = None
            accuracy = None
            if labels is not None:
                
                label_ids = tf.reshape(labels, [-1])
                one_hot_labels = tf.one_hot(
                    label_ids, depth=self.bert_config.vocab_size, dtype=tf.float32)
                log_probs = tf.nn.log_softmax(flat_logits,axis=-1)
                per_example_loss = -tf.reduce_sum(log_probs * one_hot_labels, axis=[-1])
                if label_weights is None:
                    label_weights = tf.ones_like(per_example_loss)
                label_weights = tf.cast(label_weights,dtype=per_example_loss.dtype)
                flat_label_weights = tf.reshape(label_weights,[-1])
                numerator = tf.reduce_sum(flat_label_weights * per_example_loss)
                denominator = tf.reduce_sum(flat_label_weights) + 1e-5
                loss = numerator / denominator

                accuracy = tf.metrics.accuracy(
                    labels, tf.argmax(logits,axis=-1),weights=label_weights)

                # tf.summary.scalar('accuracy', accuracy[0]*100)
            
            return logits, probs, loss, accuracy



            
            




