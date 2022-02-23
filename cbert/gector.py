import tensorflow as tf
import modeling

class GECToRModel:
    def __init__(self, bert_config,num_labels_classes,num_detect_classes=2):
        self.bert_config = bert_config
        self.num_labels_classes = num_labels_classes
        self.num_detect_classes = num_detect_classes

    def create_model(self, input_ids, input_mask, segment_ids,labels=None,d_tags=None,is_training=True):
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

        if is_training:
            input_tensor = modeling.dropout(input_tensor, self.bert_config.hidden_dropout_prob)

        with tf.variable_scope('gector', reuse=tf.AUTO_REUSE):
            logits_labels = tf.layers.dense(input_tensor,self.num_labels_classes,name='correct')
            logits_d = tf.layers.dense(input_tensor,self.num_detect_classes,name='detect')

            class_probabilities_labels = tf.nn.softmax(logits_labels, dim=-1)
            class_probabilities_d = tf.nn.softmax(logits_d, dim=-1)
            
            loss = None
            labels_accuracy = None
            d_accuracy = None
            if labels is not None and d_tags is not None:
                loss_labels = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=logits_labels)
                label_weights = tf.cast(input_mask,dtype=loss_labels.dtype)
                numerator = tf.reduce_sum(label_weights * loss_labels)
                denominator = tf.reduce_sum(label_weights) + 1e-5
                loss_labels = numerator / denominator

                loss_d = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=d_tags,logits=logits_d)
                numerator = tf.reduce_sum(label_weights * loss_d)
                loss_d = numerator / denominator
                loss = loss_labels + loss_d

                weights = tf.cast(labels>0,dtype=loss_labels.dtype)
                labels_accuracy = tf.metrics.accuracy(
                    labels, tf.argmax(logits_labels,axis=-1),weights=weights)
                d_accuracy = tf.metrics.accuracy(
                    d_tags, tf.argmax(logits_d,axis=-1),weights=weights)
            
            return class_probabilities_labels,class_probabilities_d, loss, labels_accuracy,d_accuracy









            

            



