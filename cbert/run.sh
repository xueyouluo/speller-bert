export BERT_DIR=/data/xueyou/data/bert_pretrain/chinese_roberta_wwm_ext_L-12_H-768_A-12
export DATA_DIR=/nfs/users/xueyou/data/speller/cbert/data/combine_part_*
export OUTPUT_DIR=/nfs/users/xueyou/data/speller/cbert/models/baseline_update_punctuation_more_keep

mkdir -p ${OUTPUT_DIR}

horovodrun -np 4 -H localhost:4 python train.py \
    --input_file=${DATA_DIR} \
    --output_dir=${OUTPUT_DIR} \
    --init_checkpoint=${BERT_DIR}/bert_model.ckpt \
    --max_seq_len=128 \
    --learning_rate=2e-5 \
    --train_batch_size=32 \
    --num_train_steps=500000 \
    --num_warmup_steps=50000 \
    --tie_embedding=True \
    --amp=true \
    --hvd=true \
    --num_accumulation_steps=1

