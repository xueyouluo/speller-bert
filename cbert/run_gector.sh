export BERT_DIR=/data/xueyou/data/bert_pretrain/chinese_roberta_wwm_ext_L-12_H-768_A-12
export DATA_DIR=/data/xueyou/data/speller/gector/finetune/train_part_*
export OUTPUT_DIR=/data/xueyou/data/speller/gector/finetune/model_fix_data_more

mkdir -p ${OUTPUT_DIR}

horovodrun -np 4 -H localhost:4 python train_gector.py \
    --input_file=${DATA_DIR} \
    --output_dir=${OUTPUT_DIR} \
    --init_checkpoint=/data/xueyou/data/speller/gector/baseline/gector \
    --max_seq_len=128 \
    --learning_rate=5e-5 \
    --train_batch_size=32 \
    --num_train_steps=100000 \
    --num_warmup_steps=10000 \
    --amp=true \
    --hvd=true \
    --num_accumulation_steps=1

