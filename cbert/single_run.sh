# export BERT_DIR=/nfs/users/xueyou/data/bert_pretrain/cbert
export BERT_DIR=/nfs/users/xueyou/data/bert_pretrain/cbert
export DATA_DIR=/nfs/users/xueyou/data/speller/cbert/combine_more_shape.jsonl
export OUTPUT_DIR=/nfs/users/xueyou/data/speller/cbert/models/finetune_plome_cbert

mkdir -p ${OUTPUT_DIR}

python train.py \
    --input_file=${DATA_DIR} \
    --output_dir=${OUTPUT_DIR} \
    --init_checkpoint=${BERT_DIR}/bert_model.ckpt \
    --max_seq_len=128 \
    --learning_rate=3e-5 \
    --train_batch_size=64 \
    --num_train_steps=35000 \
    --num_warmup_steps=3500 \
    --tie_embedding=true \
    --finetune=true \
    --cbert=true \
    --amp=true \
    --hvd=false \
    --num_accumulation_steps=1

