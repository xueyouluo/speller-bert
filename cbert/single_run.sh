# export BERT_DIR=/nfs/users/xueyou/data/bert_pretrain/cbert
export BERT_DIR=/nfs/users/xueyou/data/bert_pretrain/cbert
# export BERT_DIR=/nfs/users/xueyou/data/speller/cbert/models/baseline_update_punctuation
export DATA_DIR=/nfs/users/xueyou/data/speller/cbert/combine_v2.jsonl
export OUTPUT_DIR=/data/xueyou/data/speller/cbert/finetune_plome_cbert_v2

mkdir -p ${OUTPUT_DIR}

python train.py \
    --input_file=${DATA_DIR} \
    --output_dir=${OUTPUT_DIR} \
    --init_checkpoint=${BERT_DIR}/bert_model.ckpt \
    --max_seq_len=128 \
    --learning_rate=3e-5 \
    --train_batch_size=64 \
    --num_train_steps=80000 \
    --num_warmup_steps=8000 \
    --tie_embedding=true \
    --finetune=true \
    --cbert=true \
    --amp=true \
    --hvd=false \
    --num_accumulation_steps=1

