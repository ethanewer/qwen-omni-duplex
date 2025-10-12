CUDA_VISIBLE_DEVICES=4,5,6,7 ACCELERATE_SPLIT_BATCHES=true \
uv run accelerate launch \
  --config_file configs/fsdp_config.yaml \
  train_adaptor.py \
  --do_train True \
  --do_eval True \
  --num_train_epochs 1 \
  --max_steps 10000 \
  --fp16 False \
  --bf16 True \
  --tf32 False \
  --gradient_checkpointing False \
  --eval_strategy steps \
  --eval_steps 500 \
  --save_strategy steps \
  --save_steps 500 \
  --save_total_limit 3 \
  --logging_strategy steps \
  --logging_steps 10 \
  --report_to tensorboard \
  --learning_rate 0.001 \
  --warmup_ratio 0.01 \
  --lr_scheduler_type cosine \
  --per_device_train_batch_size 16 \
  --gradient_accumulation_steps 1 \
  --per_device_eval_batch_size 64 \
  --eval_accumulation_steps 16 \
  --disable_tqdm False \
  --data_path ../../../../mnt/efs/fs1/extracted_audio_features/ \
  --final_filename final_adaptor.pt \
  --max_eval_dataset_size 4096 \
  $@
