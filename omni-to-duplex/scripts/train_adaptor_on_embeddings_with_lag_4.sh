CUDA_VISIBLE_DEVICES=6,7 \
uv run accelerate launch \
  --config_file configs/fsdp_config_2_gpu.yaml \
  train_adaptor_on_embeddings.py \
  --do_train True \
  --do_eval True \
  --num_train_epochs 1 \
  --max_steps 50000 \
  --fp16 False \
  --bf16 True \
  --tf32 False \
  --gradient_checkpointing False \
  --eval_strategy steps \
  --eval_steps 1000 \
  --save_strategy no \
  --logging_strategy steps \
  --logging_steps 10 \
  --report_to tensorboard \
  --learning_rate 0.001 \
  --warmup_ratio 0.001 \
  --lr_scheduler_type cosine \
  --per_device_train_batch_size 32 \
  --gradient_accumulation_steps 2 \
  --per_device_eval_batch_size 64 \
  --eval_accumulation_steps 16 \
  --disable_tqdm False \
  --data_path ../../../../mnt/efs/fs1/extracted_audio_features/ \
  --final_filename saves/adaptor_lag_4.pt \
  --max_eval_dataset_size 4096 \
  --lag_timesteps 4
