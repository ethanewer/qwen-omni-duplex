CUDA_VISIBLE_DEVICES=4,5,6,7 \
uv run accelerate launch \
  --config_file configs/fsdp_config.yaml \
  train_adaptor.py \
  --do_train True \
  --do_eval False \
  --num_train_epochs 1 \
  --fp16 False \
  --bf16 True \
  --tf32 False \
  --gradient_checkpointing False \
  --save_strategy steps \
  --save_steps 2500 \
  --save_total_limit 3 \
  --logging_strategy steps \
  --logging_steps 100 \
  --report_to tensorboard \
  --learning_rate 1e-5 \
  --warmup_ratio 0.1 \
  --lr_scheduler_type cosine \
  --per_device_train_batch_size 128 \
  --gradient_accumulation_steps 1 \
  --disable_tqdm False \
  $@
