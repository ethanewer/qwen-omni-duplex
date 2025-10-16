CUDA_VISIBLE_DEVICES=4,5 \
uv run train_duplex_model.py \
  --do_train True \
  --do_eval False \
  --num_train_epochs 1 \
  --max_steps 10 \
  --fp16 False \
  --bf16 True \
  --tf32 False \
  --gradient_checkpointing False \
  --logging_strategy steps \
  --logging_steps 1 \
  --learning_rate 0.001 \
  --warmup_ratio 0.0002 \
  --lr_scheduler_type cosine \
  --per_device_train_batch_size 16 \
  --gradient_accumulation_steps 1 \
  --disable_tqdm False \
  --output_dir saves/test
