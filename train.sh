
# Finetuning baseline RoBERTa

python3 src/train.py --data_path "./data" \
            --model_name "ixa-ehu/roberta-eus-euscrawl-large-cased" \
            --fp16 True \
            --num_train_epochs 10 \
            --weight_decay 5e-5 \
            --learning_rate 5e-5 \
            --per_device_train_batch_size 32 \
            --per_device_eval_batch_size 4 \
            --gradient_accumulation_steps 1 \
            --eval_accumulation_steps 10 \
            --seed 42 \
            --auto_find_batch_size False \
            --optim "adamw_torch" \
            --lr_scheduler_type "cosine"\
            --save_total_limit 2 \
            --evaluation_strategy "epoch" \
            --save_strategy "epoch" \
            --logging_strategy "steps" \
            --logging_steps 25 \
            --metric_for_best_model "f1" \
            --load_best_model_at_end True \
            --output_dir "./models/" \
            --run_name "finetuning-roberta-euscrawl" \
            --do_train True \
            --do_eval True \


# Finetuning using supervised contrastive learning

python3 src/train.py --data_path "./data/" \
            --model_name "ixa-ehu/roberta-eus-euscrawl-large-cased" \
            --contrastive True \
            --contrastive_lam 0.2 \
            --contrastive_temp 0.2 \
            --fp16 True \
            --num_train_epochs 10 \
            --weight_decay 5e-5 \
            --learning_rate 5e-5 \
            --per_device_train_batch_size 32 \
            --per_device_eval_batch_size 4 \
            --gradient_accumulation_steps 1 \
            --eval_accumulation_steps 10 \
            --seed 42 \
            --auto_find_batch_size False \
            --optim "adamw_torch" \
            --lr_scheduler_type "cosine"\
            --save_total_limit 2 \
            --evaluation_strategy "epoch" \
            --save_strategy "epoch" \
            --logging_strategy "steps" \
            --logging_steps 25 \
            --metric_for_best_model "f1" \
            --load_best_model_at_end True \
            --output_dir "./models/" \
            --run_name "finetuning-roberta-euscrawl-scl" \
            --do_train True \
            --do_eval True \

# Finetuning using supervised contrastive learning and different pooling strategies

for strategy in "min" "max" "mean" "sum" "attention"
do

    python3 src/train.py --data_path "./data/" \
                --model_name "ixa-ehu/roberta-eus-euscrawl-large-cased" \
                --pooling_strategy "${strategy}" \
                --contrastive True \
                --contrastive_lam 0.2 \
                --contrastive_temp 0.2 \
                --fp16 True \
                --num_train_epochs 10 \
                --weight_decay 5e-5 \
                --learning_rate 5e-5 \
                --per_device_train_batch_size 32 \
                --per_device_eval_batch_size 4 \
                --gradient_accumulation_steps 1 \
                --eval_accumulation_steps 10 \
                --seed 42 \
                --auto_find_batch_size False \
                --optim "adamw_torch" \
                --lr_scheduler_type "cosine"\
                --save_total_limit 2 \
                --evaluation_strategy "epoch" \
                --save_strategy "epoch" \
                --logging_strategy "steps" \
                --logging_steps 25 \
                --metric_for_best_model "f1" \
                --load_best_model_at_end True \
                --output_dir "./models/" \
                --run_name "finetuning-roberta-euscrawl-scl-${strategy}_pooling" \
                --do_train True \
                --do_eval True \

done