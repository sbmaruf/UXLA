#!/bin/sh
#SBATCH -o logs/%j.out
#SBATCH -p DGXq
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --nodelist=node19

# module load cuda90/toolkit/9.0.176
module load cuda10.1/toolkit/10.1.243

# source /home/sbmaruf/.bashrc
# source ~/anaconda3/bin/activate pytorch_source
source ~/anaconda3/bin/activate sbmaruf
# source ~/anaconda3/bin/activate pytorch_node08_source

export CUDA_VISIBLE_DEVICES="2"

project_name="base-multilingual-xlmr-supervised"
root_project_folder="./dumped"
mkdir -p $root_project_folder
project_folder=$root_project_folder"/"$project_name
mkdir -p $project_folder

for do_lower_case in 0; do
for warmup_steps in 200; do
for src_lang in "en"; do
    for max_steps in 3000; do 
            for seed in 1000 3000 4000; do
                for penalty in 0 2; do
                    
                    sleep 1
                    
                    folder_address="penalty-"$penalty
                    folder_address=$folder_address"-max_steps-"$max_steps
                    folder_address=$folder_address"-warmup_steps-"$warmup_steps
                    folder_address=$folder_address"-do_lower_case-"$do_lower_case
                    folder_address=$folder_address"-src_lang-"$src_lang
                    folder_address=$folder_address"-seed-"$seed
                    OUT_ADDRESS=$project_folder"/"$folder_address
                    if [ ! -d $OUT_ADDRESS ] 
                    then
                    
                        mkdir -p $OUT_ADDRESS
                        touch "$OUT_ADDRESS/script.sh"
                        cat $0 > "$OUT_ADDRESS/script.sh"
                        
                        python -u main.py \
                        --train "data/$src_lang/$src_lang.train.iob2;utf-8;$src_lang" \
                        --src_lang "$src_lang" \
                        --dev_lang "en;es;de;nl;ar;fi" \
                        --tgt_lang "en;es;de;nl;ar;fi" \
                        --model_type "xlmroberta" \
                        --model_name_or_path "xlm-roberta-large" \
                        --tokenizer_name "xlm-roberta-large" \
                        --output_dir $OUT_ADDRESS \
                        --do_train \
                        --evaluate_during_training \
                        --per_gpu_train_batch_size 4 \
                        --per_gpu_eval_batch_size 16 \
                        --gradient_accumulation_steps 5 \
                        --logging_steps 50 \
                        --seed $seed \
                        --learning_rate 3e-5 \
                        --weight_decay 0.0 \
                        --max_steps $max_steps \
                        --warmup_steps $warmup_steps \
                        --penalty $penalty \
			            --do_lower_case $do_lower_case
                        
                        knockknock slack --webhook-url "https://hooks.slack.com/services/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" --channel "cross-lingual-lm" echo $OUT_ADDRESS"-"$CUDA_VISIBLE_DEVICES
                    fi
                done
            done
        done
    done
done
done
# xlmroberta
# xlm-roberta-large
# bert
# bert-base-multilingual-cased
