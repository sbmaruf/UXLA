#!/bin/sh
#SBATCH -o logs/%j.out
#SBATCH -p DGXq
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --nodelist=node19

module load cuda90/toolkit/9.0.176

source /home/sbmaruf/.bashrc
source ~/anaconda3/bin/activate pytorch_source
# source ~/anaconda3/bin/activate XZ-Adapt
# source ~/anaconda3/bin/activate pytorch_node08_source

export CUDA_VISIBLE_DEVICES="1"

project_name="process_augmentation"
root_project_folder="./dumped"
mkdir -p $root_project_folder
project_folder=$root_project_folder"/"$project_name
mkdir -p $project_folder


for src_lang in "en"; do
    for max_steps in 5000 10000 15000 20000 25000 30000; do 
        # for warmup_steps in 100 200 300 500 1000 1500; do
        warmup_steps=$(($max_steps*10 / 100))
            for seed in 1234 1000 2000 3000 4000; do
                for penalty in 0; do
                    
                    sleep 1
                    
                    folder_address="penalty-"$penalty
                    folder_address=$folder_address"-max_steps-"$max_steps
                    folder_address=$folder_address"-warmup_steps-"$warmup_steps
                    folder_address=$folder_address"-src_lang-"$src_lang
                    folder_address=$folder_address"-seed-"$seed
                    OUT_ADDRESS=$project_folder"/"$folder_address
                    if [ ! -d $OUT_ADDRESS ] 
                    then
                    
                        mkdir -p $OUT_ADDRESS
                        touch "$OUT_ADDRESS/script.sh"
                        cat $0 > "$OUT_ADDRESS/script.sh"
                        
                        python -u main.py \
                        --train "dumped/lm_augmentation_1/dataset-data_en_en_train_iob2_utf-8_en/en.train.iob2.aug.ner_aug_0.topk_3;utf-8;en" \
                        --dev "data/$src_lang/$src_lang.testa.iob2;utf-8;$src_lang" \
                        --src_lang "$src_lang" \
                        --tgt_lang "en;es;de;nl;ar;fi" \
                        --model_type "bert" \
                        --model_name_or_path "dumped/base-multilingual_bert/penalty-2-max_steps-3000-warmup_steps-200-src_lang-en-seed-$seed/best_dev_model_data/en/en_testa_iob2_utf-8_en/pytorch_model.bin" \
                        --tokenizer_name "dumped/base-multilingual_bert/penalty-2-max_steps-3000-warmup_steps-200-src_lang-en-seed-$seed/best_dev_model_data/en/en_testa_iob2_utf-8_en/config.json" \
                        --output_dir $OUT_ADDRESS \
                        --process_augmentation \
                        --evaluate_during_training \
                        --per_gpu_train_batch_size 16 \
                        --per_gpu_eval_batch_size 256 \
                        --gradient_accumulation_steps 1 \
                        --logging_steps 50 \
                        --seed $seed \
                        --learning_rate 3e-5 \
                        --weight_decay 0.0 \
                        --max_steps $max_steps \
                        --warmup_steps $warmup_steps \
                        --penalty $penalty \
                        --merge_info "en;merge" "es;no_merge" "de;no_merge" "nl;no_merge" "ar;no_merge" "fi;no_merge"
 
                    fi
                done
            done
        # done
    done
done
# xlmroberta
# xlm-roberta-large
# bert
# bert-base-multilingual-cased