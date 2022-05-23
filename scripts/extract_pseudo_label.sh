#!/bin/sh
#SBATCH -o logs/%j.out
#SBATCH -p DGXq
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --nodelist=node20

module load cuda90/toolkit/9.0.176

# source /home/sbmaruf/.bashrc
# source ~/anaconda3/bin/activate pytorch_source
# source ~/anaconda3/bin/activate XZ-Adapt
source ~/anaconda3/bin/activate pytorch_node08_source

export CUDA_VISIBLE_DEVICES='4'

lang="es;nl;de;ar;fi"
min_number_of_word=5
max_number_of_word=60

project_name="extract_pseudo_label"
root_project_folder="./dumped"
mkdir -p $root_project_folder
project_folder=$root_project_folder"/"$project_name
mkdir -p $project_folder

lambda=.1
k_size=1
noise_threshold=0
n_mixture_component=5
n_mixture_select=1
posterior_threshold=.1
logit_bank_type='non-clustered'

for seed in 1234 1000 2000 3000 4000; do
    for penalty in 0 2; do
        for lambda in .1 .9; do
            for k_size in 1 3; do
                for posterior_threshold in .1 .2 .5 .7 .8 .85 .9; do
                    
                    folder_address="seed-"$seed
                    folder_address=$folder_address"-penalty-"$penalty
                    folder_address=$folder_address"-lambda-"$lambda
                    folder_address=$folder_address"-k_size-"$k_size
                    folder_address=$folder_address"-posterior_threshold-"$posterior_threshold

                    OUT_ADDRESS=$project_folder"/"$folder_address

                    if [ ! -d $OUT_ADDRESS ] 
                    then
                        mkdir -p $OUT_ADDRESS
                        touch "$OUT_ADDRESS/script.sh"
                        cat $0 > "$OUT_ADDRESS/script.sh"
                        
                        python -u main.py \
                        --src_lang "en" \
                        --dev_lang "en" \
                        --tgt_lang "$lang" \
                        --model_type 'xlm-roberta-large' \
                        --model_name_or_path "path to the warmup model" \
                        --config_name "config path to the warmup model" \
                        --output_dir $OUT_ADDRESS \
                        --noise_threshold $noise_threshold \
                        --n_mixture_component $n_mixture_component \
                        --n_mixture_select $n_mixture_select \
                        --posterior_threshold $posterior_threshold \
                        --covariance_type "full" \
                        --min_length_restriction -1 \
                        --max_length_restriction 10000 \
                        --aug_mode "train" \
                        --export_pseudo_data \
                        --lazy_inference \
                        --evaluate_during_training \
                        --per_gpu_train_batch_size 4 \
                        --per_gpu_eval_batch_size 32 \
                        --gradient_accumulation_steps 2 \
                        --logging_steps 50 \
                        --seed $seed \
                        --logit_dict_cache_address $OUT_ADDRESS/"train_loss_dict" \
                        --lam $lambda \
                        --k_size $k_size \
                        --logit_bank_type $logit_bank_type \
                        --penalty 0
                    fi
                done
            done
        done
    done
done
