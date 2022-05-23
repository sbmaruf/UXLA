#!/bin/sh
#SBATCH -o logs/%j.out
#SBATCH -p NV100q
#SBATCH -n 1
#SBATCH --nodelist=node24
#SSSSSBATCH --gres=gpu:1

module load cuda10.1/toolkit/10.1.243

# source /home/sbmaruf/.bashrc
# source ~/anaconda3/bin/activate pytorch_source
source ~/anaconda3/bin/activate sbmaruf
# source ~/anaconda3/bin/activate pytorch_node08_source

export CUDA_VISIBLE_DEVICES="2"

src_lang="en"
tokenizer_name="xlm-roberta-large"
model_type="xlmroberta"
alpha_schedule="fixed"
model_name_or_path="xlm-roberta-large"

project_name="multi-mix-ccl-multi-model-ep-3-topk"
root_project_folder="dumped" 
mkdir -p $root_project_folder
project_folder=$root_project_folder"/"$project_name
mkdir -p $project_folder

for num_of_aug in 3; do
    # for aug_desc in  "0:tgt_self" "0:src;src_aug" "0:src_aug" "0:src;tgt_self" "0:tgt_aug" "0:src;tgt_aug" "0:tgt_self;tgt_aug"  "0:src;tgt_self;tgt_aug" "0:src_aug;tgt_aug" "0:src;src_aug;tgt_self;tgt_aug" ; do
    for aug_desc in "2:src;src_aug;tgt_self;tgt_aug"; do
    for tgt_lang in "es" "de" "ar" "fi" ; do
        for percent in 100; do
            for num_train_epochs in 1; do
                for max_steps in -1; do
                    for top_k in 80 100; do
                        for aug_per in 30; do   
                            for warmup_steps in 200; do
                                for data_distil_type in "top_k" ; do   #"gmm"
                                    for penalty in 0; do
                                        for lang_alpha in .7; do
                                            for dropout in .1; do
                                                for learning_rate in 5e-6; do
                                                    for batch_size in 4; do
                                                    for ga in 4; do
                                                    for seed in 1000; do         
                                                        sleep 1

                                                        dev_lang=$tgt_lang
                                                        aug_lang=$src_lang";"$tgt_lang
                                                        new_aug_lang=${aug_lang//[\/.;]/_}
                                                        new_aug_desc=${aug_desc//[\/.;]/_}
                                                        new_aug_desc=${new_aug_desc//[\/.:]/_}
                                                        logging_steps=50

                                                        folder_address="tgt-"$tgt_lang
                                                        folder_address="aug-"$new_aug_lang
                                                        folder_address=$folder_address"-per-"$percent
                                                        folder_address=$folder_address"-n_tr_ep-"$num_train_epochs
                                                        folder_address=$folder_address"-mx_steps-"$max_steps
                                                        folder_address=$folder_address"-top_k-"$top_k
                                                        folder_address=$folder_address"-aug_per-"$aug_per
                                                        folder_address=$folder_address"-n_aug-"$num_of_aug
                                                        folder_address=$folder_address"-w_st-"$warmup_steps
                                                        folder_address=$folder_address"-dis-"$data_distil_type
                                                        folder_address=$folder_address"-pen-"$penalty
                                                        folder_address=$folder_address"-la-"$lang_alpha
                                                        folder_address=$folder_address"-dp-"$dropout
                                                        folder_address=$folder_address"-lr_rate-"$learning_rate
                                                        folder_address=$folder_address"-aug_desc-"$new_aug_desc
                                                        folder_address=$folder_address"-batch-"$batch_size
                                                        folder_address=$folder_address"-ga-"$ga
                                                        folder_address=$folder_address"-sd-"$seed
                                                        
                                                        OUT_ADDRESS=$project_folder"/"$folder_address

                                                        encoding="utf-8"
                                                        if [ $tgt_lang == "de" ]; then
                                                            encoding="latin-1"
                                                        fi

                                                        if [ ! -d $OUT_ADDRESS ] 
                                                        then
                                                        
                                                            mkdir -p $OUT_ADDRESS
                                                            touch "$OUT_ADDRESS/script.sh"
                                                            cat $0 > "$OUT_ADDRESS/script.sh"
                                                            
                                                            python -u main.py \
                                                            --external_data "dumped/lm_aug_pickle/dataset-data_"$src_lang"_"$src_lang"_train_iob2_utf-8_$src_lang-aug_type-successive_max-aug_per-$aug_per-num_of_aug-$num_of_aug-only_ner_aug-0-topk-1-train_data_percentage-$percent-seed-1234/$src_lang.train.iob2.aug;pkl;$src_lang" \
                                                            "dumped/lm_aug_pickle/dataset-data_"$tgt_lang"_"$tgt_lang"_train_iob2_"$encoding"_$tgt_lang-aug_type-successive_max-aug_per-$aug_per-num_of_aug-$num_of_aug-only_ner_aug-0-topk-1-train_data_percentage-$percent-seed-1234/$tgt_lang.train.iob2.aug;pkl;$tgt_lang" \
                                                            --src_lang $src_lang \
                                                            --dev_lang $dev_lang \
                                                            --tgt_lang $tgt_lang \
                                                            --aug_lang $aug_lang \
                                                            --thetas "dumped/multi-mix-ccl-multi-model-ep-2-topk/aug-en_$tgt_lang-per-100-n_tr_ep-1-mx_steps--1-top_k-100-aug_per-30-n_aug-3-w_st-200-dis-top_k-pen-0-la-.7-dp-.1-lr_rate-5e-6-aug_desc-1_tgt_aug-batch-4-ga-4-sd-1000/best_dev_model.init.model.0/" \
                                                            "dumped/multi-mix-ccl-multi-model-ep-2-topk/aug-en_$tgt_lang-per-100-n_tr_ep-1-mx_steps--1-top_k-100-aug_per-30-n_aug-3-w_st-200-dis-top_k-pen-0-la-.7-dp-.1-lr_rate-5e-6-aug_desc-1_tgt_aug-batch-4-ga-4-sd-1000/best_dev_model.init.model.1/" \
                                                            "dumped/multi-mix-ccl-multi-model-ep-2-topk/aug-en_$tgt_lang-per-100-n_tr_ep-1-mx_steps--1-top_k-100-aug_per-30-n_aug-3-w_st-200-dis-top_k-pen-0-la-.7-dp-.1-lr_rate-5e-6-aug_desc-1_tgt_aug-batch-4-ga-4-sd-1000/best_dev_model.init.model.2/" \
                                                            --model_type $model_type \
                                                            --tokenizer_name $tokenizer_name \
                                                            --model_name_or_path $model_name_or_path \
                                                            --dropout $dropout \
                                                            --aug_mode "train" \
                                                            --output_dir $OUT_ADDRESS \
                                                            --n_mixture_component 2 \
                                                            --n_mixture_select 1 \
                                                            --posterior_threshold $posterior_threshold \
                                                            --per_gpu_train_batch_size $batch_size \
                                                            --per_gpu_eval_batch_size 32 \
                                                            --gradient_accumulation_steps $ga \
                                                            --seed $seed \
                                                            --learning_rate $learning_rate \
                                                            --penalty $penalty \
                                                            --do_semi_sup_training \
                                                            --semi_sup_type "multi-mix" \
                                                            --max_semi_sup_epoch 3 \
                                                            --semi_sup_start_epoch 2 \
                                                            --top_k $top_k \
                                                            --top_k_increment 20 \
                                                            --partial_train_in_semi_sup_epochs 0 \
                                                            --semi_sup_max_steps 0 \
                                                            --evaluate_during_training \
                                                            --data_distil_type $data_distil_type \
                                                            --logging_steps $logging_steps \
                                                            --train_data_percentage $percent \
                                                            --lang_alpha $lang_alpha \
                                                            --alpha_schedule $alpha_schedule \
                                                            --penalty $penalty \
                                                            --num_train_epochs $num_train_epochs \
                                                            --aug_desc $aug_desc
                                                        
                                                        knockknock slack --webhook-url "https://hooks.slack.com/services/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" --channel "cross-lingual-lm" echo $OUT_ADDRESS
                                                        fi
                                                    done
                                                    done
                                                    done
                                                done
                                            done
                                        done
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    done
    done
done


