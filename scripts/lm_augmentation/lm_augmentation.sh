#!/bin/sh
#SBATCH -o logs/%j.out
#SBATCH -p PV1003q
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --nodelist=node14
# module load cuda90/toolkit/9.0.176
# source /home/sbmaruf/.bashrc
# source ~/anaconda3/bin/activate pytorch_source
# source ~/anaconda3/bin/activate XZ-Adapt
# source ~/anaconda3/bin/activate pytorch_node08_source

export CUDA_VISIBLE_DEVICES='3'

project_name="lm_aug_pickle_small_test"
root_project_folder="./dumped"
mkdir -p $root_project_folder
project_folder=$root_project_folder"/"$project_name
mkdir -p $project_folder

# "successive_cross" "per_token"
for train_data_percentage in 1; do # 1 10 20 75
for dataset in  "data/en/en.train.iob2;utf-8;en" "data/es/es.train.iob2;utf-8;es" "data/de/de.train.iob2;latin-1;de" "data/nl/nl.train.iob2;utf-8;nl" "data/ar/ar.train.iob2;utf-8;ar" "data/fi/fi.train.iob2;utf-8;fi" ; do
    for aug_type in "successive_max"  ; do
        for aug_per in 5; do # 15 30 40 50
            for num_of_aug in 1; do
                for topk in 1; do
                        for only_ner_aug in 0; do
                            for seed in 1234; do

                                if [ $aug_type != "per_token" ]
                                then
                                    only_ner_aug=0
                                fi

                                if [ $aug_type = "per_token" ]
                                then
                                    aug_per=0
                                    num_of_aug=0
                                fi

                                name=${dataset//[\/.;]/_}

                                folder_address="dataset-"$name
                                folder_address=$folder_address"-aug_type-"$aug_type
                                folder_address=$folder_address"-aug_per-"$aug_per
                                folder_address=$folder_address"-num_of_aug-"$num_of_aug
                                folder_address=$folder_address"-only_ner_aug-"$only_ner_aug
                                folder_address=$folder_address"-topk-"$topk
                                folder_address=$folder_address"-train_data_percentage-"$train_data_percentage
                                folder_address=$folder_address"-seed-"$seed

                                OUT_ADDRESS=$project_folder"/"$folder_address
                                
                                if [ ! -d $OUT_ADDRESS ] 
                                then
                                    
                                    mkdir -p $OUT_ADDRESS

                                    python lm_augmentation.py \
                                    --dataset $dataset \
                                    --output_dir $OUT_ADDRESS \
                                    --aug_type $aug_type \
                                    --aug_per $aug_per \
                                    --topk $topk \
                                    --num_of_aug $num_of_aug \
                                    --only_ner_aug $only_ner_aug \
                                    --seed $seed \
                                    --train_data_percentage $train_data_percentage
                                fi
                            done
                        done
                    done
                done
            done
        done
    done
done
