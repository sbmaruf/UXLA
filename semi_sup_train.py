import os
import copy
import json
import torch
import pickle
import random
import hashlib
import subprocess

import numpy as np
from tensorboardX import SummaryWriter
from collections import OrderedDict
from train import export_logit, evaluate, training_loop
from unsup_utils import select_samples_with_GMM, select_data_from_logit, unsupervised_sample_selection
from utils_ner import select_and_write_data, write_conll_data, select_and_write_source_data, read_from_path, read_examples_from_file
from model import load_model, save_model_checkpoint
from load_examples import  load_and_cache_examples
from lm_augmentation import augment_data

def create_aug_data(args, external_dataset_address, langs, model, tokenizer, labels, mode, top_k, pad_token_label_id, logger):
    
    logger.info("[::] Lang = {}, Mode = {} data prediction.".format(langs, mode))    
    logger.info("--"*10) 
    if external_dataset_address is not None:
        for dt in external_dataset_address:
            logger.info("Dataset info {}, dataset length {}".format(
                dt, 
                len(
                    read_from_path(
                        dt.split(";")[0], 
                        encoding=dt.split(";")[1]
                    )
                )
            ))
    temp_train_data_percentage = args.train_data_percentage 
    args.train_data_percentage = 100
    loss_dict, logit_dict, _, _, _= export_logit(
                    args, 
                    model, tokenizer, labels, 
                    pad_token_label_id, mode=mode, external_data=external_dataset_address,
                    prefix="", langs = langs, logger=logger, debug=0
                )
    args.train_data_percentage = temp_train_data_percentage
    
    external_files = []
    for dict_key in loss_dict.keys():
        
        logger.info("{} Distillation".format(dict_key))
        logger.info("--"*10)
        
        temp_top_k = args.top_k
        args.top_k = args.top_k if top_k is None else top_k
        indices, gmm_model = data_distillation(
            args,
            dict_key,
            pseudo_loss_dict=loss_dict,
            logit_dict=logit_dict,
            mode="train",
            logger=logger,
            debug=0
        )
        args.top_k = temp_top_k
        
        total_indexes = indices
        address, indices = select_and_write_data(
                    dict_key,
                    args.output_dir,
                    total_indexes, 
                    loss_dict, 
                    labels, 
                    logger=logger,
                    postfix = str(random.randint(0,10000000))
                )
        external_files.append(address)
        
    return external_files

def create_file_name(file_name1, file_name2, logger):
    new_name = file_name1 + "__" + file_name2
    if len(new_name) > 50:
        temp = new_name
        logger.info("File name size exceeds 50 length, using hash instead")
        new_name = hashlib.sha256(new_name.encode('utf-8')).hexdigest()
        logger.info("[sha256_hashing]:: {} : {} ".format(temp, new_name))
    return new_name

        

def merge_two_dataset(file_info_1, file_info_2, output_dir, logger):
    address1, encoding1, lang1 = file_info_1.split(";")[0], file_info_1.split(";")[1], file_info_1.split(";")[2]
    address2, encoding2, lang2 = file_info_2.split(";")[0], file_info_2.split(";")[1], file_info_2.split(";")[2]
    if encoding1 != encoding2:
        encoding1 = "latin-1"
        encoding2 = "latin-1"
    assert lang1 == lang2
    file_name1 = os.path.split(address1)[-1]
    file_name2 = os.path.split(address2)[-1]
    new_name = create_file_name(file_name1, file_name2, logger)
    full_path = os.path.join(output_dir, new_name)
    assert os.path.exists(full_path) == False
    subprocess.check_output("touch {}".format(full_path), shell=True)
    subprocess.check_output("cat {}  >> {}".format(address1, full_path), shell=True)
    subprocess.check_output("echo \"\n\"  >> {}".format(full_path), shell=True)
    subprocess.check_output("cat {}  >> {}".format(address2, full_path), shell=True)
    new_file_info = full_path+";"+encoding1+";"+lang1
    return new_file_info


def partial_single_self_training(
        args, MODEL_CLASSES,
        model, tokenizer, 
        labels, pad_token_label_id, num_labels,
        is_GMM_selection=0,
        logger=None,
    ):
    best_dev_scores = None
    test_scores_in_best_src_dev = None
    logger.info("Dev Evaluation")
    logger.info("--"*10)
    dev_scores = evaluate(
                    args, 
                    model, tokenizer, labels, 
                    pad_token_label_id, "dev", 
                    prefix="", langs = args.tgt_lang,
                    logger=logger
                )
    logger.info("Test Evaluation")
    logger.info("--"*10)
    test_scores = evaluate(
                        args, 
                        model, tokenizer, labels, 
                        pad_token_label_id, "test", 
                        prefix="", langs = args.tgt_lang,
                        logger=logger
                    )
    
    backup_overwrite_cache = args.overwrite_cache
    args.overwrite_cache = True
    
    logger.info("Source Train Data Loading")  
    logger.info("--"*10)  
    train_dataset = OrderedDict()
    # source dataset load
    train_dataset, guids = load_and_cache_examples(
                        args, tokenizer, labels, pad_token_label_id, 
                        mode="train", langs=args.src_lang, logger=logger
                    )
    datasets = []
    
    if "self_src" in args.aug_desc.split(";"):
        if args.train_data_percentage == 100:
            logger.warning("[***] :: train_data_percentage is 100 and we are augmenting source dataset...!!!!")
        logger.info("Pseudo Source Train Dataset Creation Process Started ...")
        logger.info("=="*10)
        src_train_dataset = create_aug_data(
            args, external_dataset_address=None, 
            langs=args.src_lang, 
            model=model, tokenizer=tokenizer, labels=labels, 
            mode="train", top_k=args.top_k, 
            pad_token_label_id=pad_token_label_id, 
            logger=logger
        )
        datasets += src_train_dataset
    
    if "self_tgt" in args.aug_desc.split(";"):
        
        logger.info("Pseudo Target Train Dataset Creation Process Started ...")
        logger.info("=="*10)
        tgt_train_datasets = create_aug_data(
            args, external_dataset_address=None, 
            langs=args.tgt_lang, 
            model=model, tokenizer=tokenizer, labels=labels, 
            mode="train", top_k=100, 
            pad_token_label_id=pad_token_label_id, 
            logger=logger
        )
        datasets += tgt_train_datasets
    
    if "src_aug" in args.aug_desc.split(";"):
        
        logger.info("Pseudo Augmented Source Dataset Creation Process Started ...")
        logger.info("=="*10)
        
        src_aug_datasets = create_aug_data(
            args, external_dataset_address=args.external_data, 
            langs=args.src_lang, 
            model=model, tokenizer=tokenizer, labels=labels, 
            mode="aug", top_k=args.top_k, 
            pad_token_label_id=pad_token_label_id, 
            logger=logger
        )
        datasets += src_aug_datasets
        
    if "tgt_aug" in args.aug_desc.split(";"):
        
        logger.info("Pseudo Augmented Target Dataset Creation Process Started ...")
        logger.info("=="*10)
        
        tgt_aug_datasets = create_aug_data(
            args, external_dataset_address=args.external_data, 
            langs=args.tgt_lang, 
            model=model, tokenizer=tokenizer, labels=labels, 
            mode="aug", top_k=args.top_k, 
            pad_token_label_id=pad_token_label_id, 
            logger=logger
        )
        datasets += tgt_aug_datasets
    
    total_sent_len = 0
    if args.merge_datasets:
        for idx, dataset in enumerate(datasets):
            total_sent_len = len(read_from_path(dataset.split(";")[0], encoding=dataset.split(";")[1]))
            if idx == 0:
                continue
            logger.info("Merging Dataset {} ({}) and {} ({})".format(
                    datasets[idx-1], len(read_from_path(datasets[idx-1].split(";")[0], encoding=datasets[idx-1].split(";")[1])),
                    datasets[idx], len(read_from_path(datasets[idx].split(";")[0], encoding=datasets[idx].split(";")[1]))))
            
            datasets[idx] = merge_two_dataset(datasets[idx-1], datasets[idx], args.output_dir, logger)
            logger.info("New Dataset address {} ({})".format(
                    datasets[idx], len(read_from_path(datasets[idx].split(";")[0], encoding=datasets[idx].split(";")[1]))))
        datasets = [datasets[-1]]
        try:
            found_sent_len = len(read_from_path(datasets[0].split(";")[0], encoding=datasets[0].split(";")[1]))
            assert total_sent_len == found_sent_len
        except:
            logger.info("Total Number of sentence {}, found {}".format(total_sent_len, found_sent_len))
    
    
    logger.info("[::] Creating Tensor of Training Datasets...")
    
    temp_train_data_percentage = args.train_data_percentage
    args.train_data_percentage = 100
    for dataset in datasets:
        logger.info("Dataset Name : {}".format(dataset))
        tensor_dataset, _ = load_and_cache_examples(
                                args, tokenizer, labels, pad_token_label_id, external_data=[dataset],
                                mode="aug", langs=dataset.split(";")[-1], logger=logger
                            )
        for k, v in tensor_dataset.items():
            assert k not in train_dataset
            train_dataset[k] = v
    args.train_data_percentage = temp_train_data_percentage

    args.overwrite_cache = backup_overwrite_cache
    tot_sample = 0
    for k, v in train_dataset.items():
        tot_sample += len(v)
    
    logger.info("TRAINING IS STARTING ...")
    logger.info("=="*10)
    
    if tot_sample > 0:
        if args.semi_sup_max_steps == 0:
            args.max_steps = tot_sample*3//(args.per_gpu_train_batch_size * args.gradient_accumulation_steps)
        else:
            args.max_steps = args.semi_sup_max_steps
        args.warmup_steps = (args.max_steps*10)//100
        global_step, tr_loss, IsUpdated,  best_dev_scores, test_scores_in_best_src_dev = training_loop(
                            args, train_dataset, 
                            model, tokenizer, labels, pad_token_label_id, 
                            logger=logger, 
                            prev_best_dev_scores = best_dev_scores, 
                            prev_test_scores_in_best_src_dev=test_scores_in_best_src_dev
                        )
        logger.info("TRAINING DONE FOR [::] {}".format(args.external_data))
    else:
        logger.info("TRAINING POSTPONED DUE TO INSUFFICIENT SAMPLES [::]")


# def partial_single_self_training(
#         args, MODEL_CLASSES,
#         model, tokenizer, 
#         labels, pad_token_label_id, num_labels,
#         is_GMM_selection=0,
#         logger=None,
#     ):
#     best_dev_scores = None
#     test_scores_in_best_src_dev = None
#     logger.info("Dev Evaluation")
#     logger.info("--"*10)
#     dev_scores = evaluate(
#                     args, 
#                     model, tokenizer, labels, 
#                     pad_token_label_id, "dev", 
#                     prefix="", langs = args.tgt_lang,
#                     logger=logger
#                 )
#     logger.info("Test Evaluation")
#     logger.info("--"*10)
#     test_scores = evaluate(
#                         args, 
#                         model, tokenizer, labels, 
#                         pad_token_label_id, "test", 
#                         prefix="", langs = args.tgt_lang,
#                         logger=logger
#                     )
    
#     backup_overwrite_cache = args.overwrite_cache
#     args.overwrite_cache = True
    
#     logger.info("Source Train Data Loading")  
#     logger.info("--"*10)  
#     # source dataset load
#     train_dataset, _ = load_and_cache_examples(
#                         args, tokenizer, labels, pad_token_label_id, 
#                         mode="train", langs=args.src_lang, logger=logger
#                     )
#     if "self_tgt" in args.aug_desc.split(";"):
#         logger.info("Original Target Train Prediction")
#         logger.info("--"*10)
#         temp_train_data_percentage = args.train_data_percentage
#         args.train_data_percentage = 100
#         loss_dict, logit_dict, _, _, _= export_logit(
#                                 args, 
#                                 model, tokenizer, labels, 
#                                 pad_token_label_id, args.aug_mode, 
#                                 prefix="", langs = args.tgt_lang, logger=logger, debug=0
#                             )
#         args.train_data_percentage = temp_train_data_percentage

#         logger.info("Original Target Train Distillation")
#         logger.info("--"*10)
        
        
#         external_files = []
#         for dict_key in loss_dict.keys():
#             temp_top_k = args.top_k
#             args.top_k = 100
#             indices, gmm_model = data_distillation(
#                 args,
#                 dict_key,
#                 pseudo_loss_dict=loss_dict,
#                 logit_dict=logit_dict,
#                 mode="train",
#                 logger=logger,
#                 debug=0
#             )
#             args.top_k = temp_top_k
#             total_indexes = indices
#             address, indices = select_and_write_data(
#                         dict_key,
#                         args.output_dir,
#                         total_indexes, 
#                         loss_dict, 
#                         labels, 
#                         logger=logger,
#                         postfix = str(random.randint(0,10000000))
#                     )
#             external_files.append(address)
    
#         logger.info("Target Train Pseudo Data Loading")  
#         logger.info("--"*10)  
#         # target pseudo language dataset
#         temp_train_data_percentage = args.train_data_percentage
#         args.train_data_percentage = 100
#         target_dataset, _ = load_and_cache_examples(
#                             args, tokenizer, labels, pad_token_label_id, external_data=external_files,
#                             mode="aug", langs=args.tgt_lang, logger=logger
#                         )
#         args.train_data_percentage = temp_train_data_percentage
        
#         logger.info("Accumulating Target Train Pseudo Data to Training Datasets ...")   
#         for k, v in target_dataset.items():
#             assert k not in train_dataset
#             train_dataset[k] = v
    
#     langs = "None"
#     if "src_aug" in args.aug_desc.split(";"):
#         langs = args.src_lang
#     if "tgt_aug" in args.aug_desc.split(";"):
#         if langs == "None" or langs == "":
#             langs = args.tgt_lang
#         else:
#             langs = langs + ";" + args.tgt_lang

#     logger.info("Source {} and/or Target {} Augmented data prediction : langs flag {}...".format(args.src_lang, args.tgt_lang, langs))   
#     logger.info("--"*10)  
#     temp_train_data_percentage = args.train_data_percentage
#     args.train_data_percentage = 100
#     loss_dict, logit_dict, _, _, _= export_logit(
#                     args, 
#                     model, tokenizer, labels, 
#                     pad_token_label_id, "aug", external_data=args.external_data,
#                     prefix="", langs = langs, logger=logger, debug=0
#                 )
#     args.train_data_percentage = temp_train_data_percentage
    
#     for dict_key in loss_dict.keys():
#         logger.info("{} Distillation".format(dict_key))
#         logger.info("--"*10)
#         indices, gmm_model = data_distillation(
#             args,
#             dict_key,
#             pseudo_loss_dict=loss_dict,
#             logit_dict=logit_dict,
#             mode="train",
#             logger=logger,
#             debug=0
#         )
#         total_indexes = indices
#         address, indices = select_and_write_data(
#                     dict_key,
#                     args.output_dir,
#                     total_indexes, 
#                     loss_dict, 
#                     labels, 
#                     logger=logger,
#                     postfix = str(random.randint(0,10000000))
#                 )
#         lang = dict_key.split(";")[-1]
#         logger.info("{} Data Loading".format(dict_key))
#         logger.info("--"*10)
#         temp_train_data_percentage = args.train_data_percentage
#         args.train_data_percentage = 100
#         aug_dataset, _ = load_and_cache_examples(
#                     args, tokenizer, labels, pad_token_label_id,  external_data=[address],
#                     mode="aug", langs=lang, logger=logger
#                 )
#         args.train_data_percentage = temp_train_data_percentage
        
#         logger.info("Accumulating {} Pseudo Data to Training Datasets".format(dict_key))
#         logger.info("--"*10)
#         for k, v in aug_dataset.items():
#             assert k not in train_dataset
#             train_dataset[k] = v


#     args.overwrite_cache = backup_overwrite_cache
#     tot_sample = 0
#     for k, v in train_dataset.items():
#         tot_sample += len(v)
    
#     logger.info("TRAINING IS STARTING ...")
#     logger.info("=="*10)
    
#     if tot_sample > 0:
#         if args.semi_sup_max_steps == 0:
#             args.max_steps = tot_sample*3//(args.per_gpu_train_batch_size * args.gradient_accumulation_steps)
#         else:
#             args.max_steps = args.semi_sup_max_steps
#         args.warmup_steps = (args.max_steps*10)//100
#         global_step, tr_loss, IsUpdated,  best_dev_scores, test_scores_in_best_src_dev = training_loop(
#                             args, train_dataset, 
#                             model, tokenizer, labels, pad_token_label_id, 
#                             logger=logger, 
#                             prev_best_dev_scores = best_dev_scores, 
#                             prev_test_scores_in_best_src_dev=test_scores_in_best_src_dev
#                         )
#         logger.info("TRAINING DONE FOR [::] {}".format(args.external_data))
#     else:
#         logger.info("TRAINING POSTPONED DUE TO INSUFFICIENT SAMPLES [::]")



def classical_self_training(
        args, MODEL_CLASSES,
        model, tokenizer, 
        labels, pad_token_label_id, num_labels,
        is_GMM_selection=0,
        logger=None
    ):
    IsUpdated = True
    best_dev_scores = None
    test_scores_in_best_src_dev = None
    total_cnt = 0
    random.seed(args.seed)
    
    total_indexes = None
    while IsUpdated:
    
        ###################
        # evaluate the score of the best model
        ###################
        dev_scores = evaluate(
                            args, 
                            model, tokenizer, labels, 
                            pad_token_label_id, "dev", 
                            prefix="", langs = args.tgt_lang,
                            logger=logger
                        )
        test_scores = evaluate(
                            args, 
                            model, tokenizer, labels, 
                            pad_token_label_id, "test", 
                            prefix="", langs = args.tgt_lang,
                            logger=logger
                        )

        ###################
        # extract logit/loss etc. info from the model with respect to dataset.
        ###################
        loss_dict, logit_dict, _, _, _= export_logit(
                            args, 
                            model, tokenizer, labels, 
                            pad_token_label_id, args.aug_mode, 
                            prefix="", langs = args.tgt_lang, logger=logger
                        )

        external_files = []
        for dict_key in loss_dict.keys():
            indexes = select_data_from_logit(
                        args,
                        dict_key,
                        logit_dict, 
                        loss_dict,
                        path=args.output_dir, 
                        bin_increment=.01, 
                        top_k=args.top_k,
                        noise_threshold=0, 
                        min_length_restriction=-10, 
                        max_length_restriction=1500000,
                        mode="train",
                        logger=logger,
                        debug=0,
                        isGMM=0
                )

            total_indexes = indexes if total_indexes is None else list(set(total_indexes+indexes))
            address, indices = select_and_write_data(
                        dict_key,
                        args.output_dir,
                        total_indexes, 
                        loss_dict, 
                        labels, 
                        logger=logger,
                        postfix = str(random.randint(0,10000000))
                    )
            external_files.append(address)
        
        args.external_data = external_files
        backup_overwrite_cache = args.overwrite_cache
        args.overwrite_cache = True
        train_dataset, _ = load_and_cache_examples(
                            args, tokenizer, labels, pad_token_label_id, 
                            mode="aug", langs=args.tgt_lang, logger=logger
                        )
        args.overwrite_cache = backup_overwrite_cache

        tot_sample = 0
        for k, v in train_dataset.items():
            tot_sample += len(v)
        if tot_sample > 0:
            
            if args.semi_sup_max_steps == 0:
                args.max_steps = tot_sample*6//(args.per_gpu_train_batch_size * args.gradient_accumulation_steps)
            else:
                args.max_steps = args.semi_sup_max_steps
            args.warmup_steps = (args.max_steps*10)//100
            global_step, tr_loss, IsUpdated,  best_dev_scores, test_scores_in_best_src_dev = training_loop(
                                args, train_dataset, 
                                model, tokenizer, labels, pad_token_label_id, 
                                logger=logger, 
                                prev_best_dev_scores = best_dev_scores, 
                                prev_test_scores_in_best_src_dev=test_scores_in_best_src_dev
                            )
            logger.info("TRAINING DONE FOR [::] {}".format(args.external_data))
        else:
            logger.info("TRAINING POSTPONED DUE TO INSUFFICIENT SAMPLES [::]")
            IsUpdated = False
    
    


def pseudo_self_training(
        args, MODEL_CLASSES,
        model, tokenizer, 
        labels, pad_token_label_id,  num_labels,
        is_GMM_selection=0,
        logger=None
    ):

    IsUpdated = True
    best_dev_scores = None
    test_scores_in_best_src_dev = None
    total_cnt = 0
    random.seed(args.seed)

    while IsUpdated:
        
        ###################
        # load best model
        ###################
        config, tokenizer, model = load_model(
            args.model_type, MODEL_CLASSES, 
            args.best_dev_model, args.best_dev_config, args.tokenizer_name,
            num_labels, args.cache_dir, args.do_lower_case, args.device
        )
        
        ###################
        # evaluate the score of the best model
        ###################
        dev_scores = evaluate(
                            args, 
                            model, tokenizer, labels, 
                            pad_token_label_id, "dev", 
                            prefix="", langs = args.tgt_lang,
                            logger=logger
                        )
        test_scores = evaluate(
                            args, 
                            model, tokenizer, labels, 
                            pad_token_label_id, "test", 
                            prefix="", langs = args.tgt_lang,
                            logger=logger
                        )
        
        ###################
        # extract logit/loss etc. info from the model with respect to dataset.
        ###################
        loss_dict, logit_dict, _, _, _ = export_logit(
                            args, 
                            model, tokenizer, labels, 
                            pad_token_label_id, args.aug_mode, 
                            prefix="", langs = args.tgt_lang, logger=logger
                        )
        
        ###################
        # Create dataset
        ###################
        external_files = []
        for dict_key in loss_dict.keys():
            if is_GMM_selection == 0:
                indexes = np.array(list(range(len(loss_dict[dict_key]))))
                address, indices = select_and_write_data(
                    dict_key,
                    args.output_dir,
                    indexes, 
                    loss_dict, 
                    labels, 
                    logger=logger, 
                    postfix = str(random.randint(0,10000000))
                )
                external_files.append(address)
            else:
                # rng = []
                # val = 0.0
                # while val <= 1:
                #     val += .05
                #     rng.append(val)
                # rng = args.posterior_threshold
                # for posterior_threshold in rng: 
                    # args.posterior_threshold = [posterior_threshold]
                indexes, _ = select_samples_with_GMM(
                    args=args,
                    dict_key=dict_key, 
                    loss_dict=loss_dict, 
                    path=args.output_dir, 
                    bin_increment=.01, 
                    noise_threshold=args.noise_threshold, 
                    min_length_restriction=args.min_length_restriction, 
                    max_length_restriction=args.max_length_restriction,
                    mode=args.aug_mode,
                    logger=logger,
                    debug=0
                )
                address, indices = select_and_write_data(
                    dict_key,
                    args.output_dir,
                    indexes, 
                    loss_dict, 
                    labels, 
                    logger=logger,
                    postfix = str(random.randint(0,10000000)) + "." + str(round(args.posterior_threshold[0], 2))
                )
                external_files.append(address)
        
        ###################
        # Train
        ###################
        for external_file in external_files:
            args.external_data = [external_file]
            logger.info("NEW TRAINING LOOP [::]")
            logger.info("="*20)
            logger.info("="*20)
            logger.info("TRAINING MODEL ON [::] {}".format(args.external_data))
            backup_overwrite_cache = args.overwrite_cache
            args.overwrite_cache = True
            train_dataset, _ = load_and_cache_examples(
                                args, tokenizer, labels, pad_token_label_id, 
                                mode="aug", langs=args.tgt_lang, logger=logger
                            )
            args.overwrite_cache = backup_overwrite_cache
            tot_sample = 0
            for k, v in train_dataset.items():
                tot_sample += len(v)
            if tot_sample > 0:
                if args.semi_sup_max_steps == 0:
                    args.max_steps = tot_sample*3//(args.per_gpu_train_batch_size * args.gradient_accumulation_steps)
                else:
                    args.max_steps = args.semi_sup_max_steps
                args.warmup_steps = (args.max_steps*10)//100
                global_step, tr_loss, IsUpdated,  best_dev_scores, test_scores_in_best_src_dev = training_loop(
                                    args, train_dataset, 
                                    model, tokenizer, labels, pad_token_label_id, 
                                    logger=logger, 
                                    prev_best_dev_scores = best_dev_scores, 
                                    prev_test_scores_in_best_src_dev=test_scores_in_best_src_dev
                                )
                logger.info("TRAINING DONE FOR [::] {}".format(args.external_data))
            else:
                logger.info("TRAINING POSTPONED DUE TO INSUFFICIENT SAMPLES [::]")
            if is_GMM_selection == 1:
                IsUpdated = False
            


def select_theta(args, MODEL_CLASSES, num_labels, theta, retrain=0, logger=None):
    if retrain == 0:
        model_name_or_path = os.path.join(theta, "pytorch_model.bin")
        config_name = os.path.join(theta, "config.json")
    elif retrain == 1:
        model_name_or_path = "bert-base-multilingual-cased"
        config_name = "bert-base-multilingual-cased"
    else:
        raise NotImplementedError()
    
    config, tokenizer, model = load_model(
            args.model_type, MODEL_CLASSES, 
            model_name_or_path, config_name, args.tokenizer_name,
            num_labels, args.cache_dir, args.do_lower_case, args.device
        )
    if retrain == 1:
        save_address = save_model_checkpoint(
                        args, None, None,
                        model, 
                        logger=logger,
                        overwrite_address = theta
                    )
    return config, tokenizer, model


def retrieve_sentences_with_idx(data, intersect_indices):
    sentences = read_from_path(data[1].split(";")[0], encoding=data[1].split(";")[1])
    indices = data[2]
    assert len(sentences) == len(indices)
    intersect_sentences = []
    for index, sentence in zip(indices, sentences):
        if index in intersect_indices:
            intersect_sentences.append(sentence)
    return intersect_sentences

def combine_file_name(k_file_info, j_file_info):
    root_address = os.path.split(k_file_info)[0]
    k_file_name = os.path.split(k_file_info)[1].split(";")[0]
    j_file_name = os.path.split(j_file_info)[1].split(";")[0]
    assert k_file_info.split(";")[1] == j_file_info.split(";")[1]
    assert k_file_info.split(";")[2] == j_file_info.split(";")[2]
    prefix_name = k_file_name
    postfix_name = ""
    idx = -1
    for __i, (k_content, j_content) in enumerate(zip(k_file_name.split("."), j_file_name.split("."))):
        if k_content != j_content:
            idx = __i
            break
    assert idx != -1
    for __i, j_content in enumerate(j_file_name.split(".")):
        if __i >= idx:
            postfix_name = postfix_name + "." + j_content
    file_name = prefix_name + "._._." + postfix_name 
    # print(k_file_name)
    # print(j_file_name)
    # print(file_name)
    full_file_name = os.path.join(root_address, file_name)
    full_file_info = full_file_name+'.int' + ';' + k_file_info.split(";")[1] + ';' + k_file_info.split(";")[2]
    return full_file_info


def get_intersected_dataset_from_indices(kdata, jdata, logger=None):
    kindices = kdata[2]
    jindices = jdata[2]
    intersect_indices = [ index for index in kindices if index in jindices ]
    logger.info("Original data size kdata = {}, jdata = {},  k_data \intersect j_data = {} ".format(len(kindices), len(jindices), len(intersect_indices)))
    # print(intersect_indices)
    k_intersect_sentence = retrieve_sentences_with_idx(kdata, intersect_indices)
    j_intersect_sentence = retrieve_sentences_with_idx(jdata, intersect_indices)
    assert len(k_intersect_sentence) == len(j_intersect_sentence)
    intersected_sentences_with_lables = []
    for k_sent_info, j_sent_info in zip(k_intersect_sentence, j_intersect_sentence):
        flag = 1
        for k_w, j_w in zip(k_sent_info, j_sent_info):
            # print(k_w[0],  j_w[0] )
            assert k_w[0] == j_w[0]
            if k_w[-1] != j_w[-1]:
                flag = 0
                break
        if flag:
            intersected_sentences_with_lables.append(k_sent_info)
    logger.info("{} and {} dataset is INTERSECTING ...".format(kdata[1], jdata[1]))
    full_file_info = combine_file_name(kdata[1], jdata[1]) 
    logger.info("intersected file name {}".format(full_file_info))
    file_info = write_conll_data(full_file_info, intersected_sentences_with_lables, logger=logger)
    return file_info, intersect_indices


def get_unioned_dataset_from_indices(kdata, jdata, logger=None):
    kindices = kdata[2]
    jindices = jdata[2]
    intersect_indices = [ index for index in kindices if index in jindices ]
    logger.info("Original data size kdata = {}, jdata = {},  k_data \intersect j_data = {} ".format(len(kindices), len(jindices), len(intersect_indices)))
    # print(intersect_indices)
    k_intersect_sentence = retrieve_sentences_with_idx(kdata, intersect_indices)
    j_intersect_sentence = retrieve_sentences_with_idx(jdata, intersect_indices)
    assert len(k_intersect_sentence) == len(j_intersect_sentence)
    sentences_with_lables = []
    for k_sent_info, j_sent_info in zip(k_intersect_sentence, j_intersect_sentence):
        flag = 1
        for k_w, j_w in zip(k_sent_info, j_sent_info):
            # print(k_w[0],  j_w[0] )
            assert k_w[0] == j_w[0]
            if k_w[-1] != j_w[-1]:
                flag = 0
                break
        if flag:
            sentences_with_lables.append(k_sent_info)
#     print(len(sentences_with_lables))
    k_not_intersected, j_not_intersected = [], []
    for index in kindices:
        if index not in intersect_indices:
            k_not_intersected.append(index)
    for index in jindices:
        if index not in intersect_indices:
            j_not_intersected.append(index)
    k_not_intersect_sentence = retrieve_sentences_with_idx(kdata, k_not_intersected)
    j_not_intersect_sentence = retrieve_sentences_with_idx(jdata, j_not_intersected)
    for k_sent_info in  k_not_intersect_sentence:
        sentences_with_lables.append(k_sent_info)
    for j_sent_info in  j_not_intersect_sentence:
        sentences_with_lables.append(j_sent_info)    
    logger.info("k_not_intersected {}, j_not_intersected {}, intersect_indices {}, total indices {}"
        .format(len(k_not_intersect_sentence), len(j_not_intersect_sentence), len(intersect_indices), len(sentences_with_lables)))
    logger.info("{} and {} dataset is UNIONED ...".format(kdata[1], jdata[1]))
    full_file_info = combine_file_name(kdata[1], jdata[1]) 
    logger.info("UNIONED file name {}".format(full_file_info))
    file_info = write_conll_data(full_file_info, sentences_with_lables, logger=logger)
    unioned_indices = intersect_indices + k_not_intersected + j_not_intersected
    return file_info, unioned_indices


def unsupervised_augmentation(args, data_address_list, logger=None):
    address, output_aug_sentences = augment_data(
                    dataset_list=data_address_list, 
                    output_dir=args.output_dir,
                    aug_type="successive_max", 
                    only_ner_aug=0,
                    topk=1, 
                    aug_per=15,
                    num_of_aug=5,
                    seed=1234,
                    logger=logger,
                    is_small_name=1,
                    debug=0
                )
    return address, output_aug_sentences


def supervised_augmentation(args, X_src_p, logger=None):
    X_src_p_p_ss_mx, output_aug_sentences_ = unsupervised_augmentation(args, X_src_p, logger)
    X_src_p_p_per_tkn, output_aug_sentences__ = augment_data(
        dataset_list=X_src_p, 
        output_dir=args.output_dir,
        aug_type="per_token", 
        only_ner_aug=1,
        topk=3, 
        aug_per=15,
        num_of_aug=5,
        seed=1234,
        logger=logger,
        is_small_name=1,
        debug=0
    )
    X_src_p_p = X_src_p_p_ss_mx + X_src_p_p_per_tkn
    return X_src_p_p


def augmentation_lable_post_processing(
        args, X_src_p_p, theta, 
        num_labels, MODEL_CLASSES, labels, pad_token_label_id, 
        post_tag, logger=None
    ):
    
    orig_X_src_p_p  = {}
    for address in X_src_p_p:
        orig_X_src_p_p[address] = read_from_path(address.split(";")[0], encoding=address.split(";")[1])
    
    config, tokenizer, model = select_theta(args, MODEL_CLASSES, num_labels, theta, logger=logger)
    
    temp = args.overwrite_cache
    args.overwrite_cache=True
    eval_dataset, _ = load_and_cache_examples(
                        args, tokenizer, labels, pad_token_label_id, 
                        mode="aug", langs=args.aug_lang, logger=logger,
                        external_data=X_src_p_p
                    )
    args.overwrite_cache = temp
    test_scores = evaluate(
            args, 
            model, tokenizer, labels, 
            pad_token_label_id, "aug", 
            prefix="", 
            langs = args.aug_lang, 
            logger=logger,
            eval_dataset=eval_dataset
        )
    new_aug_dataset_address = []
    for k, v in test_scores.items():
        predictions = v[1]
        assert k in orig_X_src_p_p
        orig_label = orig_X_src_p_p[k]
        assert len(orig_label) == len(predictions)
        current_lang = k.split(";")[-1]
        all_new_sentence_info = []
        for pred, orig in zip(predictions, orig_label):
            assert len(pred) == len(orig)
    #         print("pred : ", pred)
    #         print("orig : ", orig)
            new_labels = []
            for p, o in zip(pred, orig):
                if current_lang in args.aug_label_propagate:
                    new_labels.append(o[-1])
                    if new_labels[-1] == "X":
                        new_labels[-1] = p
                else:
                    new_labels.append(p)
    #         print("new_label : ", new_labels)
            new_sent_info = [ [w[0], w[1], l] for w, l in zip(orig, new_labels)]
    #         print("new_sent_info : ", new_sent_info)
    #         break
            all_new_sentence_info.append(new_sent_info)
        file_info = k.split(";")[0] + ".prd.mrg." + post_tag + ";" + k.split(";")[1] + ";" + k.split(";")[2]
        file_info = write_conll_data(file_info, all_new_sentence_info, logger=logger)
        new_aug_dataset_address.append(file_info)
    return new_aug_dataset_address
    
def get_training_dataset(
            args, semi_sup_epoch_idx,
            all_X_src_p_p, all_X_src_p,
            all_X_tgt_p_p, all_X_tgt_p
        ):
    external_training_data_address_list = []
    if args.retrain or (args.partial_train_in_semi_sup_epochs and semi_sup_epoch_idx==0):
        for dataset in args.train:
            lang = dataset.split(";")[-1]
            if lang == args.src_lang:
                external_training_data_address_list.append(dataset)
    
    if args.partial_train_in_semi_sup_epochs == 0 or \
        (semi_sup_epoch_idx > 0 and semi_sup_epoch_idx < (args.max_semi_sup_epoch-1) ):
        for dataset in all_X_src_p:
            external_training_data_address_list.append(dataset)
        for dataset in all_X_src_p_p:
            external_training_data_address_list.append(dataset)
        for dataset in all_X_tgt_p:
            external_training_data_address_list.append(dataset)
        for dataset in all_X_tgt_p_p:
            external_training_data_address_list.append(dataset)
    else:
        if semi_sup_epoch_idx == 0:
            for dataset in all_X_src_p:
                external_training_data_address_list.append(dataset)
            for dataset in all_X_src_p_p:
                external_training_data_address_list.append(dataset)
        elif semi_sup_epoch_idx == (args.max_semi_sup_epoch-1):
            for dataset in all_X_tgt_p:
                external_training_data_address_list.append(dataset)
            for dataset in all_X_tgt_p_p:
                external_training_data_address_list.append(dataset)
    return external_training_data_address_list

def data_distillation(
        args,
        dict_key,
        pseudo_loss_dict=None,
        logit_dict=None,
        mode="train",
        logger=None,
        debug=0
    ):
    gmm_model= None
    if args.data_distil_type =='gmm':
        logger.info("applying GMM on {} data ...".format(dict_key))
        indices, gmm_model = select_samples_with_GMM(
                    args,
                    dict_key,
                    pseudo_loss_dict,
                    path=args.output_dir, 
                    bin_increment=.01, 
                    noise_threshold=0, 
                    min_length_restriction=-10, 
                    max_length_restriction=1000000,
                    mode=mode,
                    logger=logger,
                    debug=debug
                )       
    elif args.data_distil_type =='top_k':                 
        logger.info("Selecting topk : {} \% confident data from {} ...".format(args.top_k, dict_key))
        indices = select_data_from_logit(
            args,
            dict_key,
            logit_dict, 
            pseudo_loss_dict,
            args.output_dir, 
            bin_increment=.01, 
            top_k=args.top_k,
            noise_threshold=0, 
            min_length_restriction=10, 
            max_length_restriction=150,
            mode=mode,
            logger=logger,
            debug=debug,
            isGMM=0
        )
    else:
        raise NotImplementedError()
    return indices, gmm_model
    
def getNullScore(args):
    scores = {}
    for i in range(len(args.thetas)):
        scores[i] = (None, None) # best_dev_scores, test_scores_in_best_src_dev
    return scores

def getMultiHeadNullScore(args):
    scores = {}
    scores[-1] = ({}, {})
    return scores

def co_distill_log(args, logger):
    args.top_k = min(100, args.top_k)
    if args.data_distil_type == "top_k":
        logger.info("Co-distillation value top_k : {}".format(args.top_k))
    else:
        logger.info("Co-distillation value gmm, "
                    "\n noise_threshold : {}\n" 
                    "n_mixture_component : {} \n" 
                    "n_mixture_select : {}\n" 
                    "posterior_threshold : {}\n" "covariance_type : {}".format(
                        args.noise_threshold,
                        args.n_mixture_component,
                        args.n_mixture_select,
                        args.posterior_threshold,
                        args.covariance_type)
                )
        
def distill(args, model, tokenizer, labels, pad_token_label_id, lang, mode="train", theta_idx=0, 
           semi_sup_epoch_idx=0, k=0, external_data=None, examples=None, head_idx=0, logger=None):    
    
    logger.info("Retriving pseudo informations for {} languages {} data ...".format(lang, mode))
    # load inferred data from model.
    pseudo_loss_dict, logit_dict, orig_lable_bank, orig_loss_dict, guids = export_logit(
                args, 
                model, tokenizer, labels, 
                pad_token_label_id, mode, 
                external_data=external_data, examples=examples,
                prefix="", langs = lang, head_idx=head_idx, logger=logger
            )
    assert len(pseudo_loss_dict.keys()) == 1
    # assert len(guids) == len(pseudo_loss_dict[ list(pseudo_loss_dict.keys())[0] ])
    
    for dict_key, v in pseudo_loss_dict.items():  
        logger.info("Processing {} data ...".format(dict_key))
        lang = dict_key.split(";")[-1]
        indices, gmm_model = data_distillation(
                    args,
                    dict_key,
                    pseudo_loss_dict=pseudo_loss_dict,
                    logit_dict=logit_dict,
                    mode=mode,
                    logger=logger,
                    debug=0
                )
        indices = sorted(indices)
        if examples is None:
            percentage = args.train_data_percentage if mode == "train" else 100
            examples = read_examples_from_file(
                            address = dict_key.split(";")[0], 
                            encoding = dict_key.split(";")[1], lang = dict_key.split(";")[2], 
                            mode=mode, 
                            seed=args.seed, percentage=percentage
                        ) 
        logger.info("writting dataset ... ")
        address, indices = select_and_write_data(
            dict_key,
            args.output_dir,
            indices, 
            examples,
            pseudo_loss_dict, 
            labels, 
            mode="distill",
            seed=args.seed,
            logger=logger,
            postfix = str(random.randint(0,1000000)) + ".unsup" + \
                        ".th_" + str(theta_idx) + ".sse_" + str(semi_sup_epoch_idx)+".k_" + str(k) + ".h_" + str(head_idx)
        )
        return dict_key, address, indices, gmm_model
           


def export_model_to_experiment_folder(
        args, MODEL_CLASSES, labels, 
        pad_token_label_id, num_labels, logger
    ):
    
    for theta_idx, theta in enumerate(args.thetas):
        # save the model to an additional place. 
        logger.info("loading model from {} ...".format(theta))
        config, tokenizer, model = select_theta(args, MODEL_CLASSES, num_labels, theta, logger=logger)
        logger.info("Evaluating tgt language(s) dev and test dataset{} ...".format(args.tgt_lang))
        for head_idx in range(args.num_of_heads):
            dev_scores = evaluate(
                            args, 
                            model, tokenizer, labels, 
                            pad_token_label_id, "dev", 
                            prefix="", langs = args.tgt_lang,
                            logger=logger,
                            head_idx=head_idx
                        )    
            test_scores = evaluate(
                                args, 
                                model, tokenizer, labels, 
                                pad_token_label_id, "test", 
                                prefix="", langs = args.tgt_lang,
                                logger=logger,
                                head_idx=head_idx
                            )
        save_address = save_model_checkpoint(
                args, args.output_dir, "init.model." + str(theta_idx), 
                model, 
                logger=logger
            )
        logger.info("for theta_idx = {} : Checkpoint saved at {} ...".format(theta_idx, save_address))
        # new theta address (so that previous thetas are intact and can be debugged later)
        args.thetas[theta_idx] = save_address  
    return args


def gen_LM(dataset_info_list, mode, seed, percent, lang, pkl_list, logger):
    logger.info("gen_LM function started")
    address = None
    for dt in dataset_info_list:
        if dt.split(";")[-1] == lang:
            address = dt.split(";")[0]
            encoding = dt.split(";")[1]
    assert address is not None    
    logger.info("Loading {} :: {} **dataset** {} is loading with {}% with seed {}".format(lang, mode, address, percent, seed))
    pseudo_examples = read_examples_from_file(address, encoding, lang, mode, seed, percentage=percent)

    example_guid_list = []
    for example in pseudo_examples:
        # assert mode == example.guid.split("-")[0]
        example_guid_list.append(example.guid.split("-")[-1])
    example_guid_set = set(example_guid_list)
    
    pkl_file = None
    for pt in pkl_list:
        if pt.split(";")[-1] == lang:
            pkl_file = pt.split(";")[0]
    assert pkl_file is not None    
    logger.info("Loading {} :: {} **Augmented dataset** from {} (pkl) with seed {}".format(lang, mode, pkl_file, seed))
    with open(pkl_file, "rb") as filePtr:
        all_aug_examples = pickle.load(filePtr)
   
    aug_examples = []
    for example in all_aug_examples:
        # assert mode == example.guid.split("-")[0]
        if example.guid.split("-")[-1] in example_guid_set:
            labels = example.labels
            for i, _ in enumerate(labels):
                if labels[i] == "X":
                    labels[i] = "O"
            example.labels = labels
            aug_examples.append(example)
    logger.info("Loaded {} augmented examples for {} language".format(len(aug_examples), lang))
    
    return (aug_examples, pkl_file+";pkl;"+lang, lang)


def check_acceptance(aug_desc, semi_sup_epoch, curr_aug_type): 
    aug_desc = aug_desc.split("|")
    for aug in aug_desc:
        epoch_num = aug.split(":")[0]
        if int(epoch_num) == semi_sup_epoch:
            if curr_aug_type in aug.split(":")[1]:
                return 1
    return 0


def Multi_Mix_Single_Model_Multi_Head(
        args, 
        MODEL_CLASSES, 
        labels, 
        pad_token_label_id, 
        num_labels,
        logger=None
    ):
    assert len(args.thetas) == 1
    
    logger.info("Starting Multimix Training")
    logger.info("-"*20)
    
    scores = getNullScore(args)
    scores = checkSinglePrevScores(args.thetas[0], scores)
    logger.info("SCORES : {}".format(json.dumps(scores, indent=4)))

    args = export_model_to_experiment_folder(
        args, 
        MODEL_CLASSES, labels, pad_token_label_id,
        num_labels,
        logger
    )
    
    co_distill_log(args, logger)
    
    for semi_sup_epoch_idx in range(args.semi_sup_start_epoch, args.max_semi_sup_epoch): 
        logger.info("\n\nSTARTING SEMISUPERVISED EPOCH {}\n\n".format(semi_sup_epoch_idx))
        config, tokenizer, model = select_theta(args, MODEL_CLASSES, num_labels, args.thetas[0], logger=logger)
        logger.info("Evaluating tgt language(s) dev and test dataset{} with model K = {} ...".format(args.tgt_lang, 0))
        
        for head_idx in range(args.num_of_heads):
            dev_scores_k = evaluate(
                            args, 
                            model, tokenizer, labels, 
                            pad_token_label_id, "dev", 
                            prefix="", langs = args.tgt_lang,
                            logger=logger,
                            head_idx=head_idx
                        )    
            test_scores_k = evaluate(
                            args, 
                            model, tokenizer, labels, 
                            pad_token_label_id, "test", 
                            prefix="", langs = args.tgt_lang,
                            logger=logger, 
                            head_idx=head_idx
                        )
        # All the datasets used for training will be stored in `train_dataset_addresses`
        train_dataset_addresses = []
        
        # Selecting source dataset
        source_train_dataset = None
        for dt in args.train:
            if dt.split(";")[-1] == args.src_lang:
                source_train_dataset = dt
                break
        assert source_train_dataset is not None
        
        if check_acceptance(args.aug_desc, semi_sup_epoch_idx, "src"):
            train_dataset_addresses.append(source_train_dataset)

        # source language data augmentation 
        if check_acceptance(args.aug_desc, semi_sup_epoch_idx, "src_aug"):
            # Generation is precomputed
            data_src_aug, data_src_aug_address, data_src_lang = gen_LM(
                    dataset_info_list=args.train, 
                    mode="aug", seed=args.seed, percent=args.train_data_percentage, 
                    lang=args.src_lang, 
                    pkl_list=args.external_data,
                    logger = logger
                )
            # data_src_aug_file_name = os.path.join(args.output_dir, os.path.split(data_src_aug_address)[-1]) + ";pkl;" + data_src_lang
            
            (dict_key_src, address_info_src, indices_src, gmm_model_src) = distill(
                                        args, model, tokenizer, labels, pad_token_label_id, 
                                        data_src_lang, mode="aug", 
                                        theta_idx=0, semi_sup_epoch_idx=semi_sup_epoch_idx, k=0, 
                                        external_data=[data_src_aug_address], examples=data_src_aug, logger=logger
                                    )
            train_dataset_addresses.append( address_info_src )
    
        # Original target train 
        if check_acceptance(args.aug_desc, semi_sup_epoch_idx, "tgt_self"):
            (dict_key_tgt, address_info_tgt, indices_tgt, gmm_model_tgt) = distill(
                                args, model, tokenizer, labels, pad_token_label_id, 
                                lang=args.tgt_lang, mode="train", 
                                theta_idx=0, semi_sup_epoch_idx=semi_sup_epoch_idx, k=0, 
                                external_data=None, logger=logger
                            )
            train_dataset_addresses.append( address_info_tgt )
            
        if check_acceptance(args.aug_desc, semi_sup_epoch_idx, "tgt_aug"):
            # Generation is precomputed
            data_tgt_aug, data_tgt_aug_address, data_tgt_lang = gen_LM(dataset_info_list=args.train, 
                                                                    mode="aug", seed=args.seed, percent=args.train_data_percentage, 
                                                                    lang=args.tgt_lang, 
                                                                    pkl_list=args.external_data, logger=logger)
            # data_tgt_aug_file_name = os.path.join(args.output_dir, os.path.split(data_tgt_aug_address)[-1]) + ";pkl;" + data_tgt_lang
            (dict_key_tgt_aug, address_info_tgt_aug, indices_tgt_aug, gmm_model_tgt_aug) = distill(
                                        args, model, tokenizer, labels, pad_token_label_id, 
                                        lang=data_tgt_lang, mode="aug", 
                                        theta_idx=0, semi_sup_epoch_idx=semi_sup_epoch_idx, k=0, 
                                        external_data=[data_tgt_aug_address], examples=data_tgt_aug, logger=logger
                                    )
        
            train_dataset_addresses.append( address_info_tgt_aug )
        
        logger.info("training datasets are :")
        for idx, dataset in enumerate(train_dataset_addresses):
            logger.info("{}. {} , total number of sample {}".format(
                    idx, dataset, 
                    len(read_from_path(dataset.split(";")[0], encoding=dataset.split(";")[1]))
                )
            )
        
        temp = args.overwrite_cache
        args.overwrite_cache = True
        train_dataset, guids = load_and_cache_examples(
                    args, tokenizer, labels, pad_token_label_id, 
                    mode="aug", langs=args.aug_lang, external_data=train_dataset_addresses, logger=logger
                )
        args.overwrite_cache = temp
        
        tot_sample = 0
        for _, v in train_dataset.items():
            tot_sample += len(v)
        logger.info("Total number of sample {}".format(tot_sample))
        if tot_sample > 0:
            args.max_steps = int((tot_sample * args.num_train_epochs)//(args.per_gpu_train_batch_size * args.gradient_accumulation_steps))
            args.warmup_steps = int((args.max_steps*10)//100)
            logger.info("Total number of steps {}".format(args.max_steps))
            logger.info("Number of warmup steps {}".format(args.warmup_steps))
            best_dev_scores, test_scores_in_best_src_dev = scores[0]
            
            save_address = save_model_checkpoint(
                args, args.output_dir, "sse_" + str(semi_sup_epoch_idx)+".k_{}".format(0), 
                model, checkpoint="checkpoint",
                logger=logger
            )
            global_step, tr_loss, isUpdated,  best_dev_scores, test_scores_in_best_src_dev = training_loop(
                            args, train_dataset, 
                            model, tokenizer, labels, pad_token_label_id, 
                            logger=logger, 
                            prev_best_dev_scores = best_dev_scores, 
                            prev_test_scores_in_best_src_dev=test_scores_in_best_src_dev, 
                            tf_board_header= "sse_{}.k_{}".format(semi_sup_epoch_idx, 0)
                        )
            
            scores[0] = (best_dev_scores, test_scores_in_best_src_dev)
            logger.info("SEMI_SUP_EPOCH {}, best_dev_scores - {}".format(semi_sup_epoch_idx, best_dev_scores))
            logger.info("SEMI_SUP_EPOCH {}, test_scores_in_best_src_dev - {}".format(semi_sup_epoch_idx, test_scores_in_best_src_dev))
            
            save_address = save_model_checkpoint(
                args, args.output_dir, "sse_" + str(semi_sup_epoch_idx)+".k_{}_post".format(0), 
                model, checkpoint="checkpoint",
                logger=logger
            )
            
            logger.info("SCORES : {}".format(json.dumps(scores, indent=4)))
            with open(os.path.join(args.output_dir, 'scores.json'), 'w') as outfile:
                outfile.write(json.dumps(scores, indent=4))
            
            # tensor board log update
            
            tb_path = os.path.join(args.output_dir, "tf_board")
            if not os.path.exists(tb_path):
                os.makedirs(tb_path)
            tb_writer = SummaryWriter(tb_path)
            
            for score_d_k, score_v in best_dev_scores.items():
                tb_writer.add_scalar("sse_dev_th-{}_{}-F1".format(0, score_d_k), score_v, semi_sup_epoch_idx)
            for score_d_k, score_v in test_scores_in_best_src_dev.items():
                tb_writer.add_scalar("sse_test_th-{}_{}-F1".format(0, score_d_k), score_v, semi_sup_epoch_idx)





def Multi_Mix_Single_Model(
        args, 
        MODEL_CLASSES, 
        labels, 
        pad_token_label_id, 
        num_labels,
        logger=None
    ):
    assert len(args.thetas) == 1
    
    logger.info("Starting Multimix Training")
    logger.info("-"*20)
    
    scores = getNullScore(args)
    scores = checkSinglePrevScores(args.thetas[0], scores)
    logger.info("SCORES : {}".format(json.dumps(scores, indent=4)))
    
    args = export_model_to_experiment_folder(
        args, 
        MODEL_CLASSES, labels, pad_token_label_id,
        num_labels,
        logger
    )
    
    co_distill_log(args, logger)
    logger.info("SCORES : {}".format(json.dumps(scores, indent=4)))
    
    for semi_sup_epoch_idx in range(args.semi_sup_start_epoch, args.max_semi_sup_epoch): 
        logger.info("\n\nSTARTING SEMISUPERVISED EPOCH {}\n\n".format(semi_sup_epoch_idx))
        config, tokenizer, model = select_theta(args, MODEL_CLASSES, num_labels, args.thetas[0], logger=logger)
        logger.info("Evaluating tgt language(s) dev and test dataset{} with model K = {} ...".format(args.tgt_lang, 0))
        dev_scores_k = evaluate(
                        args, 
                        model, tokenizer, labels, 
                        pad_token_label_id, "dev", 
                        prefix="", langs = args.tgt_lang,
                        logger=logger
                    )    
        test_scores_k = evaluate(
                        args, 
                        model, tokenizer, labels, 
                        pad_token_label_id, "test", 
                        prefix="", langs = args.tgt_lang,
                        logger=logger
                    )
        # All the datasets used for training will be stored in `train_dataset_addresses`
        train_dataset_addresses = []
        
        # Selecting source dataset
        source_train_dataset = None
        for dt in args.train:
            if dt.split(";")[-1] == args.src_lang:
                source_train_dataset = dt
                break
        assert source_train_dataset is not None
        
        if check_acceptance(args.aug_desc, semi_sup_epoch_idx, "src"):
            train_dataset_addresses.append(source_train_dataset)

        # source language data augmentation 
        if check_acceptance(args.aug_desc, semi_sup_epoch_idx, "src_aug"):
            # Generation is precomputed
            data_src_aug, data_src_aug_address, data_src_lang = gen_LM(
                    dataset_info_list=args.train, 
                    mode="aug", seed=args.seed, percent=args.train_data_percentage, 
                    lang=args.src_lang, 
                    pkl_list=args.external_data,
                    logger = logger
                )
            # data_src_aug_file_name = os.path.join(args.output_dir, os.path.split(data_src_aug_address)[-1]) + ";pkl;" + data_src_lang
            
            (dict_key_src, address_info_src, indices_src, gmm_model_src) = distill(
                                        args, model, tokenizer, labels, pad_token_label_id, 
                                        data_src_lang, mode="aug", 
                                        theta_idx=0, semi_sup_epoch_idx=semi_sup_epoch_idx, k=0, 
                                        external_data=[data_src_aug_address], examples=data_src_aug, logger=logger
                                    )
            train_dataset_addresses.append( address_info_src )
    
        # Original target train 
        if check_acceptance(args.aug_desc, semi_sup_epoch_idx, "tgt_self"):
            (dict_key_tgt, address_info_tgt, indices_tgt, gmm_model_tgt) = distill(
                                args, model, tokenizer, labels, pad_token_label_id, 
                                lang=args.tgt_lang, mode="train", 
                                theta_idx=0, semi_sup_epoch_idx=semi_sup_epoch_idx, k=0, 
                                external_data=None, logger=logger
                            )
            train_dataset_addresses.append( address_info_tgt )
            
        if check_acceptance(args.aug_desc, semi_sup_epoch_idx, "tgt_aug"):
            # Generation is precomputed
            data_tgt_aug, data_tgt_aug_address, data_tgt_lang = gen_LM(dataset_info_list=args.train, 
                                                                    mode="aug", seed=args.seed, percent=args.train_data_percentage, 
                                                                    lang=args.tgt_lang, 
                                                                    pkl_list=args.external_data, logger=logger)
            # data_tgt_aug_file_name = os.path.join(args.output_dir, os.path.split(data_tgt_aug_address)[-1]) + ";pkl;" + data_tgt_lang
            (dict_key_tgt_aug, address_info_tgt_aug, indices_tgt_aug, gmm_model_tgt_aug) = distill(
                                        args, model, tokenizer, labels, pad_token_label_id, 
                                        lang=data_tgt_lang, mode="aug", 
                                        theta_idx=0, semi_sup_epoch_idx=semi_sup_epoch_idx, k=0, 
                                        external_data=[data_tgt_aug_address], examples=data_tgt_aug, logger=logger
                                    )
        
            train_dataset_addresses.append( address_info_tgt_aug )
        
        logger.info("training datasets are :")
        for idx, dataset in enumerate(train_dataset_addresses):
            logger.info("{}. {} , total number of sample {}".format(
                    idx, dataset, 
                    len(read_from_path(dataset.split(";")[0], encoding=dataset.split(";")[1]))
                )
            )
        
        temp = args.overwrite_cache
        args.overwrite_cache = True
        train_dataset, guids = load_and_cache_examples(
                    args, tokenizer, labels, pad_token_label_id, 
                    mode="aug", langs=args.aug_lang, external_data=train_dataset_addresses, logger=logger
                )
        args.overwrite_cache = temp
        
        tot_sample = 0
        for _, v in train_dataset.items():
            tot_sample += len(v)
        logger.info("Total number of sample {}".format(tot_sample))
        if tot_sample > 0:
            args.max_steps = int((tot_sample * args.num_train_epochs)//(args.per_gpu_train_batch_size * args.gradient_accumulation_steps))
            args.warmup_steps = int((args.max_steps*10)//100)
            logger.info("Total number of steps {}".format(args.max_steps))
            logger.info("Number of warmup steps {}".format(args.warmup_steps))
            best_dev_scores, test_scores_in_best_src_dev = scores[0]
            
            save_address = save_model_checkpoint(
                args, args.output_dir, "sse_" + str(semi_sup_epoch_idx)+".k_{}".format(0), 
                model, checkpoint="checkpoint",
                logger=logger
            )
            global_step, tr_loss, isUpdated,  best_dev_scores, test_scores_in_best_src_dev = training_loop(
                            args, train_dataset, 
                            model, tokenizer, labels, pad_token_label_id, 
                            logger=logger, 
                            prev_best_dev_scores = best_dev_scores, 
                            prev_test_scores_in_best_src_dev=test_scores_in_best_src_dev, 
                            tf_board_header= "sse_{}.k_{}".format(semi_sup_epoch_idx, 0)
                        )
            
            scores[0] = (best_dev_scores, test_scores_in_best_src_dev)
            logger.info("SEMI_SUP_EPOCH {}, best_dev_scores - {}".format(semi_sup_epoch_idx, best_dev_scores))
            logger.info("SEMI_SUP_EPOCH {}, test_scores_in_best_src_dev - {}".format(semi_sup_epoch_idx, test_scores_in_best_src_dev))
            
            if isUpdated:
                update_single_theta_model(args, semi_sup_epoch_idx, logger)
            
            
            logger.info("SCORES : {}".format(json.dumps(scores, indent=4)))
            with open(os.path.join(args.output_dir, 'scores.json'), 'w') as outfile:
                outfile.write(json.dumps(scores, indent=4))
            
            # tensor board log update
            
            tb_path = os.path.join(args.output_dir, "tf_board")
            if not os.path.exists(tb_path):
                os.makedirs(tb_path)
            tb_writer = SummaryWriter(tb_path)
            
            for score_d_k, score_v in best_dev_scores.items():
                tb_writer.add_scalar("sse_dev_th-{}_{}-F1".format(0, score_d_k), score_v, semi_sup_epoch_idx)
            for score_d_k, score_v in test_scores_in_best_src_dev.items():
                tb_writer.add_scalar("sse_test_th-{}_{}-F1".format(0, score_d_k), score_v, semi_sup_epoch_idx)

def checkSinglePrevScores(theta, scores):
    if (theta.endswith("/") or theta.endswith("\\")):
        theta = theta[0:-1]            
    theta_address = os.path.join(os.path.split(theta)[0], "scores.json")
    if os.path.exists(theta_address):
        with open(theta_address, "r") as filePtr:
            scores = json.load(filePtr)
    new_score = {}
    for k, v in scores.items():
        new_score[int(k)] = v
    return new_score
         
            
def Multi_Mix(
        args, 
        MODEL_CLASSES, 
        labels, 
        pad_token_label_id, 
        num_labels,
        logger=None
    ):
    
    logger.info("Starting Multimix Training")
    logger.info("-"*20)
    
    scores = getNullScore(args)
    scores = checkSinglePrevScores(args.thetas[0], scores)
    logger.info("SCORES : {}".format(json.dumps(scores, indent=4)))
    
    args = export_model_to_experiment_folder(
        args, 
        MODEL_CLASSES, labels, pad_token_label_id,
        num_labels,
        logger
    )
    
    co_distill_log(args, logger)
    theta_0_train_cnt = 0
    theta_1_train_cnt = 0
    theta_2_train_cnt = 0
    for semi_sup_epoch_idx in range(args.semi_sup_start_epoch, args.max_semi_sup_epoch): 
        
        # 0, 1, 2, 3, 4, 5
        # 0 -> src train, src augment
        # 1 -> tgt train
        # 2 -> tgt augment
        # 3 -> tgt train, tgt augment
        # 4 -> src train, src augment, tgt train, tgt augment
        
        logger.info("\n\nSTARTING SEMISUPERVISED EPOCH {}\n\n".format(semi_sup_epoch_idx))
        
        for k in range(len(args.thetas)):
            
            data_tgt = {}    
            data_tgt[k] = []
            
            logger.info("loading k = {} Model".format(k))
            config_k, tokenizer_k, model_k = select_theta(args, MODEL_CLASSES, num_labels, args.thetas[k], logger=logger)
            # check result
            logger.info("Evaluating tgt language(s) dev and test dataset{} with model K = {} ...".format(args.tgt_lang, k))
            dev_scores_k = evaluate(
                            args, 
                            model_k, tokenizer_k, labels, 
                            pad_token_label_id, "dev", 
                            prefix="", langs = args.tgt_lang,
                            logger=logger
                        )    
            test_scores_k = evaluate(
                            args, 
                            model_k, tokenizer_k, labels, 
                            pad_token_label_id, "test", 
                            prefix="", langs = args.tgt_lang,
                            logger=logger
                        )
            
            if check_acceptance(args.aug_desc, semi_sup_epoch_idx, "tgt_self"):
                (dict_key_tgt_k, address_info_tgt_k, indices_tgt_k, gmm_model_tgt_k) = distill(
                                    args, model_k, tokenizer_k, labels, pad_token_label_id, 
                                    lang=args.tgt_lang, mode="train", 
                                    theta_idx=k, semi_sup_epoch_idx=semi_sup_epoch_idx, k=k, 
                                    external_data=None, logger=logger
                                )
                data_tgt[k] = (address_info_tgt_k, indices_tgt_k)
            
            #  Second seciton of the algorithm started.
            for j in range(len(args.thetas)):
                if j == k:
                    continue
                i = None
                for t in range(len(args.thetas)):
                    if t != j and t != k:
                        i = t
                        break 
                
                # All the datasets used for training will be stored in `train_dataset_addresses`
                train_dataset_addresses = []
                
                # Selecting source dataset
                source_train_dataset = None
                for dt in args.train:
                    if dt.split(";")[-1] == args.src_lang:
                        source_train_dataset = dt
                        break
                assert source_train_dataset is not None
                
                # theta[i] params will be trained, based on agreement of j and k
                logger.info("Model seletion params, k : {} , j : {} , i : {}".format(k, j, i))
                
                logger.info("loading j = {} Model".format(j))
                config_j, tokenizer_j, model_j = select_theta(args, MODEL_CLASSES, num_labels, args.thetas[j], logger=logger)

                if check_acceptance(args.aug_desc, semi_sup_epoch_idx, "src"):
                    train_dataset_addresses.append(source_train_dataset)
                    
                # source language data augmentation 
                if check_acceptance(args.aug_desc, semi_sup_epoch_idx, "src_aug"):
                    # Generation is precomputed
                    data_src_aug, data_src_aug_address, data_src_lang = gen_LM(
                            dataset_info_list=args.train, 
                            mode="aug", seed=args.seed, percent=args.train_data_percentage, 
                            lang=args.src_lang, 
                            pkl_list=args.external_data,
                            logger = logger
                        )
                    # data_src_aug_file_name = os.path.join(args.output_dir, os.path.split(data_src_aug_address)[-1]) + ";pkl;" + data_src_lang
                    
                    (dict_key_src_k, address_info_src_k, indices_src_k, gmm_model_src_k) = distill(
                                                args, model_k, tokenizer_k, labels, pad_token_label_id, 
                                                data_src_lang, mode="aug", 
                                                theta_idx=k, semi_sup_epoch_idx=semi_sup_epoch_idx, k=k, 
                                                external_data=[data_src_aug_address], examples=data_src_aug, logger=logger
                                            )
                    
                    (dict_key_src_j, address_info_src_j, indices_src_j, gmm_model_src_j) = distill(
                                                args, model_j, tokenizer_j, labels, pad_token_label_id, 
                                                data_src_lang, mode="aug", 
                                                theta_idx=j, semi_sup_epoch_idx=semi_sup_epoch_idx, k=k, 
                                                external_data=[data_src_aug_address], examples=data_src_aug, logger=logger
                                            )
                    
                    k_aug_src_data = (None, address_info_src_k, indices_src_k, None)
                    j_aug_src_data = (None, address_info_src_j, indices_src_j, None)
                    if args.agreement_param == 1:
                        data_src_aug_distiled, _ = get_intersected_dataset_from_indices(k_aug_src_data, j_aug_src_data, logger=logger)
                    elif args.agreement_param == 2:
                        data_src_aug_distiled, _ = get_unioned_dataset_from_indices(k_aug_src_data, j_aug_src_data, logger=logger)
                    train_dataset_addresses.append( data_src_aug_distiled )
                    
                # Original target train 
                if check_acceptance(args.aug_desc, semi_sup_epoch_idx, "tgt_self"):
                    (dict_key_tgt_j, address_info_tgt_j, indices_tgt_j, gmm_model_tgt_j) = distill(
                                        args, model_j, tokenizer_j, labels, pad_token_label_id, 
                                        lang=args.tgt_lang, mode="train", 
                                        theta_idx=j, semi_sup_epoch_idx=semi_sup_epoch_idx, k=k, 
                                        external_data=None, logger=logger
                                    )
                    data_tgt[j] = (address_info_tgt_j, indices_tgt_j)
                    
                    k_tgt_data = (None, address_info_tgt_k, indices_tgt_k, None)
                    j_tgt_data = (None, address_info_tgt_j, indices_tgt_j, None)
                    if args.agreement_param == 1:
                        data_tgt_distiled, _ = get_intersected_dataset_from_indices(k_tgt_data, j_tgt_data, logger=logger)
                    elif args.agreement_param == 2:
                        data_tgt_distiled, _ = get_unioned_dataset_from_indices(k_tgt_data, j_tgt_data, logger=logger)
                    train_dataset_addresses.append( data_tgt_distiled )
                    
                # target language data augmentation 
                if check_acceptance(args.aug_desc, semi_sup_epoch_idx, "tgt_aug"):
                    # Generation is precomputed
                    data_tgt_aug, data_tgt_aug_address, data_tgt_lang = gen_LM(dataset_info_list=args.train, 
                                                                           mode="aug", seed=args.seed, percent=args.train_data_percentage, 
                                                                           lang=args.tgt_lang, 
                                                                           pkl_list=args.external_data, logger=logger)
                    # data_tgt_aug_file_name = os.path.join(args.output_dir, os.path.split(data_tgt_aug_address)[-1]) + ";pkl;" + data_tgt_lang
                    
                    (dict_key_tgt_aug_k, address_info_tgt_aug_k, indices_tgt_aug_k, gmm_model_tgt_aug_k) = distill(
                                                args, model_k, tokenizer_k, labels, pad_token_label_id, 
                                                lang=data_tgt_lang, mode="aug", 
                                                theta_idx=k, semi_sup_epoch_idx=semi_sup_epoch_idx, k=k, 
                                                external_data=[data_tgt_aug_address], examples=data_tgt_aug, logger=logger
                                            )
                    
                    (dict_key_tgt_aug_j, address_info_tgt_aug_j, indices_tgt_aug_j, gmm_model_tgt_aug_j) = distill(
                                                args, model_j, tokenizer_j, labels, pad_token_label_id, 
                                                lang=data_tgt_lang, mode="aug", 
                                                theta_idx=j, semi_sup_epoch_idx=semi_sup_epoch_idx, k=k, 
                                                external_data=[data_tgt_aug_address], examples=data_tgt_aug, logger=logger
                                            )
                    
                    k_aug_tgt_data = (None, address_info_tgt_aug_k, indices_tgt_aug_k, None)
                    j_aug_tgt_data = (None, address_info_tgt_aug_j, indices_tgt_aug_j, None)
                    if args.agreement_param == 1:
                        data_tgt_aug_distiled, _ = get_intersected_dataset_from_indices(k_aug_tgt_data, j_aug_tgt_data, logger=logger)
                    elif args.agreement_param == 2:
                        data_tgt_aug_distiled, _ = get_unioned_dataset_from_indices(k_aug_tgt_data, j_aug_tgt_data, logger=logger)
                    train_dataset_addresses.append( data_tgt_aug_distiled )

                
                logger.info("training datasets are :")
                for idx, dataset in enumerate(train_dataset_addresses):
                    logger.info("{}. {} , total number of sample {}".format(
                            idx, dataset, 
                            len(read_from_path(dataset.split(";")[0], encoding=dataset.split(";")[1]))
                        )
                    )
                
                config_i, tokenizer_i, model_i = select_theta(
                            args, MODEL_CLASSES, num_labels,
                            args.thetas[i], retrain=0, logger=logger
                        )
                
                temp = args.overwrite_cache
                args.overwrite_cache = True
                train_dataset, guids = load_and_cache_examples(
                            args, tokenizer_i, labels, pad_token_label_id, 
                            mode="aug", langs=args.aug_lang, external_data=train_dataset_addresses, logger=logger
                        )
                args.overwrite_cache = temp
                
                tot_sample = 0
                for _, v in train_dataset.items():
                    tot_sample += len(v)
                logger.info("Total number of sample {}".format(tot_sample))
                if tot_sample > 0:
                    args.max_steps = int((tot_sample * args.num_train_epochs)//(args.per_gpu_train_batch_size * args.gradient_accumulation_steps))
                    args.warmup_steps = int((args.max_steps*10)//100)
                    logger.info("Total number of steps {}".format(args.max_steps))
                    logger.info("Number of warmup steps {}".format(args.warmup_steps))
                    best_dev_scores, test_scores_in_best_src_dev = scores[i]
                    
                    save_address_i = save_model_checkpoint(
                        args, args.output_dir, "sse_" + str(semi_sup_epoch_idx)+".i_{}.j_{}.k_{}_i".format(i,j,k), 
                        model_i, checkpoint="checkpoint",
                        logger=logger
                    )
                    save_address_j = save_model_checkpoint(
                        args, args.output_dir, "sse_" + str(semi_sup_epoch_idx)+".i_{}.j_{}.k_{}_j".format(i,j,k), 
                        model_j, checkpoint="checkpoint",
                        logger=logger
                    )
                    save_address_j = save_model_checkpoint(
                        args, args.output_dir, "sse_" + str(semi_sup_epoch_idx)+".i_{}.j_{}.k_{}_k".format(i,j,k), 
                        model_k, checkpoint="checkpoint",
                        logger=logger
                    )
                    model_j = model_j.to(torch.device("cpu"))
                    model_k = model_k.to(torch.device("cpu"))
                    global_step, tr_loss, isUpdated,  best_dev_scores, test_scores_in_best_src_dev = training_loop(
                                    args, train_dataset, 
                                    model_i, tokenizer_i, labels, pad_token_label_id, 
                                    logger=logger, 
                                    prev_best_dev_scores = best_dev_scores, 
                                    prev_test_scores_in_best_src_dev=test_scores_in_best_src_dev, 
                                    tf_board_header= "sse_{}.i_{}.j_{}.k_{}".format(semi_sup_epoch_idx, i,j,k)
                                )
                    
                    scores[i] = (best_dev_scores, test_scores_in_best_src_dev)
                    logger.info("SEMI_SUP_EPOCH {}, K - {}, J - {}, best_dev_scores - {}".format(semi_sup_epoch_idx, k, j, best_dev_scores))
                    logger.info("SEMI_SUP_EPOCH {}, K - {}, J - {}, test_scores_in_best_src_dev - {}".format(semi_sup_epoch_idx, k, j, test_scores_in_best_src_dev))
                    
                    if isUpdated:
                        update_theta_i_model(args, i, j, k, semi_sup_epoch_idx, logger=logger)
                    
                    logger.info("SCORES : {}".format(json.dumps(scores, indent=4)))
                    with open(os.path.join(args.output_dir, 'scores.json'), 'w') as outfile:
                        outfile.write(json.dumps(scores, indent=4))
                    
                    # tensor board log update
                    if i == 0:
                        theta_0_train_cnt += 1
                        theta_cnt_no = theta_0_train_cnt
                    elif i == 1:
                        theta_1_train_cnt += 1
                        theta_cnt_no = theta_1_train_cnt
                    elif i == 2:
                        theta_2_train_cnt += 1
                        theta_cnt_no = theta_2_train_cnt
                    tb_path = os.path.join(args.output_dir, "tf_board")
                    if not os.path.exists(tb_path):
                        os.makedirs(tb_path)
                    tb_writer = SummaryWriter(tb_path)
                    
                    for score_d_k, score_v in best_dev_scores.items():
                        tb_writer.add_scalar("sse_dev_th-{}_{}-F1".format(i, score_d_k), score_v, theta_cnt_no)
                    for score_d_k, score_v in test_scores_in_best_src_dev.items():
                        tb_writer.add_scalar("sse_test_th-{}_{}-F1".format(i, score_d_k), score_v, theta_cnt_no)
                    
                    model_j = model_j.to(args.device)
                    model_k = model_k.to(args.device)
                    
                    
def update_theta_i_model(args, i, j, k, semi_sup_epoch_idx, logger):
    name = args.dev_lang.replace(".", "_").replace(";", "_")
    best_dev_model = os.path.join(args.output_dir, "best_dev_model.{}".format(name))

    cmd = "rm -rf {}".format(args.thetas[i])
    logger.info("Removing model for theta {}".format(args.thetas[i]))
    logger.info("Executing command :{}: ...".format(cmd))
    subprocess.check_output(cmd, shell=True)
    
    cmd = "mv {} {}".format(best_dev_model, args.thetas[i])
    logger.info("Renaming {} to {} folder".format(best_dev_model, args.thetas[i]))
    logger.info("Executing command :{}: ...".format(cmd))
    subprocess.check_output(cmd, shell=True)
    new_path = os.path.join(args.output_dir, "checkpoint.sse_" + str(semi_sup_epoch_idx)+".i_{}.j_{}.k_{}_i_post_train_best".format(i,j,k))
    cmd = "cp -r {} {}".format(args.thetas[i], new_path)
    logger.info("Executing command :{}: ...".format(cmd))
    subprocess.check_output(cmd, shell=True)
    

def update_single_theta_model(args, semi_sup_epoch_idx, logger):
    cmd = "rm -rf {}".format(args.thetas[0])
    logger.info("Removing model for theta {}".format(args.thetas[0]))
    logger.info("Executing command :{}: ...".format(cmd))
    subprocess.check_output(cmd, shell=True)
    name = args.dev_lang.replace(".", "_").replace(";", "_")
    best_dev_model = os.path.join(args.output_dir, "best_dev_model.{}".format(name))
    cmd = "mv {} {}".format(best_dev_model, args.thetas[0])
    logger.info("Renaming {} to {} folder".format(best_dev_model, args.thetas[0]))
    logger.info("Executing command :{}: ...".format(cmd))
    subprocess.check_output(cmd, shell=True)
    new_path = os.path.join(args.output_dir, "checkpoint.sse_" + str(semi_sup_epoch_idx)+".k_{}.post_train_best".format(0))
    cmd = "cp -r {} {}".format(args.thetas[0], new_path)
    logger.info("Executing command :{}: ...".format(cmd))
    subprocess.check_output(cmd, shell=True)
                


def Multi_Mix_Single_Model_Multi_Head(
        args, 
        MODEL_CLASSES, 
        labels, 
        pad_token_label_id, 
        num_labels,
        logger=None
    ):
    
    logger.info("Starting Multimix Training")
    logger.info("-"*20)

    scores = getNullScore(args)
    scores = checkSinglePrevScores(args.thetas[0], scores)
    logger.info("SCORES : {}".format(json.dumps(scores, indent=4)))
    
    args = export_model_to_experiment_folder(
        args, 
        MODEL_CLASSES, labels, pad_token_label_id,
        num_labels,
        logger
    )

    co_distill_log(args, logger)
    scores = getMultiHeadNullScore(args)

    for semi_sup_epoch_idx in range(args.semi_sup_start_epoch, args.max_semi_sup_epoch): 
        
        logger.info("\n\nSTARTING SEMISUPERVISED EPOCH {}\n\n".format(semi_sup_epoch_idx))
        config, tokenizer, model = select_theta(args, MODEL_CLASSES, num_labels, args.thetas[0], logger=logger)
        logger.info("Evaluating tgt language(s) dev and test dataset{} with model K = {} ...".format(args.tgt_lang, 0))

        for head_idx in range(args.num_of_heads):
            dev_scores_k = evaluate(
                            args, 
                            model, tokenizer, labels, 
                            pad_token_label_id, "dev", 
                            prefix="", langs = args.tgt_lang,
                            logger=logger,
                            head_idx=head_idx
                        )    
            test_scores_k = evaluate(
                            args, 
                            model, tokenizer, labels, 
                            pad_token_label_id, "test", 
                            prefix="", langs = args.tgt_lang,
                            logger=logger, 
                            head_idx=head_idx
                        )
        # All the datasets used for training will be stored in `train_dataset_addresses`
        train_dataset_addresses = []

        # Selecting source dataset
        source_train_dataset = None
        for dt in args.train:
            if dt.split(";")[-1] == args.src_lang:
                source_train_dataset = dt
                break
        assert source_train_dataset is not None

        if check_acceptance(args.aug_desc, semi_sup_epoch_idx, "src"):
            train_dataset_addresses.append(source_train_dataset)
        
        if check_acceptance(args.aug_desc, semi_sup_epoch_idx, "src_aug"):
            # Generation is precomputed
            data_src_aug, data_src_aug_address, data_src_lang = gen_LM(
                    dataset_info_list=args.train, 
                    mode="aug", seed=args.seed, percent=args.train_data_percentage, 
                    lang=args.src_lang, 
                    pkl_list=args.external_data,
                    logger = logger
                )
            # data_src_aug_file_name = os.path.join(args.output_dir, os.path.split(data_src_aug_address)[-1]) + ";pkl;" + data_src_lang

            (dict_key_src_h_0, address_info_src_h_0, indices_src_h_0, gmm_model_src_h_0) = distill(
                                                            args, model, tokenizer, labels, pad_token_label_id, 
                                                            data_src_lang, mode="aug", 
                                                            theta_idx=0, semi_sup_epoch_idx=semi_sup_epoch_idx, k=0, 
                                                            external_data=[data_src_aug_address], examples=data_src_aug, head_idx=0, logger=logger
                                                        )

            (dict_key_src_h_1, address_info_src_h_1, indices_src_h_1, gmm_model_src_h_1) = distill(
                                                            args, model, tokenizer, labels, pad_token_label_id, 
                                                            data_src_lang, mode="aug", 
                                                            theta_idx=0, semi_sup_epoch_idx=semi_sup_epoch_idx, k=0, 
                                                            external_data=[data_src_aug_address], examples=data_src_aug, head_idx=1, logger=logger
                                                        )

            (dict_key_src_h_2, address_info_src_h_2, indices_src_h_2, gmm_model_src_h_2) = distill(
                                                            args, model, tokenizer, labels, pad_token_label_id, 
                                                            data_src_lang, mode="aug", 
                                                            theta_idx=0, semi_sup_epoch_idx=semi_sup_epoch_idx, k=0, 
                                                            external_data=[data_src_aug_address], examples=data_src_aug, head_idx=2, logger=logger
                                                        )
            
            h_0_aug_src_data = (None, address_info_src_h_0, indices_src_h_0, None)
            h_1_aug_src_data = (None, address_info_src_h_1, indices_src_h_1, None)
            h_2_aug_src_data = (None, address_info_src_h_2, indices_src_h_2, None)
            h_0_h_1_data_src_aug_distiled, h_0_h_1_indices = get_intersected_dataset_from_indices(h_0_aug_src_data, h_1_aug_src_data, logger=logger)
            h_0_h_1_aug_src_data = (None, h_0_h_1_data_src_aug_distiled, h_0_h_1_indices, None)
            data_src_aug_distiled, _ = get_intersected_dataset_from_indices(h_0_h_1_aug_src_data, h_2_aug_src_data, logger=logger)
            train_dataset_addresses.append( data_src_aug_distiled )
        
        if check_acceptance(args.aug_desc, semi_sup_epoch_idx, "tgt_self"):   
            (dict_key_tgt_h_0, address_info_tgt_h_0, indices_tgt_h_0, gmm_model_tgt_h_0) = distill(
                                                        args, model, tokenizer, labels, pad_token_label_id, 
                                                        args.tgt_lang, mode="train", 
                                                        theta_idx=0, semi_sup_epoch_idx=semi_sup_epoch_idx, k=0, 
                                                        external_data=None, examples=None, head_idx=0, logger=logger
                                                    )
            (dict_key_tgt_h_1, address_info_tgt_h_1, indices_tgt_h_1, gmm_model_tgt_h_1) = distill(
                                                        args, model, tokenizer, labels, pad_token_label_id, 
                                                        args.tgt_lang, mode="train", 
                                                        theta_idx=0, semi_sup_epoch_idx=semi_sup_epoch_idx, k=0, 
                                                        external_data=None, examples=None, head_idx=1, logger=logger
                                                    )
            (dict_key_tgt_h_2, address_info_tgt_h_2, indices_tgt_h_2, gmm_model_tgt_h_2) = distill(
                                                        args, model, tokenizer, labels, pad_token_label_id, 
                                                        args.tgt_lang, mode="train", 
                                                        theta_idx=0, semi_sup_epoch_idx=semi_sup_epoch_idx, k=0, 
                                                        external_data=None, examples=None, head_idx=2, logger=logger
                                                    )
            

            h_0_tgt_self_data = (None, address_info_tgt_h_0, indices_tgt_h_0, None)
            h_1_tgt_self_data = (None, address_info_tgt_h_1, indices_tgt_h_1, None)
            h_2_tgt_self_data = (None, address_info_tgt_h_2, indices_tgt_h_2, None)
            h_0_h_1_data_tgt_self_distiled, h_0_h_1_indices = get_intersected_dataset_from_indices(h_0_tgt_self_data, h_1_tgt_self_data, logger=logger)
            h_0_h_1_tgt_self_data = (None, h_0_h_1_data_tgt_self_distiled, h_0_h_1_indices, None)
            data_tgt_self_distiled, _ = get_intersected_dataset_from_indices(h_0_h_1_tgt_self_data, h_2_tgt_self_data, logger=logger)
            train_dataset_addresses.append( data_tgt_self_distiled )
        
        if check_acceptance(args.aug_desc, semi_sup_epoch_idx, "tgt_aug"):
            # Generation is precomputed
            data_tgt_aug, data_tgt_aug_address, data_tgt_lang = gen_LM(dataset_info_list=args.train, 
                                                                mode="aug", seed=args.seed, percent=args.train_data_percentage, 
                                                                lang=args.tgt_lang, 
                                                                pkl_list=args.external_data, logger=logger)
            # data_tgt_aug_file_name = os.path.join(args.output_dir, os.path.split(data_tgt_aug_address)[-1]) + ";pkl;" + data_tgt_lang


            (dict_key_tgt_aug_h_0, address_info_tgt_aug_h_0, indices_tgt_aug_h_0, gmm_model_tgt_aug_h_0) = distill(
                                                            args, model, tokenizer, labels, pad_token_label_id, 
                                                            args.tgt_lang, mode="aug", 
                                                            theta_idx=0, semi_sup_epoch_idx=semi_sup_epoch_idx, k=0, 
                                                            external_data=[data_tgt_aug_address], examples=data_tgt_aug, head_idx=0, logger=logger
                                                        )

            (dict_key_tgt_aug_h_1, address_info_tgt_aug_h_1, indices_tgt_aug_h_1, gmm_model_tgt_aug_h_1) = distill(
                                                            args, model, tokenizer, labels, pad_token_label_id, 
                                                            args.tgt_lang, mode="aug", 
                                                            theta_idx=0, semi_sup_epoch_idx=semi_sup_epoch_idx, k=0, 
                                                            external_data=[data_tgt_aug_address], examples=data_tgt_aug, head_idx=1, logger=logger
                                                        )

            (dict_key_tgt_aug_h_2, address_info_tgt_aug_h_2, indices_tgt_aug_h_2, gmm_model_tgt_aug_h_2) = distill(
                                                            args, model, tokenizer, labels, pad_token_label_id, 
                                                            args.tgt_lang, mode="aug", 
                                                            theta_idx=0, semi_sup_epoch_idx=semi_sup_epoch_idx, k=0, 
                                                            external_data=[data_tgt_aug_address], examples=data_tgt_aug, head_idx=2, logger=logger
                                                        )


            h_0_aug_tgt_data = (None, address_info_tgt_h_0, indices_tgt_h_0, None)
            h_1_aug_tgt_data = (None, address_info_tgt_h_1, indices_tgt_h_1, None)
            h_2_aug_tgt_data = (None, address_info_tgt_h_2, indices_tgt_h_2, None)
            h_0_h_1_data_tgt_aug_distiled, h_0_h_1_indices = get_intersected_dataset_from_indices(h_0_aug_tgt_data, h_1_aug_tgt_data, logger=logger)
            h_0_h_1_tgt_aug_data = (None, h_0_h_1_data_tgt_aug_distiled, h_0_h_1_indices, None)
            data_tgt_aug_distiled, _ = get_intersected_dataset_from_indices(h_0_h_1_tgt_aug_data, h_2_aug_tgt_data, logger=logger)
            train_dataset_addresses.append( data_tgt_aug_distiled )
        
        logger.info("training datasets are :")
        for idx, dataset in enumerate(train_dataset_addresses):
            logger.info("{}. {} , total number of sample {}".format(
                    idx, dataset, 
                    len(read_from_path(dataset.split(";")[0], encoding=dataset.split(";")[1]))
                )
            )
        temp = args.overwrite_cache
        args.overwrite_cache = True
        train_dataset, guids = load_and_cache_examples(
                    args, tokenizer, labels, pad_token_label_id, 
                    mode="aug", langs=args.aug_lang, external_data=train_dataset_addresses, logger=logger
                )
        args.overwrite_cache = temp
        
        tot_sample = 0
        for _, v in train_dataset.items():
            tot_sample += len(v)
        logger.info("Total number of sample {}".format(tot_sample))
        if tot_sample > 0:
            args.max_steps = int((tot_sample * args.num_train_epochs)//(args.per_gpu_train_batch_size * args.gradient_accumulation_steps))
            args.warmup_steps = int((args.max_steps*10)//100)
            logger.info("Total number of steps {}".format(args.max_steps))
            logger.info("Number of warmup steps {}".format(args.warmup_steps))

            multi_head_best_dev_scores, multi_head_test_scores_in_best_src_dev = scores[semi_sup_epoch_idx-1]
            save_address = save_model_checkpoint(
                args, args.output_dir, "sse_" + str(semi_sup_epoch_idx)+".multihead",
                model, checkpoint="checkpoint",
                logger=logger
            )
            (global_step, 
            tr_loss, 
            isUpdated,  
            multi_head_best_dev_scores, 
            multi_head_test_scores_in_best_src_dev) = multi_head_training_loop(
                            args, train_dataset, 
                            model, tokenizer, labels, pad_token_label_id, 
                            logger=logger, 
                            multi_head_best_dev_scores = multi_head_best_dev_scores, 
                            multi_head_test_scores_in_best_src_dev=multi_head_test_scores_in_best_src_dev, 
                            tf_board_header= "sse_{}".format(semi_sup_epoch_idx)
                        )
            save_address = save_model_checkpoint(
                args, args.output_dir, "sse_" + str(semi_sup_epoch_idx)+".multihead",
                model, checkpoint="checkpoint",
                logger=logger
            )
            scores[semi_sup_epoch_idx] = (multi_head_best_dev_scores, multi_head_test_scores_in_best_src_dev)
            logger.info("SEMI_SUP_EPOCH {}, best_dev_scores - {}".format(semi_sup_epoch_idx, multi_head_best_dev_scores))
            logger.info("SEMI_SUP_EPOCH {}, test_scores_in_best_src_dev - {}".format(semi_sup_epoch_idx, multi_head_test_scores_in_best_src_dev))

            if isUpdated:
                update_single_theta_model(args, semi_sup_epoch_idx, logger)

            logger.info("SCORES : {}".format(json.dumps(scores, indent=4)))
            with open(os.path.join(args.output_dir, 'scores.json'), 'w') as outfile:
                outfile.write(json.dumps(scores, indent=4))

            tb_path = os.path.join(args.output_dir, "tf_board")
            if not os.path.exists(tb_path):
                os.makedirs(tb_path)
            tb_writer = SummaryWriter(tb_path)

            for score_d_k, score_v in multi_head_best_dev_scores.items():
                tb_writer.add_scalar("sse_dev_th-{}_{}-F1-avg".format(0, score_d_k), score_v[0], semi_sup_epoch_idx)
            for score_d_k, score_v in multi_head_test_scores_in_best_src_dev.items():
                tb_writer.add_scalar("sse_test_th-{}_{}-F1-avg".format(0, score_d_k), score_v[0], semi_sup_epoch_idx)