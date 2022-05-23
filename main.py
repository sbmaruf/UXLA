from __future__ import absolute_import, division, print_function

import os
import glob
import json
import torch
import pickle
import random
import logging
import subprocess
import numpy as np
import jsbeautifier

from args import load_args
from load_examples import load_and_cache_examples
from train import evaluate, export_logit, inference, ensamble_eval, single_model_eval
from train import training_loop, multi_head_training_loop
from collections import OrderedDict
from utils_ner import (convert_examples_to_features, 
                       get_labels, 
                       read_examples_from_file, 
                       read_from_path, 
                       write_data, 
                       manual_check_nl,
                       select_and_write_data,
                       select_and_write_logits)
from transformers import (WEIGHTS_NAME, 
                          BertConfig, 
                          BertForTokenClassification, 
                          BertTokenizer,
                          XLMRobertaConfig, 
                          XLMRobertaForTokenClassification, 
                          XLMRobertaTokenizer)
from torch.nn import CrossEntropyLoss
from model import load_model, Context_NER_BERT, Context_NER_XLMR, save_model_checkpoint
from unsup_utils import select_samples_with_GMM
from logger import create_logger
from semi_sup_train import Multi_Mix_Single_Model_Multi_Head, Multi_Mix_Single_Model, pseudo_self_training, classical_self_training, Multi_Mix, partial_single_self_training

opts = jsbeautifier.default_options()
logger = logging.getLogger(__name__)


MODEL_CLASSES = {
    "bert": (BertConfig, Context_NER_BERT, BertTokenizer),
    "xlmroberta": (XLMRobertaConfig, Context_NER_XLMR, XLMRobertaTokenizer)
}

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

def dist_training(args):
    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device
    return args


def save_running_code(src_folder, tgt_folder, exclude_folders=[], logger=None):
    code_backup = os.path.join(tgt_folder, 'code_backup')
    os.makedirs(code_backup, exist_ok=True)
    __str_exclude = ""
    for folders in exclude_folders:
        __str_exclude = __str_exclude + " --exclude " + str(folders)
    _command = "rsync -a -progress {} {} {}".format(src_folder, code_backup, __str_exclude)
    logger.info("Executing \'"+_command)
    subprocess.check_output(_command, shell=True)


def backup_codes_to_project_folder(params, logger):
    exclude_folders = []
    try:
        with open(".gitignore", "r") as filePtr:
            for line in filePtr:
                line=line.strip()
                if line == "" or line.startswith("#"):
                    continue
                exclude_folders.append(line.strip())
    except:
        logger.warning("No .gitignore file found. \
            Please make sure the all large files are not copied to the experiment folder.")

    backup_codes_path = os.path.join(params.output_dir, "code")
    os.makedirs(backup_codes_path, exist_ok=True)
    save_running_code(
        src_folder=os.getcwd(), 
        tgt_folder=backup_codes_path, 
        exclude_folders=exclude_folders,
        logger=logger
    )


def main():
    
    args = load_args()
    
    logger = create_logger(os.path.join(args.output_dir, "cross-lingual-ner.log"))
    logger.info("{}".format(jsbeautifier.beautify(json.dumps(args.__dict__), opts)))
    logger.info("CUDA : {}".format(torch.cuda.is_available()))
    args = dist_training(args)
    set_seed(args)

    # backup_codes_to_project_folder(args, logger)

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    # Prepare CONLL-2003 task
    labels = get_labels(args.label)
    num_labels = len(labels)
    # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
    pad_token_label_id = CrossEntropyLoss().ignore_index

    if args.do_ensamble_eval:
        ensamble_eval(args, labels, pad_token_label_id, MODEL_CLASSES, logger=logger)

    if args.do_eval:
        single_model_eval(args, labels, pad_token_label_id, MODEL_CLASSES, logger=logger)
        # you need to process dutch data a bit differently. Here is the code. 
        # for k, (results, prediction) in test_scores.items():
        #     if k == "data/nl/nl.testb.iob2;utf-8;nl":
        #         predictions = test_scores[k][1]
        #         manual_check_nl(args.output_dir, predictions, logger)
        #     else:
        #         predictions = test_scores[k][1]
        #         write_data(args.output_dir, k, predictions, logger)


    if args.do_train:
        config, tokenizer, model = load_model(
            args.model_type, MODEL_CLASSES, 
            args.model_name_or_path, args.config_name, args.tokenizer_name,
            num_labels, args.cache_dir, args.do_lower_case, args.device, dropout=args.dropout,
            num_of_heads=args.num_of_heads
        )
        train_dataset, _ = load_and_cache_examples(
                            args, tokenizer, labels, pad_token_label_id, 
                            mode="train", langs=args.src_lang, logger=logger
                        )
        tot_sample = 0
        for _, v in train_dataset.items():
            tot_sample += len(v)    
        if args.max_steps == -1:
            args.max_steps = int((tot_sample * args.num_train_epochs)//(args.per_gpu_train_batch_size * args.gradient_accumulation_steps))
        if args.warmup_steps == -1:
            args.warmup_steps = (args.max_steps*10)//100
        if args.num_of_heads == 1:
            global_step, tr_loss, _, _, _ = training_loop(
                                args, train_dataset, 
                                model, tokenizer, labels, pad_token_label_id, 
                                logger=logger
                            )
        else:
            global_step, tr_loss, _, _, _ = multi_head_training_loop(
                            args, train_dataset, 
                            model, tokenizer, labels, pad_token_label_id, 
                            logger=logger
                        )
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
       
        
    if args.semi_sup_type == "multi-mix":
        if len(args.thetas) == 3:            
            Multi_Mix(
                args, 
                MODEL_CLASSES,
                labels, 
                pad_token_label_id, 
                num_labels,
                logger=logger
            )


if __name__ == "__main__":
    main()
