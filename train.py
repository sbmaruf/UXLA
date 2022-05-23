import os
import sys
import json
import copy
import math
from tqdm import tqdm
import torch
import random
import pickle
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
from seqeval.metrics import precision_score, recall_score, f1_score
from collections import OrderedDict
from tensorboardX import SummaryWriter
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from load_examples import load_and_cache_examples
from sklearn import mixture
from model import save_model_checkpoint, load_model

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def multinomial_prob(dataset_len, alpha=.5):
    tot_number_of_sent_in_all_lang = 0
    prob = OrderedDict()
    for k, v in dataset_len.items():
        tot_number_of_sent_in_all_lang += v
    for k, v in dataset_len.items():
        neu = v
        den = tot_number_of_sent_in_all_lang
        p = neu/den 
        prob[k] = p
        
    q = OrderedDict()
    q_den = 0.0
    for k, v in prob.items():
        q_den += (v**alpha)
    sum_ = 0.0
    for k, v in prob.items():
        q[k] =  (v**alpha)/q_den
        sum_ += q[k]
    assert math.fabs(1-sum_) < 1e-2
    return q


def iterator_selection_prob(alpha, train_dataset, logger=None):
    dataset_len = OrderedDict()
    for k, v in train_dataset.items():
        dataset_len[k] = len(v)
    for k, v in dataset_len.items():
        logger.info("Total Number of sentences in {} : {}".format(k, v))
    prob = multinomial_prob(dataset_len, alpha=alpha)
    logger.info("Language iterator selection probability.")
    ret_prob_index, ret_prob_list  = [], []
    for k,v in prob.items():
        ret_prob_index.append(k)
        ret_prob_list.append(v)
    for k, v in zip(ret_prob_index, ret_prob_list):
        logger.info("{} : {}".format(k, v))
    return dataset_len, ret_prob_index, ret_prob_list


def softmax(X, theta = 1.0, axis = None):
    """
    Compute the softmax of each element along an axis of X.

    Parameters
    ----------
    X: ND-Array. Probably should be floats.
    theta (optional): float parameter, used as a multiplier
        prior to exponentiation. Default = 1.0
    axis (optional): axis to compute values along. Default is the
        first non-singleton axis.

    Returns an array the same size as X. The result will sum to 1
    along the specified axis.
    """

    # make X at least 2d
    y = np.atleast_2d(X)

    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    # multiply y against the theta parameter,
    y = y * float(theta)

    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis = axis), axis)

    # exponentiate y
    y = np.exp(y)

    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis = axis), axis)

    # finally: divide elementwise
    p = y / ax_sum

    # flatten if X was 1D
    if len(X.shape) == 1: p = p.flatten()

    return p



def set_fp16_training(args, model, optimizer):
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
    return model, optimizer
    

def get_optimizer(args, model):
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.max_steps)
    model, optimizer = set_fp16_training(args, model, optimizer)
    return model, optimizer, scheduler


def save_results(args, best_dev_scores, test_scores_in_best_src_dev, logger):
    logger.info(json.dumps(best_dev_scores, indent=4))
    logger.info(json.dumps(test_scores_in_best_src_dev, indent=4))

    with open(os.path.join(args.output_dir, 'best_dev_scores.json'), 'w') as outfile:
        json.dump(best_dev_scores, outfile)

    with open(os.path.join(args.output_dir, 'test_scores_in_best_src_dev.json'), 'w') as outfile:
        json.dump(test_scores_in_best_src_dev, outfile)


# def save_model_checkpoint(args, key, model, logger):
#     key = key.replace(".", "_").replace(";", "_")
#     output_dir = os.path.join(args.output_dir, "best_dev_model_{}".format(key))
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#     model_to_save = model.module if hasattr(model, "module") else model 
#     model_to_save.save_pretrained(output_dir)
#     torch.save(args, os.path.join(output_dir, "training_args.bin"))
#     logger.info("Saving model checkpoint to {}".format(output_dir))


def training_loop(
    args, 
    train_dataset, model, tokenizer, 
    labels, pad_token_label_id, 
    logger=None, 
    prev_best_dev_scores = None, 
    prev_test_scores_in_best_src_dev=None, 
    prev_best_test_scores=None,
    tf_board_header="single"
):
    best_dev_scores = {file_name:0 for file_name in args.dev} if prev_best_dev_scores is None else prev_best_dev_scores
    test_scores_in_best_src_dev = {} if prev_test_scores_in_best_src_dev is None else prev_test_scores_in_best_src_dev
    best_test_scores = {file_name:0 for file_name in args.dev} if prev_best_test_scores is None else prev_best_test_scores
    
    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    
    if args.local_rank in [-1, 0]:
        tb_path = os.path.join(args.output_dir, "tf_board")
        if not os.path.exists(tb_path):
            os.makedirs(tb_path)
        tb_writer = SummaryWriter(tb_path)
    
    logger.info("Evaluate before starting the training loop ...")
    logger.info("-"*20)
    result_prediction = evaluate(
                            args, model, tokenizer, 
                            labels, pad_token_label_id, 
                            "dev", langs = args.dev_lang, 
                            logger=logger
                        )
    for key, (result, prediction) in result_prediction.items():
        dataset_f1_score = result["f1"]
        best_dev_scores[key] = dataset_f1_score
        tb_writer.add_scalar("{}_best_dev_{}_F1".format(tf_board_header, key), dataset_f1_score, global_step)
        tb_writer.add_scalar("{}_eval_{}_F1".format(tf_board_header, key), dataset_f1_score, global_step)                            
        test_results_prediction = evaluate(
                                    args, model, tokenizer, 
                                    labels, pad_token_label_id, 
                                    "test", langs = args.tgt_lang, 
                                    logger=logger
                                )
        for key1, (result, prediction) in test_results_prediction.items():
            rel_key = key+"_"+key1
            tb_writer.add_scalar("{}_test_{}_F1".format(tf_board_header, rel_key), result["f1"], global_step)
            test_scores_in_best_src_dev[ rel_key ] = result["f1"]
            tb_writer.add_scalar("{}_test_on_best_dev_{}_F1".format(tf_board_header, rel_key), result["f1"], global_step)


    dataset_len, lang_prob_index, lang_prob = iterator_selection_prob(args.lang_alpha, train_dataset, logger=logger)
    train_data_loader = []
    for k in lang_prob_index:
        if k in train_dataset:
            dataset = train_dataset[k]
            train_sampler = RandomSampler(dataset) if args.local_rank == -1 else DistributedSampler(dataset)
            data_loader = DataLoader(dataset, sampler=train_sampler, batch_size=args.per_gpu_train_batch_size)
            train_data_loader.append((k, data_loader))

    model, optimizer, scheduler = get_optimizer(args, model)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num of batch = {}".format(dataset_len))
    logger.info("  Instantaneous batch size per GPU = {}".format(args.per_gpu_train_batch_size))
    logger.info("  Gradient Accumulation steps = {}".format(args.gradient_accumulation_steps))
    logger.info("  Effective train batch size (w. parallel, distributed & accumulation) = {}".format(
                        args.per_gpu_train_batch_size * args.gradient_accumulation_steps))
    logger.info("  Total optimization steps = {}".format(args.max_steps))


    train_iterators = []
    for i in range(len(train_data_loader)):
        assert train_data_loader[i][0] == lang_prob_index[i]
        train_iterators.append(iter(train_data_loader[i][1]))
    tot_num_of_iterator = len(train_iterators)


    # set_seed(args)    
    
    num_of_batch_trained = [ 0 for i in range(tot_num_of_iterator) ]
    isUpdated = 0
    for step in range(args.max_steps*args.gradient_accumulation_steps):
        
        model.train()
        
        iterator_id = np.random.choice(range(tot_num_of_iterator), p=lang_prob)
        try:
            batch = train_iterators[iterator_id].__next__()
        except StopIteration:
            train_iterators[iterator_id] = iter(train_data_loader[iterator_id][1])
            batch = train_iterators[iterator_id].__next__()
        num_of_batch_trained[ iterator_id ] += 1
        
        batch = tuple(t.to(args.device) for t in batch)
        inputs = {"input_ids": batch[0],
                    "attention_mask": batch[1],
                    "labels": batch[3],
                    "penalty": args.penalty}
        if args.model_type != "distilbert":
            # XLM and RoBERTa don't use segment_ids
            inputs["token_type_ids"] = batch[2] if args.model_type in ["bert", "xlnet"] else None  

        outputs, per_token_loss = model(**inputs)
        loss = outputs[0]
        
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps

        loss.backward()
        tr_loss += loss.item()

        if (step + 1) % args.gradient_accumulation_steps == 0:
            if args.fp16:
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            
            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            model.zero_grad()
            global_step += 1
            if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                if args.alpha_schedule == "random":
                    alpha = np.random.random_sample()
                    if alpha == 0.0:
                        alpha = .1
                    elif alpha == 1.0:
                        alhpa = .9
                    dataset_len, lang_prob_index, lang_prob = iterator_selection_prob(alpha, train_dataset, logger=logger)
                
                current_loss = (tr_loss - logging_loss) / args.logging_steps
                current_lr_rate = scheduler.get_lr()[0]
                tb_writer.add_scalar("{}_lr".format(tf_board_header), current_lr_rate, global_step)
                tb_writer.add_scalar("{}_loss".format(tf_board_header), current_loss, global_step)
                logging_loss = tr_loss
                logger.info("<-[[O]]-> {}/{} :: loss : {}".format(
                    step+1, args.max_steps*args.gradient_accumulation_steps, current_loss))
                logger.info("Num of batch trained {}".format([(k[0] , v) for k, v in zip(train_data_loader, num_of_batch_trained)] ))
                # Log metrics
                if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                    result_prediction = evaluate(
                                            args, model, tokenizer, 
                                            labels, pad_token_label_id, 
                                            "dev", langs = args.dev_lang, 
                                            logger=logger
                                        )
                    for key, (result, prediction) in result_prediction.items():
                        dataset_f1_score = result["f1"]
                        tb_writer.add_scalar("{}_eval_{}_F1".format(tf_board_header, key), dataset_f1_score, global_step)                            
                        # test_results_prediction = evaluate(
                        #                             args, model, tokenizer, 
                        #                             labels, pad_token_label_id, 
                        #                             "test", langs = args.tgt_lang, 
                        #                             logger=logger
                        #                         )
                        # for key1, (result, prediction) in test_results_prediction.items():
                        #     rel_key = key+"_"+key1
                        #     tb_writer.add_scalar("{}_test_{}_F1".format(tf_board_header, rel_key), result["f1"], global_step)
                        
                        if dataset_f1_score > best_dev_scores[key]:
                            isUpdated = True
                            tb_writer.add_scalar("{}_best_dev_{}_F1".format(tf_board_header, key), dataset_f1_score, global_step)
                            ####################
                            # new best validation set found
                            ####################
                            best_dev_scores[key] = dataset_f1_score
                            # lang = key.split(";")[-1]
                            # for key1, (result, prediction) in test_results_prediction.items():
                            #     rel_key = key+"_"+key1
                            #     test_scores_in_best_src_dev[ rel_key ] = result["f1"]
                            #     tb_writer.add_scalar("{}_test_on_best_dev_{}_F1".format(tf_board_header, rel_key), result["f1"], global_step)

                            #########################
                            # Save model checkpoint
                            #########################
                            logger.info("New best dev found for : {}".format(key))
                            save_model_checkpoint(
                                args, args.output_dir, args.dev_lang, 
                                model, 
                                logger=logger
                            )
                            ###########################

                    ###########################
                    # Saving results on disk in json format
                    ###########################
                    save_results(args, best_dev_scores, test_scores_in_best_src_dev, logger)

        if args.local_rank in [-1, 0]:
            tb_writer.close()

    return global_step, tr_loss / global_step, isUpdated, best_dev_scores, test_scores_in_best_src_dev




def evaluate(
        args, 
        model, tokenizer, labels, 
        pad_token_label_id, mode, 
        prefix="", 
        langs = "en;es;de;nl;ar;fi", 
        logger=None,
        eval_dataset=None,
        head_idx=0
    ):
    if eval_dataset is None:
        eval_dataset, guids = load_and_cache_examples(
                        args, tokenizer, labels, pad_token_label_id, 
                        mode=mode, langs=langs, logger=logger
                    )
    
    args.eval_batch_size = args.per_gpu_eval_batch_size
    all_eval_dataloader = []
    for k, dataset in eval_dataset.items():
        sampler = SequentialSampler(dataset) if args.local_rank == -1 else DistributedSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.eval_batch_size)
        all_eval_dataloader.append((k, dataloader))
    
    all_result = OrderedDict()

    for dataset_index, eval_dataloader in all_eval_dataloader:

        logger.info("***** Running evaluation {} ***** (head:{})".format(dataset_index, head_idx))
        logger.info("  Num examples = {}".format( len(eval_dataloader) ))
        logger.info("  Batch size = %d", args.eval_batch_size)
        
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        cnt = 0
        out_label_ids = None
        model.eval()
        total_number_of_sample = 0
        for batch in eval_dataloader:
            batch = tuple(t.to(args.device) for t in batch)
            if cnt % 20 == 0:
                logger.info("  Evaluating {}/{}".format(cnt, len(eval_dataloader)))
            cnt += 1
            with torch.no_grad():
                inputs = {"input_ids": batch[0],
                          "attention_mask": batch[1],
                          "labels": batch[3],
                          "head_idx": head_idx}
                if args.model_type != "distilbert":
                    inputs["token_type_ids"] = batch[2] if args.model_type in ["bert", "xlnet"] else None  # XLM and RoBERTa don"t use segment_ids
                outputs, per_token_loss = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                if args.n_gpu > 1:
                    tmp_eval_loss = tmp_eval_loss.mean()  # mean() to average on multi-gpu parallel evaluating

                eval_loss += tmp_eval_loss.item()
            nb_eval_steps += 1
            total_number_of_sample += logits.size()[0]
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)
        logger.info("Total number of sample evaluated : {}".format(total_number_of_sample))
        eval_loss = eval_loss / nb_eval_steps
        preds = np.argmax(preds, axis=2)

        label_map = {i: label for i, label in enumerate(labels)}

        out_label_list = [[] for _ in range(out_label_ids.shape[0])]
        preds_list = [[] for _ in range(out_label_ids.shape[0])]

        for i in range(out_label_ids.shape[0]):
            for j in range(out_label_ids.shape[1]):
                if out_label_ids[i, j] != pad_token_label_id:
                    out_label_list[i].append(label_map[out_label_ids[i][j]])
                    preds_list[i].append(label_map[preds[i][j]])

        results = {
            "loss": eval_loss,
            "precision": precision_score(out_label_list, preds_list),
            "recall": recall_score(out_label_list, preds_list),
            "f1": f1_score(out_label_list, preds_list)
        }
        
        all_result[dataset_index] = (results, preds_list)
        # all_result[dataset_index] = (results, preds_list)
        
        logger.info("***** Eval results {} *****".format(dataset_index))
        for key in sorted(results.keys()):
            logger.info("  %s = %s", key, str(results[key]))

    return all_result



def export_logit(
        args, 
        model, tokenizer, labels, 
        pad_token_label_id, mode, 
        prefix="", langs = "en",
        external_data=None,
        examples=None,
        logger=None,
        head_idx=0,
        debug = 0
    ):

    eval_dataset, guids = load_and_cache_examples(
                    args, tokenizer, labels, pad_token_label_id, 
                    mode=mode, langs=langs, 
                    external_data=external_data, examples=examples, 
                    logger=logger
                )
    
    # args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    args.eval_batch_size = 1
    all_eval_dataloader = []
    for k, dataset in eval_dataset.items():
        sampler = SequentialSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=1)
        all_eval_dataloader.append((k, dataloader))

    pseudo_loss_dict = OrderedDict()
    logit_dict = OrderedDict()
    orig_lable_dict = OrderedDict()
    orig_losss_dict = OrderedDict()
    
    for dataset_index, eval_dataloader in all_eval_dataloader:

        logger.info("***** Saving logit {} *****".format(dataset_index))
        logger.info("  Num examples = {}".format( len(eval_dataloader) ))
        logger.info("  Batch size = %d", args.eval_batch_size)

        model.eval()
        cnt = 0
        pseudo_loss_list = []
        logit_bank = []
        orig_loss_list = []
        out_label_ids = []
        for batch in eval_dataloader:
            batch = tuple(t.to(args.device) for t in batch)
            with torch.no_grad():
                
                #############
                # Original label Inference
                #############
                inputs = {"input_ids": batch[0],
                          "attention_mask": batch[1],
                          "labels": batch[3],
                          "head_idx": head_idx}
                sentence_length = np.sum(inputs["labels"].detach().cpu().numpy()!= pad_token_label_id)
                
                if args.model_type != "distilbert":
                    inputs["token_type_ids"] = batch[2] if args.model_type in ["bert", "xlnet"] else None  # XLM and RoBERTa don"t use segment_ids
                outputs, per_token_loss = model(**inputs)
                
                tmp_eval_loss, logits = outputs[:2]
                orig_loss_list.append([tmp_eval_loss.item(), sentence_length, 0, None])

                #############
                # pseudo label collection
                #############
                preds1 = logits.detach().cpu().numpy().argmax(axis=-1)
                out_label_ids1 = inputs["labels"].detach().cpu().numpy()
                mask = out_label_ids1==pad_token_label_id
                preds1[mask] = pad_token_label_id
                incorrect_token = np.sum(preds1!=out_label_ids1)
                
                preds2 = list(preds1[0])
                assert args.eval_batch_size == 1 # see above line
                preds2 = [ i  for i in preds2 if i!=pad_token_label_id]
                

                #############
                # pseudo label Inference
                #############
                _inputs = {"input_ids": batch[0],
                          "attention_mask": batch[1],
                          "labels": torch.from_numpy(preds1).cuda(),
                          "penalty": args.penalty,
                          "head_idx": head_idx}
                # print(torch.from_numpy(preds1).cuda())
                # print(batch[3])
                outputs, per_token_loss = model(**_inputs)
                pseudo_loss, pseudo_logits = outputs[:2]

                pseudo_loss_list.append([pseudo_loss.item(), sentence_length, incorrect_token, preds2])

                mask = out_label_ids1!=pad_token_label_id
                masked_logits = pseudo_logits.detach().cpu().numpy()[mask]
                
                original_label = out_label_ids1[mask] ################################
                
                logit_bank.append(masked_logits)              
                out_label_ids.append(list(original_label))
                ###################################################
            
            cnt += 1
            if cnt%100 == 0:
                logger.info("{} :: Inference done {}/{} batches".format(dataset_index, cnt, len(eval_dataloader)))
                # if debug:
                # xtremebreak ##################################
    
        pseudo_loss_dict[dataset_index] = pseudo_loss_list
        logit_dict[dataset_index] = logit_bank
        orig_lable_dict[dataset_index] = out_label_ids
        orig_losss_dict[dataset_index] = orig_loss_list
    return pseudo_loss_dict, logit_dict, orig_lable_dict, orig_losss_dict, guids


def get_knn_logit_dist(original_logits, reference_logit, reference_lable, k_size):

    distances = np.sqrt((np.square(original_logits[:,np.newaxis]-reference_logit).sum(axis=2)))
    min_indices = distances.argsort()[:,0:k_size]
    min_distances = np.sort(distances,kind='mergesort')[:,0:k_size]
    min_distances_prob = softmax(-min_distances, axis=-1)
    sample_logit = np.zeros( (min_distances_prob.shape[0],9) )
    sample_lables = reference_lable[min_indices]
    for i in range(sample_lables.shape[0]):
        for j in range(sample_lables.shape[1]):
            idx = sample_lables[i][j]
            sample_logit[i][ idx ] += min_distances_prob[i][j]
    return sample_logit




def get_knn_logit_dist_torch_clustered(logits, logit_lable_dict, k_size, number_of_class):
    global_min_distances = None
    global_min_indices = None
    global_sample_lable = None
    for k, v in logit_lable_dict.items():
        torch.cuda.empty_cache()
        reference_logit, reference_lable = logit_lable_dict[k]
        torch.cuda.empty_cache()
        logits = logits.view(-1,1,number_of_class)
        torch.cuda.empty_cache()
        distances = torch.sqrt(torch.sum((logits-reference_logit)**2, axis=2))
        min_distances, min_indices = torch.topk(distances, k=k_size, largest=False)

        global_min_distances = min_distances if global_min_distances is None else \
                        torch.cat((global_min_distances, min_distances), dim=-1)
        global_min_indices = min_indices if global_min_indices is None else \
                        torch.cat((global_min_indices, min_indices), dim=-1)
        global_sample_lable = reference_lable[min_indices] if global_sample_lable is None else \
                        torch.cat((global_sample_lable, reference_lable[min_indices]), dim=-1)
    
    min_distances_prob = torch.softmax(-global_min_distances, axis=-1)
    sample_logits = torch.zeros( (min_distances_prob.size()[0]*min_distances_prob.size()[1], number_of_class) )
    
    global_sample_lable_size = global_sample_lable.size()
    global_sample_lable = global_sample_lable.view(-1)
    
    sample_logits[ torch.arange(global_sample_lable.shape[0]), global_sample_lable ] = 1 
    min_distances_prob = min_distances_prob.view(-1)
    sample_logits = min_distances_prob[:,None].cuda() * sample_logits.cuda()
    
    sample_logits = sample_logits.view(global_min_indices.size()[0], global_min_indices.size()[1], number_of_class)
    sample_logits = torch.mean(sample_logits, axis = -2)

    return sample_logits


def get_knn_logit_dist_torch_non_clustered(logits, logit_lable_dict, k_size, number_of_class):
    
    global_min_distances = None
    global_min_indices = None
    global_sample_lable = None
    
    reference_logit, reference_lable = logit_lable_dict
    logits = logits.view(-1,1, number_of_class)
    distances = torch.sqrt(torch.sum((logits-reference_logit)**2, axis=2))
    min_distances, min_indices = torch.topk(distances, k=k_size, largest=False)

    global_min_distances = min_distances if global_min_distances is None else \
                    torch.cat((global_min_distances, min_distances), dim=-1)
    global_min_indices = min_indices if global_min_indices is None else \
                    torch.cat((global_min_indices, min_indices), dim=-1)
    global_sample_lable = reference_lable[min_indices] if global_sample_lable is None else \
                    torch.cat((global_sample_lable, reference_lable[min_indices]), dim=-1)
    
    min_distances_prob = torch.softmax(-global_min_distances, axis=-1)
    sample_logits = torch.zeros( (min_distances_prob.size()[0]*min_distances_prob.size()[1], number_of_class) )
    
    global_sample_lable_size = global_sample_lable.size()
    global_sample_lable = global_sample_lable.view(-1)
    
    sample_logits[ torch.arange(global_sample_lable.shape[0]), global_sample_lable ] = 1 
    min_distances_prob = min_distances_prob.view(-1)
    sample_logits = min_distances_prob[:,None].cuda() * sample_logits.cuda()
    
    sample_logits = sample_logits.view(global_min_indices.size()[0], global_min_indices.size()[1], number_of_class)
    sample_logits = torch.mean(sample_logits, axis = -2)

    return sample_logits


def inference(
        args, 
        model,
        tokenizer,
        labels,
        pad_token_label_id,
        mode="test",
        path="dumped/test/nl.train.iob2.logit",
        prefix="",
        langs = "nl",
        lam=.05,
        logit_bank_type='non-clustered',
        label_bank=None,
        logger=None,
        logging_iter=100
    ):
    eval_dataset, guids = load_and_cache_examples(
                        args, tokenizer, labels, pad_token_label_id, 
                        mode = mode, langs=langs, logger=logger
                    )

    args.eval_batch_size = 1
    # args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    all_eval_dataloader = []
    for k, dataset in eval_dataset.items():
        sampler = SequentialSampler(dataset) if args.local_rank == -1 else DistributedSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.eval_batch_size)
        all_eval_dataloader.append((k, dataloader))

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)


    all_result = OrderedDict()
    logit_lable_dict = OrderedDict()
    for dataset_index, eval_dataloader in all_eval_dataloader:
    #     data/es/es.train.iob2;utf-8;es
        lang = dataset_index.split(";")[-1]
        logit_lable_dict = OrderedDict()
        address = path
        with open(address, "rb") as filePtr:
            lang_logit_bank = pickle.load(filePtr)
            if logit_bank_type == "non-clustered":
                lang_logit_bank = lang_logit_bank[0]
                if label_bank is None:
                    reference_lable = np.argmax(lang_logit_bank, axis=-1)
                else:
                    reference_lable = np.array(label_bank)
                    # print(reference_lable.shape)
                logit_lable_dict = (torch.from_numpy(lang_logit_bank).cuda(), torch.from_numpy(reference_lable).cuda())
            elif logit_bank_type == "clustered":
                lang_logit_bank = lang_logit_bank[1]
                for k, v in lang_logit_bank.items():
                    reference_lable = np.argmax(v, axis=-1)
                    logit_lable_dict[k] = (torch.from_numpy(v).cuda(), torch.from_numpy(reference_lable).cuda())

        if len(logit_lable_dict) == 0:
            continue
        logger.info("***** Running evaluation {} *****".format(dataset_index))
        logger.info("  Num examples = {}".format( len(eval_dataloader) ))
        logger.info("  Batch size = %d", args.eval_batch_size)    

        model.eval()
        cnt = 0
        predictions = []
        original_label_id = []
        total_batch = len(eval_dataloader)

        for batch in eval_dataloader:
            batch = tuple(t.to(args.device) for t in batch)
            with torch.no_grad():
                inputs = {"input_ids": batch[0],
                          "attention_mask": batch[1],
                          "labels": batch[3]}
                if args.model_type != "distilbert":
                    inputs["token_type_ids"] = batch[2] if args.model_type in ["bert", "xlnet"] else None  # XLM and RoBERTa don"t use segment_ids

                outputs, per_token_loss = model(**inputs)

                tmp_eval_loss, logits = outputs[:2]
                
                if logit_bank_type == "non-clustered":
                    knn_logit = get_knn_logit_dist_torch_non_clustered(logits, logit_lable_dict, args.k_size, logits.size()[-1])
                elif logit_bank_type == "clustered":
                    knn_logit = get_knn_logit_dist_torch_clustered(logits, logit_lable_dict, args.k_size, logits.size()[-1])
                else:
                    raise NotImplementedError()
                knn_logit = knn_logit.view(logits.size())


                convex_comb_logits = lam * torch.softmax(logits, axis=-1) + (1-lam) * torch.softmax(knn_logit, axis=-1)

                label_mask = (inputs['labels'].detach().cpu().numpy() != pad_token_label_id)
                original_label = inputs['labels'].detach().cpu().numpy()[label_mask]
                original_logits = convex_comb_logits.detach().cpu().numpy()[label_mask]

                predictions.append(list(np.argmax(original_logits, axis=1)))
                original_label_id.append(list(original_label))
                assert original_logits.shape[0] == original_label.shape[0]

                cnt += 1
                if cnt % logging_iter == 0  and cnt > 0:
                    logger.info("{} :: inference done {}/{}".format(dataset_index, cnt, total_batch))


        label_map = {i: label for i, label in enumerate(labels)}
        out_label_list = []
        preds_list = []
        for pred, orig in zip(predictions, original_label_id):
            assert len(pred) == len(orig)
            temp_pred = []
            temp_orig = []
            for pred_label_idx, orig_label_idx in zip(pred, orig):
                temp_pred.append(label_map[pred_label_idx])
                temp_orig.append(label_map[orig_label_idx])
            preds_list.append(temp_pred)
            out_label_list.append(temp_orig)

        results = {
            "precision": precision_score(out_label_list, preds_list),
            "recall": recall_score(out_label_list, preds_list),
            "f1": f1_score(out_label_list, preds_list)
        }

        all_result[dataset_index] = results  #, preds_list)

        logger.info("***** Eval results {} *****".format(dataset_index))
        for key in sorted(results.keys()):
            logger.info("  %s = %s", key, str(results[key]))
    return all_result


def task_validation(
        args, 
        model, tokenizer, labels, pad_token_label_id, 
        multi_head_best_dev_scores, multi_head_test_scores_in_best_src_dev, 
        tb_writer, tf_board_header, global_step,
        logger=None
    ):
    
    avg_dev_f1 = {}
    for head_idx in range(args.num_of_heads):
        result_prediction = evaluate(
                                args, model, tokenizer, 
                                labels, pad_token_label_id, 
                                "dev", langs = args.dev_lang, 
                                logger=logger, 
                                head_idx = head_idx
                            )
        for key, (result, prediction) in result_prediction.items():
            if key not in avg_dev_f1:
                avg_dev_f1[key] = [0, []]
            avg_dev_f1[key][0] += result["f1"]
            avg_dev_f1[key][1].append(result["f1"])                         
    
    for key, v in avg_dev_f1.items():
        avg_dev_f1[key][0] = avg_dev_f1[key][0]/float(args.num_of_heads)
        tb_writer.add_scalar("{}_eval_{}_F1".format(tf_board_header, key), avg_dev_f1[key][0], global_step)
        if key not in multi_head_best_dev_scores:
            multi_head_best_dev_scores[key] = [0, []]

    # Test Result Prediction
    avg_test_f1 = {}
    for head_idx in range(args.num_of_heads):
        test_results_prediction = evaluate(
                                        args, model, tokenizer, 
                                        labels, pad_token_label_id, 
                                        "test", langs = args.tgt_lang, 
                                        logger=logger,
                                        head_idx=head_idx
                                )
        for key, (result, prediction) in test_results_prediction.items():
            if key not in avg_test_f1:
                avg_test_f1[key] = [0, []]
            avg_test_f1[key][0] += result["f1"]
            avg_test_f1[key][1].append(result["f1"])   
        
    for key, v in avg_test_f1.items():
        avg_test_f1[key][0] = avg_test_f1[key][0]/float(args.num_of_heads)
        tb_writer.add_scalar("{}_test_{}_F1".format(tf_board_header, key), avg_test_f1[key][0], global_step)
    
    for key, v in avg_dev_f1.items():
        if avg_dev_f1[key][0] > multi_head_best_dev_scores[key][0]:
            multi_head_best_dev_scores[key] = avg_dev_f1[key]

            #########################
            # Save model checkpoint
            #########################
            logger.info("New best dev found for : {}".format(key))
            save_model_checkpoint(
                args, args.output_dir, args.dev_lang, 
                model, 
                logger=logger
            )
            ###########################
            
            for test_key, test_v in avg_test_f1.items():
                rel_key = key+"_"+test_key
                multi_head_test_scores_in_best_src_dev[ rel_key ] = avg_test_f1[test_key]
    return multi_head_best_dev_scores, multi_head_test_scores_in_best_src_dev



def multi_head_training_loop(
    args, 
    train_dataset, model, tokenizer, 
    labels, pad_token_label_id, 
    logger=None, 
    multi_head_best_dev_scores = {}, 
    multi_head_test_scores_in_best_src_dev={}, 
    tf_board_header="single"
):
    
    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    
    if args.local_rank in [-1, 0]:
        tb_path = os.path.join(args.output_dir, "tf_board")
        if not os.path.exists(tb_path):
            os.makedirs(tb_path)
        tb_writer = SummaryWriter(tb_path)
    
    logger.info("Evaluate before starting the training loop ...")
    logger.info("-"*20)
    
    multi_head_best_dev_scores, multi_head_test_scores_in_best_src_dev = task_validation(
        args, 
        model, tokenizer, labels, pad_token_label_id, 
        multi_head_best_dev_scores, multi_head_test_scores_in_best_src_dev, 
        tb_writer, tf_board_header, 0,
        logger=logger
    )
            
    dataset_len, lang_prob_index, lang_prob = iterator_selection_prob(args.lang_alpha, train_dataset, logger=logger)
    train_data_loader = []
    for k in lang_prob_index:
        if k in train_dataset:
            dataset = train_dataset[k]
            train_sampler = RandomSampler(dataset) if args.local_rank == -1 else DistributedSampler(dataset)
            data_loader = DataLoader(dataset, sampler=train_sampler, batch_size=args.per_gpu_train_batch_size)
            train_data_loader.append((k, data_loader))

    model, optimizer, scheduler = get_optimizer(args, model)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num of batch = {}".format(dataset_len))
    logger.info("  Instantaneous batch size per GPU = {}".format(args.per_gpu_train_batch_size))
    logger.info("  Gradient Accumulation steps = {}".format(args.gradient_accumulation_steps))
    logger.info("  Effective train batch size (w. parallel, distributed & accumulation) = {}".format(
                        args.per_gpu_train_batch_size * args.gradient_accumulation_steps))
    logger.info("  Total optimization steps = {}".format(args.max_steps))


    train_iterators = []
    for i in range(len(train_data_loader)):
        assert train_data_loader[i][0] == lang_prob_index[i]
        train_iterators.append(iter(train_data_loader[i][1]))
    tot_num_of_iterator = len(train_iterators)


    # set_seed(args)    
    
    num_of_batch_trained = [ 0 for i in range(tot_num_of_iterator) ]
    isUpdated = 0
    for step in range(args.max_steps*args.gradient_accumulation_steps):
        
        model.train()
        
        iterator_id = np.random.choice(range(tot_num_of_iterator), p=lang_prob)
        try:
            batch = train_iterators[iterator_id].__next__()
        except StopIteration:
            train_iterators[iterator_id] = iter(train_data_loader[iterator_id][1])
            batch = train_iterators[iterator_id].__next__()
        num_of_batch_trained[ iterator_id ] += 1
        
        batch = tuple(t.to(args.device) for t in batch)
        _head_idx = np.random.choice(range(args.num_of_heads), p=[1/3.0, 1/3.0, 1/3.0])
        inputs = {"input_ids": batch[0],
                    "attention_mask": batch[1],
                    "labels": batch[3],
                    "penalty": args.penalty,
                    "head_idx": _head_idx}
        if args.model_type != "distilbert":
            # XLM and RoBERTa don't use segment_ids
            inputs["token_type_ids"] = batch[2] if args.model_type in ["bert", "xlnet"] else None  
            
        outputs, per_token_loss = model(**inputs)
        loss = outputs[0]
        
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps

        loss.backward()
        tr_loss += loss.item()

        if (step + 1) % args.gradient_accumulation_steps == 0:
            if args.fp16:
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            
            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            model.zero_grad()
            global_step += 1
            if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                if args.alpha_schedule == "random":
                    alpha = np.random.random_sample()
                    if alpha == 0.0:
                        alpha = .1
                    elif alpha == 1.0:
                        alhpa = .9
                    dataset_len, lang_prob_index, lang_prob = iterator_selection_prob(alpha, train_dataset, logger=logger)
                
                current_loss = (tr_loss - logging_loss) / args.logging_steps
                current_lr_rate = scheduler.get_lr()[0]
                tb_writer.add_scalar("{}_lr".format(tf_board_header), current_lr_rate, global_step)
                tb_writer.add_scalar("{}_loss".format(tf_board_header), current_loss, global_step)
                logging_loss = tr_loss
                logger.info("<-[[O]]-> {}/{} :: loss : {}".format(
                    step+1, args.max_steps*args.gradient_accumulation_steps, current_loss))
                logger.info("Num of batch trained {}".format([(k[0] , v) for k, v in zip(train_data_loader, num_of_batch_trained)] ))
                # Log metrics
                if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                    # Dev Result Prediction
                    
                    multi_head_best_dev_scores, multi_head_test_scores_in_best_src_dev = task_validation(
                            args, 
                            model, tokenizer, labels, pad_token_label_id, 
                            multi_head_best_dev_scores, multi_head_test_scores_in_best_src_dev, 
                            tb_writer, tf_board_header, global_step,
                            logger=logger
                        )        

                    ###########################
                    # Saving results on disk in json format
                    ###########################
                    save_results(args, multi_head_best_dev_scores, multi_head_test_scores_in_best_src_dev, logger)

        if args.local_rank in [-1, 0]:
            tb_writer.close()

    return global_step, tr_loss / global_step, isUpdated, multi_head_best_dev_scores, multi_head_test_scores_in_best_src_dev


def ensamble_evaluate(
        args, 
        models, tokenizers, labels, 
        pad_token_label_id, mode, 
        prefix="", 
        langs = "en;es;de;nl;ar;fi", 
        logger=None,
        eval_dataset=None,
        head_idx=0
    ):
    if eval_dataset is None:
        eval_dataset, guids = load_and_cache_examples(
                        args, tokenizers[0], labels, pad_token_label_id, 
                        mode=mode, langs=langs, logger=logger
                    )
    
    args.eval_batch_size = args.per_gpu_eval_batch_size
    all_eval_dataloader = []
    for k, dataset in eval_dataset.items():
        sampler = SequentialSampler(dataset) if args.local_rank == -1 else DistributedSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.eval_batch_size)
        all_eval_dataloader.append((k, dataloader))
    
    all_result = OrderedDict()

    for dataset_index, eval_dataloader in all_eval_dataloader:

        logger.info("***** Running evaluation {} ***** (head:{})".format(dataset_index, head_idx))
        logger.info("  Num examples = {}".format( len(eval_dataloader) ))
        logger.info("  Batch size = %d", args.eval_batch_size)
        
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        cnt = 0
        out_label_ids = None
        models[0].eval()
        models[1].eval()
        models[2].eval()
        total_number_of_sample = 0
        for batch in eval_dataloader:
            batch = tuple(t.to(args.device) for t in batch)
            if cnt % 20 == 0:
                logger.info("  Evaluating {}/{}".format(cnt, len(eval_dataloader)))
            cnt += 1
            with torch.no_grad():
                inputs = {"input_ids": batch[0],
                          "attention_mask": batch[1],
                          "labels": batch[3],
                          "head_idx": head_idx}
                if args.model_type != "distilbert":
                    inputs["token_type_ids"] = batch[2] if args.model_type in ["bert", "xlnet"] else None  # XLM and RoBERTa don"t use segment_ids
                outputs0, _ = models[0](**inputs)
                tmp_eval_loss0, logits0 = outputs0[:2]
                
                outputs1, _ = models[1](**inputs)
                tmp_eval_loss1, logits1 = outputs1[:2]
                
                outputs2, _ = models[2](**inputs)
                tmp_eval_loss2, logits2 = outputs2[:2]

                if args.n_gpu > 1:
                    tmp_eval_loss0 = tmp_eval_loss0.mean()  # mean() to average on multi-gpu parallel evaluating
                    tmp_eval_loss1 = tmp_eval_loss1.mean()  # mean() to average on multi-gpu parallel evaluating
                    tmp_eval_loss2 = tmp_eval_loss2.mean()  # mean() to average on multi-gpu parallel evaluating

                eval_loss += (tmp_eval_loss0.item()+tmp_eval_loss1.item()+tmp_eval_loss2.item())/3.0
            
            nb_eval_steps += 1
            total_number_of_sample += logits0.size()[0]
            
            preds0 = logits0.detach().cpu().numpy()
            preds1 = logits1.detach().cpu().numpy()
            preds2 = logits2.detach().cpu().numpy()
            c_preds = preds0+preds1+preds2/3.0
            
            if preds is None:
                preds = c_preds
                out_label_ids = inputs["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, c_preds, axis=0)
                out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)
        
        logger.info("Total number of sample evaluated : {}".format(total_number_of_sample))
        eval_loss = eval_loss / nb_eval_steps
        preds = np.argmax(preds, axis=2)

        label_map = {i: label for i, label in enumerate(labels)}

        out_label_list = [[] for _ in range(out_label_ids.shape[0])]
        preds_list = [[] for _ in range(out_label_ids.shape[0])]

        for i in range(out_label_ids.shape[0]):
            for j in range(out_label_ids.shape[1]):
                if out_label_ids[i, j] != pad_token_label_id:
                    out_label_list[i].append(label_map[out_label_ids[i][j]])
                    preds_list[i].append(label_map[preds[i][j]])

        results = {
            "loss": eval_loss,
            "precision": precision_score(out_label_list, preds_list),
            "recall": recall_score(out_label_list, preds_list),
            "f1": f1_score(out_label_list, preds_list)
        }
        
        all_result[dataset_index] = (results, preds_list)
        # all_result[dataset_index] = (results, preds_list)
        
        logger.info("***** Eval results {} *****".format(dataset_index))
        for key in sorted(results.keys()):
            logger.info("  %s = %s", key, str(results[key]))

    return all_result


def ensamble_eval(args, labels, pad_token_label_id, MODEL_CLASSES, logger=None):
    num_labels = len(labels)
    config, tokenizer, model = [None,None,None], [None,None,None], [None,None,None]
    for i in range(3):
        config[i], tokenizer[i], model[i] = load_model(
                    args.model_type, MODEL_CLASSES, 
                    args.thetas[i], args.config_name, args.tokenizer_name,
                    num_labels, args.cache_dir, args.do_lower_case, args.device, dropout=args.dropout
                )
    dev_scores, test_scores = [None,None,None], [None,None,None]
    for i in range(3):
        logger.info("\n Model {}: Evaluating by single model inference.\n".format(i))
        dev_scores[i] = evaluate(
                        args, 
                        model[i], tokenizer[i], labels, 
                        pad_token_label_id, "dev", 
                        prefix="", langs = args.tgt_lang,
                        logger=logger
                    )    
        test_scores[i] = evaluate(
                        args, 
                        model[i], tokenizer[i], labels, 
                        pad_token_label_id, "test", 
                        prefix="", langs = args.tgt_lang,
                        logger=logger
                    ) 
    # all_result[dataset_index]
    if (args.tgt_lang.split(";"))==1:
        for k, v in test_scores.items():
            s = []
            for i in range(3):
                s.append(test_scores[i][k][0]["f1"])
            logger.info("\n\n")
            logger.info("dataset : {}, LANG : {} , F1 : {} +- {}\n".format(k.split(";")[0], k.split(";")[-1]), np.average(s), np.std(s))
            logger.info("\n\n")


    prev_best_dev_scores = None
    prev_test_scores_in_best_src_dev = None
    prev_best_test_scores = None
    best_dev_scores = {file_name:0 for file_name in args.dev} if prev_best_dev_scores is None else prev_best_dev_scores
    test_scores_in_best_src_dev = {} if prev_test_scores_in_best_src_dev is None else prev_test_scores_in_best_src_dev
    best_test_scores = {file_name:0 for file_name in args.dev} if prev_best_test_scores is None else prev_best_test_scores

    if args.ensamble_type=='logit':
        result_prediction = ensamble_evaluate(
                                args, model, tokenizer, 
                                labels, pad_token_label_id, 
                                "dev", langs = args.dev_lang, 
                                logger=logger
                            )
        for key, (result, prediction) in result_prediction.items():
            dataset_f1_score = result["f1"]
            test_results_prediction = ensamble_evaluate(
                                        args, model, tokenizer, 
                                        labels, pad_token_label_id, 
                                        "test", langs = args.tgt_lang, 
                                        logger=logger
                                    )
            for key1, (result, prediction) in test_results_prediction.items():
                rel_key = key+"_"+key1
            
            if dataset_f1_score > best_dev_scores[key]:
                ####################
                # new best validation set found
                ####################
                best_dev_scores[key] = dataset_f1_score
                lang = key.split(";")[-1]
                for key1, (result, prediction) in test_results_prediction.items():
                    rel_key = key+"_"+key1
                    test_scores_in_best_src_dev[ rel_key ] = result["f1"]

                #########################
                # Save model checkpoint
                #########################
                # logger.info("New best dev found for : {}".format(key))
                # save_model_checkpoint(
                #     args, args.output_dir, args.dev_lang, 
                #     model, 
                #     logger=logger
                # )
                ###########################

        ###########################
        # Saving results on disk in json format
        ###########################
        save_results(args, best_dev_scores, test_scores_in_best_src_dev, logger)
    elif args.ensamble_type=='output':
        raise NotImplementedError()
    else:
        raise NotImplementedError()


def single_model_eval(args, labels, pad_token_label_id, MODEL_CLASSES, logger=None):
    num_labels = len(labels)
    config, tokenizer, model = load_model(
            args.model_type, MODEL_CLASSES, 
            args.model_name_or_path, args.config_name, args.tokenizer_name,
            num_labels, args.cache_dir, args.do_lower_case, args.device, dropout=args.dropout
        )

    prev_best_dev_scores = None
    prev_test_scores_in_best_src_dev = None
    prev_best_test_scores = None
    best_dev_scores = {file_name:0 for file_name in args.dev} if prev_best_dev_scores is None else prev_best_dev_scores
    test_scores_in_best_src_dev = {} if prev_test_scores_in_best_src_dev is None else prev_test_scores_in_best_src_dev
    best_test_scores = {file_name:0 for file_name in args.dev} if prev_best_test_scores is None else prev_best_test_scores

    if args.ensamble_type=='logit':
        result_prediction = evaluate(
                                args, model, tokenizer, 
                                labels, pad_token_label_id, 
                                "dev", langs = args.dev_lang, 
                                logger=logger
                            )
        for key, (result, prediction) in result_prediction.items():
            dataset_f1_score = result["f1"]
            test_results_prediction = evaluate(
                                        args, model, tokenizer, 
                                        labels, pad_token_label_id, 
                                        "test", langs = args.tgt_lang, 
                                        logger=logger
                                    )
            for key1, (result, prediction) in test_results_prediction.items():
                rel_key = key+"_"+key1
            
            if dataset_f1_score > best_dev_scores[key]:
                ####################
                # new best validation set found
                ####################
                best_dev_scores[key] = dataset_f1_score
                lang = key.split(";")[-1]
                for key1, (result, prediction) in test_results_prediction.items():
                    rel_key = key+"_"+key1
                    test_scores_in_best_src_dev[ rel_key ] = result["f1"]

                #########################
                # Save model checkpoint
                #########################
                # logger.info("New best dev found for : {}".format(key))
                # save_model_checkpoint(
                #     args, args.output_dir, args.dev_lang, 
                #     model, 
                #     logger=logger
                # )
                ###########################

        ###########################
        # Saving results on disk in json format
        ###########################
        save_results(args, best_dev_scores, test_scores_in_best_src_dev, logger)
    elif args.ensamble_type=='output':
        raise NotImplementedError()
    else:
        raise NotImplementedError()