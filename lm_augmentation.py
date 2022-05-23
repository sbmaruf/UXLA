import os
import json
import copy
import torch
import pickle
import random
import argparse
import jsbeautifier
import numpy as np
from logger import create_logger
from utils_ner import read_from_path, read_examples_from_file, InputExample
opts = jsbeautifier.default_options()
parser = argparse.ArgumentParser("Cross-lingual Contextual NER.")
parser.add_argument("--dataset",
                        nargs='*',
                        default=["data/en/en.train.iob2;utf-8;en", 
                                 "data/es/es.train.iob2;utf-8;es", 
                                 "data/de/de.train.iob2;latin-1;de", 
                                 "data/nl/nl.train.iob2;utf-8;nl", 
                                 "data/ar/ar.train.iob2;utf-8;ar", 
                                 "data/fi/fi.train.iob2;utf-8;fi"], 
                        help="dataset location. Value-type: list(string)")
parser.add_argument("--output_dir", 
                        default="./dumped", 
                        type=str, 
                        help="The output directory where the model predictions and checkpoints will be written.")
parser.add_argument("--aug_type",
                        default="per_token",
                        type=str,
                        help="what type of augmentation will be done.",
                        choices=["per_token", "successive_cross", "successive_max"])
parser.add_argument("--aug_per",
                        default=15,
                        type=float,
                        help="Percentage of token will be replaced, activated only when `aug_type`=`successive`")
parser.add_argument("--topk",
                        default=3,
                        type=int,
                        help="Number of prediction will be done for a sample.")
parser.add_argument("--num_of_aug",
                        default=3,
                        type=int,
                        help="number of augmentation will be done for a single sample.")
parser.add_argument("--only_ner_aug",
                        default=0,
                        type=int,
                        help="Only the NER tags will be augmented.")
parser.add_argument("--seed",
                        default=1234,
                        type=int,
                        help="Seed value for random index selection.")
parser.add_argument("--mode",
                        default="train",
                        type=str,
                        help="Default mode of the dataset.")
parser.add_argument("--train_data_percentage", 
                        default=100, 
                        type=int,
                        help="Percentage of training data that will be selected.")

unimportant_word = ["the"]
def get_masked_sentences_at_pos(sentence, pos):
    try:
        temp = sentence[pos]
        sentence[pos] = "<mask>"
    except:
        pos = 0
        temp = sentence[pos]
        sentence[pos] = "<mask>"
    cp_sentence = copy.deepcopy(sentence)
    mask_sent_str = " ".join([w for w in cp_sentence])
    sentence[pos] = temp
    return mask_sent_str
def get_per_token_masked_sentences(xlmr, sentence, topk, only_ner_aug=0, debug=0):
    """
    Replace each of the token by mask and create a predicted sentence.
    """
    mask_sentence_list = []
    str_sent = " ".join([w[0] for w in sentence])
    for idx, word_info in enumerate(sentence):
        if only_ner_aug == 1 and word_info[-1] == "O":
            continue
        mask_sent = get_masked_sentences_at_pos(str_sent.split(), idx)
        if debug==1:
            print("==> ", str_sent)
            print("**> ", mask_sent)
        mask_filled_sentences = xlmr.fill_mask(mask_sent, topk=topk)
        for top_idx, mask_filled_sentence in enumerate(mask_filled_sentences):
            pos = idx if len(mask_sent.split()) == len(mask_filled_sentence[0].split()) else -1
            if debug==1:
                print("\t~~>", mask_filled_sentence[0], top_idx+1, idx)
            if mask_filled_sentence[0].split() == str_sent.split() or len(mask_filled_sentence[0].split()) != len(str_sent.split()): 
                if debug==1:
                    print("\t\t>>>  sentence skipped") 
                continue
            mask_sentence_list.append((mask_filled_sentence[0], top_idx+1, ["X"  if i == idx else w[-1] for i, w in enumerate(sentence)]))
    return mask_sentence_list
def get_stochastic_indices(num_of_mask_token, total_indices, indices_var, sentence):
    cnt = 0
    indices = []
    _flag = 0
    while cnt < num_of_mask_token:
        try:
            if len(sentence[ total_indices[indices_var] ][0]) < 3:
                indices_var += 1
                _flag += 1
                if indices_var == len(total_indices):
                    indices_var = 0
                    np.random.shuffle(indices)
                if _flag == len(total_indices):
                    break
                
                continue
        except:
            print(sentence)
            print(total_indices)
            print(indices_var)
            raise
        indices.append( 
            total_indices[indices_var]
        )
        cnt = cnt + 1
        indices_var = indices_var + 1
        if indices_var == len(total_indices):
            indices_var = 0
            np.random.shuffle(indices)
    
    if len(indices)==0:
        return None, None
    mask_indices = indices
    return mask_indices, indices_var
def get_successive_masked_sentences(mask_indices, sentences, xlmr, topk, aug_type, debug=0):
    for _id, pos in enumerate(mask_indices):
        # be careful about save_sentence and check it's manipulation.
        save_sentences = []
        for sent in sentences:
            if debug==1:
                tab="".join(["\t" for i in range(_id+1)])
                print("{}==> {} {}".format(tab, pos, sent))
            # get mask
            mask_sent = get_masked_sentences_at_pos(sent.split(), pos)
            if debug==1:
                tab="".join(["\t" for i in range(_id+1)])
                print("{}**> {}".format(tab, mask_sent))
            
            # mask fill 
            mask_filled_sentences = xlmr.fill_mask(mask_sent, topk=topk)
            
            if aug_type == "successive_cross":
                # augment all
                for mask_filled_sentence in mask_filled_sentences:
                    if debug==1:
                        tab="".join(["\t" for i in range(_id+1)])
                        print("\t{}~~> {}".format(tab, mask_filled_sentence[0]))
                    save_sentences.append(mask_filled_sentence[0])
            elif aug_type == "successive_max":
                # only augment max prediction
                if debug==1:
                    tab="".join(["\t" for i in range(_id+1)])
                    print("\t{}~~> {}".format(tab, mask_filled_sentences[0][0]))
                save_sentences.append(mask_filled_sentences[0][0])
        
        sentences = copy.deepcopy(save_sentences)
    
    return sentences
def is_exists(indices_set, mask_indices):
    flag = 0
    temp = sorted(copy.deepcopy(mask_indices))
    for ind in indices_set:
        if ind == temp:
            flag = 1
            break      
    return flag, temp
    
def get_stocastic_masked_sentences(
                xlmr,
                sentence,
                aug_type,
                topk, 
                aug_per,  
                num_of_aug,
                debug=0
    ):
    """
    Two types of stocastic augmentation is there
    1. cross : each time it augment n number of sentences. so in first step
                if it augment 3 sentence, and from there each of the 3 sentence 
                can augment another 3 sentence. sentence growth : exponential.
    2. max: each time it augment `topk` number of sentences and the best prediction 
                if selected for next step. In the last step `topk` number of sentences
                are selected. sentence growth : linear.
    """
    mask_sentence_list = []
    total_number_of_token = len(sentence)
    num_of_mask_token = int(max(len(sentence)*aug_per/100.0, 1))
    # generate randon indices
    total_indices = [ i for i in range(total_number_of_token) ]
    np.random.shuffle(total_indices)
    if debug==1:
        print("positions :", total_indices)
        # print(sentence)
    
    indices_var = 0
    all_sentences = []
    indices_set = []
    for i in range(num_of_aug):
        if indices_var == len(total_indices):
            indices_var = 0
            np.random.shuffle(indices)
        
        mask_indices, indices_var = get_stochastic_indices(
                num_of_mask_token, total_indices, indices_var, sentence
            )
        
        if mask_indices is None:
            return None
        flag, sorted_mask_indices = is_exists(indices_set, mask_indices)
        if flag:
            continue
        indices_set.append(sorted_mask_indices)
        
        if debug==1:
            print("POSITIONS : {}".format(mask_indices))
            print("++>"," ".join([w[0] for w in sentence]))
        sentences = [" ".join([w[0] for w in sentence])]
        aug_sentences = get_successive_masked_sentences(mask_indices, sentences, xlmr, topk, aug_type, debug)
        for __i, sent in enumerate(aug_sentences):
            if debug==1:
                print("<0> {}. {}".format(__i+1, sent))
            if total_number_of_token != len(sent.split()):
                if debug==1:
                    print(">>> skipped {}. {}".format(__i+1, sent))
                continue
            mask_sentence_list.append((sent, __i+1, [ "X" if i in mask_indices else w[-1]  for i, w in enumerate(sentence) ] ))
    
    return mask_sentence_list
def augment_dataset(
        xlmr, file_info, 
        aug_type, 
        only_ner_aug,
        topk, 
        aug_per,
        num_of_aug,
        seed, 
        mode="train",
        logger=None,
        max_sentence_len=150,
        min_sentence_len=5,
        train_data_percentage=100,
        debug=0
    ):
    """
    This function augment sentences based on augmentation type.
    """
    address, encoding, lang = file_info.split(";")[0], file_info.split(";")[1], file_info.split(";")[2]
    # sentences = read_from_path(address, encoding=encoding, percentage=train_data_percentage, seed=seed)
    examples = read_examples_from_file(address, encoding, lang, mode, seed, percentage=train_data_percentage)
    total_number_of_sent = len(examples)
    aug_examples = []
    for idx, example in enumerate(examples):
        sent = [[w_0, w_1]for (w_0, w_1) in zip(example.words, example.labels)]
        if len([w[-1] for w in sent]) > max_sentence_len or len([w[-1] for w in sent]) < min_sentence_len:
            continue
        if aug_type=="per_token":
            augment_data = get_per_token_masked_sentences(
                xlmr,
                sent,
                topk=topk,
                only_ner_aug=only_ner_aug,
                debug=debug
            )
        elif aug_type == "successive_cross" or aug_type == "successive_max":
            augment_data = get_stocastic_masked_sentences(
                xlmr,
                sent,
                aug_type=aug_type,
                topk=topk, 
                aug_per=aug_per,  
                num_of_aug=num_of_aug,
                debug=debug
            )
        else:
            raise NotImplementedError()
        
        if augment_data is None:
            continue
        
        for sent_data in augment_data:
            try:
                words = sent_data[0].split()
                labels = sent_data[-1]
                assert len(words) == len(labels)
                aug_examples.append(
                    InputExample(
                        guid=example.guid,
                        words=words,
                        labels=labels,
                        lang=example.lang,
                        orig_words = example.words,
                        orig_label = example.labels
                    )
                )
            except:
                logger.warning("Sentence augmentation error, sent idx : {}\n"
                               " sentence : {}\n"
                               " tokens : {}\n"
                               " original sent : {}\n"
                               " original token : {}".format(
                                   example.guid, words, example.words, example.labels
                                )
                        )
        
        # if debug == 2: 
        #     input(":")
            
        if len(aug_examples) % 100 == 0:
            logger.info("Tot num of sent(s) : {}/{}, Tot aug sent(s) : {}"
                      .format(idx, total_number_of_sent, len(aug_examples)))
            # if debug == 2:
            #     break
            # break #######################################################################################
    return aug_examples
def write_conll_augmented_data(file_info, aug_sentences, logger=None):
    """
    Write text in conll format. tags are not written.
    An additional id is added which represnts sentence priority.
    The larger the id is the worst the prediction was.
    id 0 means the sentence is from original dataset.
    Hence the augmented sentences starts from id = 1
    """
    file_address, encoding, lang = file_info.split(";")[0], file_info.split(";")[1], file_info.split(";")[2]
    filePtr = open(file_address, "w", encoding=encoding)
    logger.info("Writting data on {}".format(file_address))
    for sent_info in aug_sentences:
        try:
            pred_sent, pred_idx, orig_token = sent_info[0], sent_info[1], sent_info[2]
            pred_sent = pred_sent.strip().split()
            assert len(pred_sent) == len(orig_token)
            for idx, (sent_tok, orig_tok_label) in enumerate(zip(pred_sent, orig_token)):
                __id = str(pred_idx)
                try:
                    filePtr.write("{} {} {}\n".format(sent_tok, __id, orig_tok_label))
                except:
                    pass
            filePtr.write("\n")
        except:
            pass
    output_written_file_info = file_address+";"+encoding+";"+lang
    return output_written_file_info
def write_conll_augmented_data(file_info, aug_sentences, logger=None):
    """
    Write text in conll format. tags are not written.
    An additional id is added which represnts sentence priority.
    The larger the id is the worst the prediction was.
    id 0 means the sentence is from original dataset.
    Hence the augmented sentences starts from id = 1
    """
    file_address, encoding, lang = file_info.split(";")[0], file_info.split(";")[1], file_info.split(";")[2]
    filePtr = open(file_address, "w", encoding=encoding)
    logger.info("Writting data on {}".format(file_address))
    for sent_info in aug_sentences:
        try:
            pred_sent, pred_idx, orig_token = sent_info[0], sent_info[1], sent_info[2]
            pred_sent = pred_sent.strip().split()
            assert len(pred_sent) == len(orig_token)
            for idx, (sent_tok, orig_tok_label) in enumerate(zip(pred_sent, orig_token)):
                __id = str(pred_idx)
                try:
                    filePtr.write("{} {} {}\n".format(sent_tok, __id, orig_tok_label))
                except:
                    pass
            filePtr.write("\n")
        except:
            pass
    output_written_file_info = file_address+";"+encoding+";"+lang
    return output_written_file_info
def write_pickle(file_info, aug_examples, logger=None):
    file_address, encoding, lang = file_info.split(";")[0], file_info.split(";")[1], file_info.split(";")[2]
    with open(file_address, "wb") as filePtr:
        pickle.dump(aug_examples, filePtr, pickle.HIGHEST_PROTOCOL)
    return file_info
def get_name(  
        aug_type, 
        only_ner_aug, 
        topk, 
        aug_per, 
        num_of_aug, 
        seed,
        is_small_name=0,
    ):
    """
    Prepaer a name
    """
    if is_small_name:
        return "." + "aug"
    ret_name = ".aug_type."+str(aug_type)
    ret_name = ret_name+".only_ner_aug."+str(only_ner_aug)
    ret_name = ret_name+".topk."+str(topk)
    ret_name = ret_name+".aug_per."+str(aug_per)
    ret_name = ret_name+".num_of_aug."+str(num_of_aug)
    ret_name = ret_name+".seed."+str(seed)+".aug"
    return ret_name
    
def augment_data(
    dataset_list, 
    output_dir, 
    aug_type, 
    only_ner_aug, 
    topk, 
    aug_per, 
    num_of_aug, 
    mode="train",
    xlmr=None,
    seed=1234, 
    logger=None,
    is_small_name=1,
    train_data_percentage=100,
    debug=0
):
    """
    given sufficient params it augments data, save it to disk and 
    return the datas and augmented sentences.
    """
    logger.info("dataset_list : {}".format(dataset_list))
    logger.info("output_dir : {}".format(output_dir))
    logger.info("aug_type : {}".format(aug_type))
    logger.info("only_ner_aug : {}".format(only_ner_aug))
    logger.info("topk : {}".format(topk))
    logger.info("mode : {}".format(mode))
    logger.info("aug_per : {}".format(aug_per))
    logger.info("num_of_aug : {}".format(num_of_aug))
    logger.info("seed : {}".format(seed))
    logger.info("debug : {}".format(debug))
    random.seed(seed)
    
    if xlmr is None:
        xlmr = torch.hub.load('pytorch/fairseq', 'xlmr.large')  # load LM
        xlmr.cuda()
        xlmr.eval()
    
    output_file_address = []
    output_aug_sentences = []
    for dt in dataset_list:
        logger.info("Processing dataset {}".format(dt))
        # augment function
        aug_sentences = augment_dataset(
            xlmr, dt, 
            aug_type=aug_type, 
            only_ner_aug=only_ner_aug,
            topk=topk, 
            mode=mode,
            aug_per=aug_per,
            num_of_aug=num_of_aug, 
            seed=seed,
            logger=logger,
            train_data_percentage=train_data_percentage,
            debug=debug
        ) 
        # dataset format, file_address;encoding;language 
        fine_name, encoding, lang = os.path.split(dt.split(";")[0])[-1], dt.split(";")[1], dt.split(";")[2]
        new_file_name = fine_name + get_name(aug_type, only_ner_aug, topk, aug_per, num_of_aug, seed, is_small_name)
        # add params in the file_name so that hyperparms can be identified from file name
        file_info = os.path.join(output_dir, new_file_name) + ";" + encoding + ";" + lang
        # write augmented data in the file  
        # output_written_file_info = write_conll_augmented_data(file_info, aug_sentences, logger=logger)
        output_written_file_info = write_pickle(file_info, aug_sentences, logger=logger)
        # save it for return type
        output_file_address.append(output_written_file_info)
        output_aug_sentences.append(aug_sentences)
        
    return output_file_address, output_aug_sentences
def main():
    args = parser.parse_args()
    logger = create_logger(os.path.join(args.output_dir, "ner_augment.log")) # create a logger
    logger.info("{}".format(jsbeautifier.beautify(json.dumps(args.__dict__), opts))) # params printing
    # function call, it can be called outside of this python file ( not dependent of args )
    augment_data(
        dataset_list=args.dataset, 
        output_dir=args.output_dir,
        aug_type=args.aug_type, 
        only_ner_aug=args.only_ner_aug,
        topk=args.topk, 
        mode=args.mode,
        aug_per=args.aug_per,
        num_of_aug=args.num_of_aug,
        seed=args.seed,
        train_data_percentage=args.train_data_percentage,
        logger=logger
    )
if __name__ == "__main__":
    main()