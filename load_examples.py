import os
import torch
import pickle
from collections import OrderedDict
from utils_ner import convert_examples_to_features, get_labels, read_examples_from_file
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

def load_and_cache_examples(
        args, 
        tokenizer, 
        labels, 
        pad_token_label_id, 
        mode, 
        langs="en;es;de;nl;ar;fi", 
        external_data=None,
        examples=None,
        logger=None
    ):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  

    if mode == "train":
        address_list = args.train
    elif mode == "dev":
        address_list = args.dev
    elif mode == "test":
        address_list = args.test
    elif mode == "aug":
        address_list = args.external_data if external_data is None else external_data
    else:
        raise NotImplementedError()

    langs_address = []
    for address in address_list:
        for lang in langs.split(";"):
            if lang in address.split(";")[-1]:
                langs_address.append((address,lang))

    all_lang_feature = OrderedDict()
    all_lang_guid = OrderedDict()
    for _address, lang in langs_address:
        address, encoding, lang = _address.split(";")[0], _address.split(";")[1], _address.split(";")[2]
        dataset_key = _address
        # Load data features from cache or dataset file
        address_folder = args.output_dir
        cached_features_file = os.path.join(
                                    address_folder, "cached_{}_{}_{}_{}_per_{}".format(
                                                                    mode,
                                                                    os.path.split(address)[1].replace(".","_"),
                                                                    list(filter(None, args.model_name_or_path.split("/"))).pop(),
                                                                    str(args.max_seq_length),
                                                                    str(args.train_data_percentage if mode == "train" else 100)
                                                                )
                                )
        if os.path.exists(cached_features_file) and not args.overwrite_cache:
            logger.info("Loading features from cached file {}".format(cached_features_file))
            features = torch.load(cached_features_file)
            with open(cached_features_file+".guid.pkl", "rb") as filePtr:
                guids = pickle.load(filePtr)
            all_lang_feature[dataset_key] = features
            all_lang_guid[dataset_key] = guids
        else:
            logger.info("Creating features from dataset file at {}".format(address))
            percentage = args.train_data_percentage if mode == "train" else 100
            new_examples = read_examples_from_file(
                            address, encoding, lang, mode, 
                            seed=args.seed, percentage=percentage
                        ) if examples is None else examples
            guids = []
            for example in new_examples:
                guids.append(
                    int(example.guid.split("-")[-1])
                )
            features = convert_examples_to_features(
                        new_examples, labels, args.max_seq_length, tokenizer, lang,
                        cls_token_at_end=bool(args.model_type in ["xlnet"]),
                        # xlnet has a cls token at the end
                        cls_token=tokenizer.cls_token,
                        cls_token_segment_id=2 if args.model_type in ["xlnet"] else 0,
                        sep_token=tokenizer.sep_token,
                        sep_token_extra=bool(args.model_type in ["roberta"]),
                        pad_on_left=bool(args.model_type in ["xlnet"]),
                        # pad on the left for xlnet
                        pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                        pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
                        pad_token_label_id=pad_token_label_id,
                        logger=logger
                    )
            all_lang_feature[dataset_key] = features
            all_lang_guid[dataset_key] = guids
            
            if args.local_rank in [-1, 0]:
                logger.info("Saving features into cached file {}".format(cached_features_file))
                torch.save(features, cached_features_file)
                with open(cached_features_file+".guid.pkl", "wb") as filePtr:
                    pickle.dump(guids, filePtr, pickle.HIGHEST_PROTOCOL)

    if args.local_rank == 0:
        torch.distributed.barrier()

    # Convert to Tensors and build dataset
    datasets = OrderedDict()
    for address, features in  all_lang_feature.items():
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        datasets[address] = dataset
    
    return datasets, all_lang_guid


