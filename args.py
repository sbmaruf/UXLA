import os
import argparse


def data_params(parser):
    group = parser.add_argument_group('Dataset params.')
    group.add_argument("--train",
                        nargs='*',
                        default=["data/en/en.train.iob2;utf-8;en", 
                                 "data/es/es.train.iob2;utf-8;es", 
                                 "data/de/de.train.iob2;latin-1;de", 
                                 "data/nl/nl.train.iob2;utf-8;nl", 
                                 "data/ar/ar.train.iob2;utf-8;ar", 
                                 "data/fi/fi.train.iob2;utf-8;fi"], 
                        help="Train set location. Value-type: list(string)")
    group.add_argument("--dev",
                        nargs='*',
                        default=["data/en/en.testa.iob2;utf-8;en", 
                                 "data/es/es.testa.iob2;utf-8;es", 
                                 "data/de/de.testa.iob2;latin-1;de", 
                                 "data/nl/nl.testa.iob2;utf-8;nl", 
                                 "data/ar/ar.testa.iob2;utf-8;ar", 
                                 "data/fi/fi.testa.iob2;utf-8;fi"], 
                        help="Validation set location. Value-type: list(string)")
    group.add_argument("--test",
                        nargs='*',
                        default=["data/en/en.testb.iob2;utf-8;en", 
                                 "data/es/es.testb.iob2;utf-8;es", 
                                 "data/de/de.testb.iob2;latin-1;de", 
                                 "data/nl/nl.testb.iob2;utf-8;nl", 
                                 "data/ar/ar.testb.iob2;utf-8;ar", 
                                 "data/fi/fi.testb.iob2;utf-8;fi"], 
                        help="Validation set location. Value-type: list(string)")
    group.add_argument("--logit_bank_address_keys",
                        nargs='*',
                        default=[None], 
                        help="Validation set location. Value-type: list(string)")
    group.add_argument("--external_data",
                        nargs='*',
                        default=["dumped/lm_aug_pickle/dataset-data_de_de_train_iob2_latin-1_de-aug_type-successive_max-aug_per-10-num_of_aug-1-only_ner_aug-0-topk-1-train_data_percentage-100-seed-1234/de.train.iob2.aug;pkl;de"], 
                        help="Validation set location. Value-type: list(string)")
    group.add_argument("--src_lang",
                       default="en", 
                       type=str, 
                       help="Name of the source language. Value-type: (str)")
    group.add_argument("--dev_lang",
                       default="en", 
                       type=str, 
                       help="Name of the development language (model tuned by this language dev. set). Value-type: (str)")
    group.add_argument("--tgt_lang",
                       default="en", 
                       type=str, 
                       help="Name of the tgt language. Value-type: (str)")
    group.add_argument("--aug_lang",
                       default="en;es;de;nl;ar;fi", 
                       type=str, 
                       help="Augmented language. Value-type: (str)")
    group.add_argument("--aug_label_propagate",
                       default="en", 
                       type=str, 
                       help="Which lang label will be propagated. Value-type: (str)")
    group.add_argument("--lang_alpha",
                       default=.5, 
                       type=float, 
                       help="Name of the tgt language. Value-type: (str)")
    parser.add_argument("--do_lower_case", 
                       default=0,
                       type=int,
                       help="Do we lowercase the dataset.")
    group.add_argument("--label",
                       default=None, 
                       type=str, 
                       help="Path where label file is saved.")


def model_params(parser):
    group = parser.add_argument_group('Model params.')
    group.add_argument("--model_type", 
                        default="bert", 
                        type=str, 
                        help="Model type selected in the list")
    group.add_argument("--model_name_or_path", 
                        default="bert-base-multilingual-cased", 
                        type=str, 
                        help="Path to pre-trained model or shortcut name selected in the list")
    group.add_argument("--thetas", 
                        nargs='*',
                        default=[], 
                        help="Different types of pretrained model path. Value-type: list(string)")
    group.add_argument("--config_name", 
                        default="", 
                        type=str,
                        help="Pretrained config name or path if not the same as model_name")
    group.add_argument("--tokenizer_name", 
                        default="bert-base-multilingual-cased", 
                        type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    group.add_argument("--output_dir", 
                        default="./dumped", 
                        type=str, 
                        help="The output directory where the model predictions and checkpoints will be written.")
    group.add_argument("--cache_dir", 
                        default="", 
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    group.add_argument("--dropout", 
                        default=.1, 
                        type=float,
                        help="Dropout value of the hidden representation of the LM.")
    group.add_argument("--max_seq_length", default=280, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    group.add_argument("--num_of_heads", 
                        default=1, 
                        type=int,
                        help="Total number of head on top of pretrained LM.")


def noise_model_params(parser):
    group = parser.add_argument_group('Noise model params.')
    group.add_argument("--noise_threshold", 
                       default=0, type=int,
                       help="Maximum number of wrong labels can exists in a correct sentence (for debugging purpose).")
    group.add_argument("--n_mixture_component", 
                        default=5, 
                        type=int,
                        help="Number of component in noise model.")
    group.add_argument("--n_mixture_select", 
                        default=1, 
                        type=int,
                        help="Number of mixture component data to be selected.")
    group.add_argument("--posterior_threshold", 
                        nargs='*',
                        default=[.7], 
                        type=float,
                        help="Posterior probability threshold.")
    group.add_argument("--covariance_type", 
                        default="full", 
                        type=str,
                        help="Covariance type.",
                        choices=["full", "diag", "tied", "spherical"])
    group.add_argument("--min_length_restriction", 
                        default=0,
                        type=int,
                        help="Minimum length of the sentences choose to do training (for debugging purpose).")
    group.add_argument("--max_length_restriction", 
                        default=150, 
                        type=int,
                        help="Maximum length of the sentences choose to do training (for debugging purpose).")
    group.add_argument("--aug_mode", 
                        default="train", 
                        type=str,
                        help="Pseudo label model. From where the dataset will be read. "
                             "`aug` will read from `--external_data` parameters.",
                        choices=["train", "dev", "aug"])
    group.add_argument("--aug_desc", 
                        default="0:src;src_aug;tgt_self;tgt_aug|1:src;src_aug;tgt_self;tgt_aug", 
                        type=str,
                        help="Source or target or both augmentation seperated by ;")
    



def logistics_params(parser):
    group = parser.add_argument_group('Logistics params.')
    group.add_argument("--do_train", 
                        action="store_true",
                        help="Whether to run training.")
    group.add_argument("--process_augmentation", 
                        action="store_true",
                        help="Infer training dataset.")
    group.add_argument("--do_eval", 
                        action="store_true",
                        help="Whether to run eval on the dev set.")
    group.add_argument("--do_ensamble_eval", 
                        action="store_true",
                        help="Whether to run eval on the dev set.")
    group.add_argument("--ensamble_type", 
                        default="logit",
                        type=str, 
                        help="Logit level ensamble or output level ensamble.",
                        choices=["logit", "output"])
    group.add_argument("--lazy_inference", 
                        action="store_true",
                        help="Do lazy inference.")
    group.add_argument("--export_pseudo_data", 
                        action="store_true",
                        help="Whether to run predictions on the test set.")
    group.add_argument("--evaluate_during_training", 
                        action="store_true",
                        help="Whether to run evaluation during training at each logging step.")
    group.add_argument("--per_gpu_train_batch_size", 
                        type=int, 
                        default=4,
                        help="Batch size per GPU/CPU for evaluation.")
    group.add_argument("--per_gpu_eval_batch_size", 
                        type=int, 
                        default=32,
                        help="Batch size per GPU/CPU for evaluation.")
    group.add_argument("--gradient_accumulation_steps", 
                        type=int, 
                        default=4,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    group.add_argument("--logging_steps", 
                        type=int, 
                        default=50,
                        help="Log every X updates steps.")
    group.add_argument("--save_steps", 
                        type=int, 
                        default=50,
                        help="Save checkpoint every X updates steps.")
    group.add_argument("--eval_all_checkpoints", 
                        action="store_true",
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    group.add_argument("--no_cuda", 
                        action="store_true",
                        help="Avoid using CUDA when available")
    group.add_argument("--overwrite_output_dir", 
                        action="store_true",
                        help="Overwrite the content of the output directory")
    group.add_argument("--overwrite_cache", 
                        action="store_true",
                        help="Overwrite the cached training and evaluation sets")
    group.add_argument("--seed", 
                        type=int, 
                        default=42,
                        help="random seed for initialization")
    group.add_argument("--logit_dict_cache_address", 
                        type=str, 
                        default="logit_dict",
                        help="Name of the logit_dict_cache")
    

def training_params(parser):
    group = parser.add_argument_group('Training params.')
    group.add_argument("--learning_rate", 
                        default=2e-5, 
                        type=float,
                        help="The initial learning rate for Adam.")
    group.add_argument("--weight_decay", 
                        default=0.01, 
                        type=float,
                        help="Weight decay if we apply some.")
    group.add_argument("--adam_epsilon", 
                        default=1e-8, 
                        type=float,
                        help="Epsilon for Adam optimizer.")
    group.add_argument("--max_grad_norm", 
                        default=1.0, 
                        type=float,
                        help="Max gradient norm.")
    group.add_argument("--num_train_epochs", 
                        default=3, 
                        type=int,
                        help="Total number of training epochs to perform.")
    group.add_argument("--semi_sup_start_epoch", 
                        default=0, 
                        type=int,
                        help="Starting semi sup epoch.")
    group.add_argument("--max_steps", 
                        default=22000, 
                        type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    group.add_argument("--semi_sup_max_steps", 
                        default=-1,
                        type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    group.add_argument("--warmup_steps", 
                        default=2200, 
                        type=int,
                        help="Linear warmup over percentage of batch sample.")
    # group.add_argument("--warmup_percentage", 
    #                     default=-1, 
    #                     type=int,
    #                     help="Percentage of training sample that will be used for warmup.")
    group.add_argument("--train_data_percentage", 
                        default=100, 
                        type=int,
                        help="Percentage of training data that will be selected.")
    group.add_argument("--lam", 
                        default=.5, 
                        type=float,
                        help="Lambda for for augmented loss.")
    group.add_argument("--k_size", 
                        default=3, 
                        type=int,
                        help="Integer size of KNN.")
    group.add_argument("--logit_bank_type", 
                        default="non-clustered", 
                        type=str,
                        help="Type of logit bank.")
    group.add_argument("--penalty", 
                        default=0, 
                        type=int,
                        help="Add a negative NegEntropy term with loss.")
    group.add_argument("--alpha_schedule", 
                        default="fixed", 
                        type=str,
                        help="If alphas value is choosen randomly.",
                        choices=["fixed", "random", "bionomial"])


def dist_params(parser):
    group = parser.add_argument_group('Distributed params.')
    group.add_argument("--fp16", 
                        action="store_true",
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    group.add_argument("--fp16_opt_level", 
                        type=str, 
                        default="O1",
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    group.add_argument("--local_rank", 
                        type=int, 
                        default=-1,
                        help="For distributed training: local_rank")
    group.add_argument("--server_ip", 
                        type=str, 
                        default="", 
                        help="For distant debugging.")
    group.add_argument("--server_port", 
                        type=str, 
                        default="", 
                        help="For distant debugging.")


def semi_sup_params(parser):
    group = parser.add_argument_group("Semi supervised params.")
    group.add_argument("--do_semi_sup_training", 
                        action="store_true",
                        help="Whether to do semi-supervised run training.")
    group.add_argument("--semi_sup_type",
                        default="classical",
                        help="Type of the semi supervised learning.")
    group.add_argument("--top_k",
                        default=50,
                        type=int,
                        help="Top k\% of confident data.")
    group.add_argument("--top_k_increment",
                        default=10,
                        type=int,
                        help="Top k\% increment of each semi_sup_epoch.")
    group.add_argument("--max_semi_sup_epoch",
                        default=3,
                        type=int,
                        help="Maximum number of semi-sup epoch.")
    group.add_argument("--retrain",
                        default=0,
                        type=int,
                        help="If the training starts from begining or not.")
    group.add_argument("--partial_train_in_semi_sup_epochs",
                        default=0,
                        type=int,
                        help="IIf activated, in first epoch only source language will be trained and in the last epoch tgt lang will be trained.")
    group.add_argument("--data_distil_type",
                        default='top_k',
                        type=str,
                        help="Data Distil type.")
    group.add_argument("--merge_datasets",
                        default=0,
                        type=int,
                        help="Merge all datasets.")
    group.add_argument("--agreement_param",
                        default=1,
                        type=int,
                        help="1: intersec of prediction 2. union of prediction",
                        choices=[1, 2])

def load_args():
    
    parser = argparse.ArgumentParser("Cross-lingual Contextual NER.")
    
    data_params(parser)
    model_params(parser)
    noise_model_params(parser)
    logistics_params(parser)
    training_params(parser)
    dist_params(parser)
    semi_sup_params(parser)
    args = parser.parse_args()
    args = args
    args.learning_rate = float(args.learning_rate)
    if os.path.exists(args.output_dir) \
        and len(os.listdir(args.output_dir))>1 \
            and args.do_train \
                and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))
    return args
