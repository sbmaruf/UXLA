# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Named entity recognition fine-tuning: utilities to work with CoNLL-2003 task. """

from __future__ import absolute_import, division, print_function

import logging
import os
import copy
from io import open
import subprocess
import numpy as np
import pickle
import random

class InputExample(object):
    """A single training/test example for token classification."""

    def __init__(self, guid, words, labels, lang, orig_words=None, orig_label=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            words: list. The words of the sequence.
            labels: (Optional) list. The labels for each word of the sequence. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.words = words
        self.labels = labels
        self.lang = lang
        self._size = len(labels)
        self.orig_words = orig_words
        self.orig_label = orig_label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_ids, orig_length):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.orig_length = orig_length


def read_examples_from_file(address, encoding, lang, mode, seed, percentage=100):
    guid_index = 1
    examples = []
    if encoding == "pkl":
        with open(address, "rb") as filePtr:
            examples = pickle.load(filePtr)
        for i, example in enumerate(examples):
            for j, l in enumerate(example.labels):
                if l == "X":
                    examples[i].labels[j] = "O"
                
    else:
        with open(address, encoding=encoding) as f:
            words = []
            labels = []
            for line in f:
                if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                    if words:
                        examples.append(InputExample(guid="{}-{}".format(mode, guid_index),
                                                    words=words,
                                                    labels=labels,
                                                    lang=lang))
                        guid_index += 1
                        words = []
                        labels = []
                else:
                    splits = line.split()
                    words.append(splits[0])
                    if len(splits) > 1:
                        if splits[-1].replace("\n", "") == "X":
                            splits[-1] = "O"
                        labels.append(splits[-1].replace("\n", ""))
                    else:
                        # Examples could have no label for mode = "test"
                        labels.append("O")
            if words:
                examples.append(InputExample(guid="{}-{}".format(mode, guid_index),
                                            words=words,
                                            labels=labels,
                                            lang=lang))

    if percentage < 100:
        indices = list(range(len(examples)))
        np.random.seed(seed)
        np.random.shuffle(indices)
        total_num_data_to_be_selected = (len(indices)*percentage)//100
        indices = indices[0:total_num_data_to_be_selected]
        temp_examples = [ examples[__id] for __id in indices]
        examples = copy.deepcopy(temp_examples)
    return examples

def convert_examples_to_features(examples,
                                 label_list,
                                 max_seq_length,
                                 tokenizer,
                                 lang,
                                 cls_token_at_end=False,
                                 cls_token="[CLS]",
                                 cls_token_segment_id=1,
                                 sep_token="[SEP]",
                                 sep_token_extra=False,
                                 pad_on_left=False,
                                 pad_token=0,
                                 pad_token_segment_id=0,
                                 pad_token_label_id=-1,
                                 sequence_a_segment_id=0,
                                 mask_padding_with_zero=True,
                                 logger=None):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """
    label_map = {label: i for i, label in enumerate(label_list)}
    features = []
    over_length_sentence = 0
    for (ex_index, example) in enumerate(examples):
        if ex_index % 1000 == 0:
            logger.info("Writing example {}: {} of {}".format(lang, ex_index, len(examples)))
            
        tokens = []
        label_ids = []
        for word, label in zip(example.words, example.labels):
            word_tokens = tokenizer.tokenize(word)
            if len(word_tokens) == 0 and len(label) == 1 :
                word_tokens.append("[UNK]")
            if len(word_tokens) == 0 and len(label) > 1 :
                continue
                raise ValueError
            tokens.extend(word_tokens)
            # Use the real label id for the first token of the word, and padding ids for the remaining tokens
            label_ids.extend([label_map[label]] + [pad_token_label_id] * (len(word_tokens) - 1))
        
        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        special_tokens_count = 3 if sep_token_extra else 2
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[:(max_seq_length - special_tokens_count)]
            label_ids = label_ids[:(max_seq_length - special_tokens_count)]
            logger.warning("Out of Maximum sequence length ({}) - : {}".format(max_seq_length, len(label_ids)))
            over_length_sentence += 1

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens += [sep_token]
        label_ids += [pad_token_label_id]
        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [sep_token]
            label_ids += [pad_token_label_id]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if cls_token_at_end:
            tokens += [cls_token]
            label_ids += [pad_token_label_id]
            segment_ids += [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            label_ids = [pad_token_label_id] + label_ids
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            label_ids = ([pad_token_label_id] * padding_length) + label_ids
        else:
            input_ids += ([pad_token] * padding_length)
            input_mask += ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids += ([pad_token_segment_id] * padding_length)
            label_ids += ([pad_token_label_id] * padding_length)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length

        if ex_index < 2:
            logger.info("*** Example ***")
            logger.info("guid: %s", example.guid)
            logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
            logger.info("label_ids: %s", " ".join([str(x) for x in label_ids]))

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_ids=label_ids,
                              orig_length=example._size))
    if over_length_sentence > 0:
        logger.warning("Total over length sentences : {}".format(over_length_sentence))
    return features




def get_labels(path):
    if path:
        with open(path, "r") as f:
            labels = f.read().splitlines()
        if "O" not in labels:
            labels = ["O"] + labels
        return labels
    else:
        return ["O", "B-MISC", "I-MISC",  "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]



def read_from_path(address, encoding='utf-8', percentage=100, seed=1234):
    """
    Read the data from the source address and return the sentences.
    a small bug: a special case where the first line does not load.
    todo list: as it doesn't affect the model too much, it will be
    updated later.
    """
    sentences = []
    sentence = []
    for line in open(address, 'r', encoding=encoding):  # use latin-1 if you want to take german.
        line = line.rstrip()
        if not line:
            if len(sentence) > 0:
                if 'DOCSTART' not in sentence[0][0]:
                    sentences.append(sentence)
                sentence = []
        else:
            word = line.split()
            sentence.append(word)
            # assert len(word) >= 2
    if len(sentence) > 0:
        if 'DOCSTART' not in sentence[0][0]:
            sentences.append(sentence)
    
    if percentage < 100:
        indices = list(range(len(sentences)))
        np.random.seed(seed)
        np.random.shuffle(indices)
        total_num_data_to_be_selected = (len(indices)*percentage)//100
        indices = indices[0:total_num_data_to_be_selected]
        temp_sentences = [ sentences[__id] for __id in indices]
        sentences = copy.deepcopy(temp_sentences)

    return sentences


def write_data(path, file_info, predictions, logger=None):
    file_name, encoding, lang = os.path.split(file_info.split(";")[0])[-1], file_info.split(";")[1], file_info.split(";")[2]
    address = os.path.join(path, file_name+".pred")
    logger.info("Pseudo prediction writting on file {}".format(address))
    writePtr = open(address, "w")
    original_file_address = file_info.split(";")[0]
    # print(original_file_address)
    sentences = read_from_path(original_file_address, encoding=encoding)
    # print("read sentences {}".format(len(sentences)))
    # print("predicted sentences {}".format(len(predictions)))
    assert len(sentences) == len(predictions)
    for original_sent_index, (sentence_lable, prediction) in enumerate(zip(sentences, predictions)):
        for idx, (word, lable) in enumerate(sentence_lable):
            pseudo_prediction = prediction[idx]
            writePtr.write("{} {} {}\n".format(word, lable, pseudo_prediction))
        writePtr.write("\n")
    writePtr.close()
    cmd = 'scripts/conlleval_ar < "{}"'.format(address)
    logger.info("running command : {}".format(cmd))
    out = subprocess.check_output(cmd, shell=True)
    logger.info(out.decode())
    return address + ";" + encoding + ";" + lang
        

def select_and_write_data(dict_key, path, indices, examples, train_loss_dict, labels, seed, mode = "train", logger=None, postfix=None):
    logger.info("Sample selection from :: {}".format(dict_key))
    data_repo_list = []

    file_name = os.path.split(dict_key.split(";")[0])[-1] + ".pred." + str(postfix)
    address = os.path.join(path, file_name)
    encoding = dict_key.split(";")[1]
    lang = dict_key.split(";")[2]
    
    if encoding == "pkl" and lang == "de":
        encoding="latin-1"
    elif encoding == "pkl" and lang != "de":
        encoding="utf-8"

    label_map = {i: label for i, label in enumerate(labels)}
    logger.info("writting on : {}".format(address))
    ignored_sent_cnt = 0
    ret_indices = []
    with open(address, "w", encoding=encoding) as writePtr:
        for idx in indices:
            example = examples[idx]
            assert len(example.words) == len(example.labels)
            sent = []
            for w_0, w_1 in zip(example.words, example.labels):
                sent.append([w_0, w_1])
            total_len = len(train_loss_dict[dict_key][idx][-1])
            try:
                assert len(sent) == total_len
            except:
                logger.warning("Sentence length doesn't match. original len: {} dict_len: {}, ignoring sentence ...".format(len(sent), total_len))
                ignored_sent_cnt += 1
                continue
            for word_idx in range(total_len):
                try:
                    writePtr.write("{} {} {}\n".format(sent[word_idx][0], sent[word_idx][1], labels[ train_loss_dict[dict_key][idx][-1][word_idx] ] ))
                except UnicodeEncodeError:
                    pass
                except:
                    raise
            writePtr.write("\n")
            ret_indices.append(idx)
    logger.info("Sample written on :: {}".format(address))
    if ignored_sent_cnt:
        logger.warning("total ignored sentence : {}".format(ignored_sent_cnt))
    new_file_format = address+";"+encoding+";"+lang
    return new_file_format, ret_indices


def select_and_write_source_data(dict_key, path, indices, orig_lable_bank, labels, seed, percentage=100, logger=None, postfix=None):
    logger.info("Sample selection from :: {}".format(dict_key))
    data_repo_list = []
    # "data/es/es.train.iob2;utf-8;es"
    file_name = os.path.split(dict_key.split(";")[0])[-1] + ".pred." + str(postfix)
    address = os.path.join(path, file_name)
    encoding = dict_key.split(";")[1]
    lang = dict_key.split(";")[2]
    sentences = read_from_path(dict_key.split(";")[0], encoding=encoding, percentage=percentage, seed=seed)
    label_map = {i: label for i, label in enumerate(labels)}
    with open(address, "w", encoding=encoding) as writePtr:
        for idx in indices:
            sent = sentences[idx]
            total_len = len(orig_lable_bank[dict_key][idx])
            for word_idx in range(total_len):
                writePtr.write("{} {}\n".format(sent[word_idx][0], labels[ orig_lable_bank[dict_key][idx][word_idx] ] ))
            writePtr.write("\n")
    logger.info("Sample written on :: {}".format(address))
    new_file_format = address+";"+encoding+";"+lang
    return new_file_format


def manual_check_nl(path, predictions, logger):
    def write_res(address):
        with open(address, "w") as filePtr:
            assert len(sentences) == len(predictions)
            for sentence, preds in zip(sentences, predictions):
                for word, pred in zip(sentence, preds):
                    filePtr.write("{} {} {}\n".format(word[0], word[1], pred))
                filePtr.write("\n")
    sentences = read_from_path("./data/nl/nl.testb.iob2")
    # print(len(read_examples_from_file("./data/nl/nl.testb.iob2", "utf-8", "nl", "test")))
    # print(len(read_examples_from_file("./data/temp/nl.testb.iob2.pred", "utf-8", "nl", "test")))
    new_file_address = os.path.join(path, "nl.testb.iob2.join.pred")
    write_res(new_file_address)
    cmd = "sed -e '27173d' {} > {}.1".format(new_file_address, new_file_address)
    subprocess.check_output(cmd, shell=True)
    # print(len(read_examples_from_file("./data/temp/nl.testb.iob2.pred.1", "utf-8", "nl", "test")))
    cmd = "sed -e '27344d' {}.1 > {}.2".format(new_file_address, new_file_address)
    subprocess.check_output(cmd, shell=True)
    # print(len(read_examples_from_file("./data/temp/nl.testb.iob2.pred.2", "utf-8", "nl", "test")))
    cmd = 'scripts/conlleval_ar < "{}.2"'.format(new_file_address)
    logger.info("running command : {}".format(cmd))
    out = subprocess.check_output(cmd, shell=True)
    logger.info(out.decode())


def select_and_write_logits(dict_key, path, indices, logit_dict, mode, labels, logger=None):
    logger.info("Sample selection from :: {}".format(dict_key))
    # "data/es/es.train.iob2;utf-8;es"
    file_name = os.path.split(dict_key.split(";")[0])[-1] + ".logit"
    address = os.path.join(path, file_name)
    
    id_2_label = { idx: l for idx, l in enumerate(labels) }
    clustered_logit = {"PER":None , "O":None, "ORG":None, "MISC":None, "LOC":None}
    all_logits = None
    cnt = 0
    for idx in indices:
        sentence_logits = logit_dict[dict_key][idx]
        all_logits = sentence_logits if all_logits is None else np.append(all_logits, sentence_logits, axis=0)    
        label_ids = np.argmax(sentence_logits, axis = -1)
        for idx, label_id in enumerate(label_ids):
            tag = id_2_label[label_id].split("-")[-1]
            temp_logits = np.expand_dims(sentence_logits[idx], axis=0)
            clustered_logit[ tag ] = temp_logits if clustered_logit[ tag ] is None else  \
                                            np.append(clustered_logit[ tag ], temp_logits, axis = 0)
    logit_bank = [all_logits, clustered_logit]
    with open(address, 'wb') as filePtr:
        pickle.dump(logit_bank, filePtr, protocol=pickle.HIGHEST_PROTOCOL)

    return address, dict_key



def iob2(tags):
    """
    Check that tags have a valid IOB format.
    Tags in IOB1 format are converted to IOB2.
    """
    for i, tag in enumerate(tags):
        if tag == 'O':
            continue
        split = tag.split('-')
        if len(split) != 2 or split[0] not in ['I', 'B']:
            return False
        if split[0] == 'B':
            continue
        elif i == 0 or tags[i - 1] == 'O':    # conversion IOB1 to IOB2
            tags[i] = 'B' + tag[1:]
        elif tags[i - 1][1:] == tag[1:]:
            continue
        else:  # conversion IOB1 to IOB2
            tags[i] = 'B' + tag[1:]
    return True


def iob_iobes(tags):
    """
    IOB -> IOBES
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if tag == 'O':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'B':
            if i + 1 != len(tags) and \
                            tags[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('B-', 'S-'))
        elif tag.split('-')[0] == 'I':
            if i + 1 < len(tags) and \
                            tags[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('I-', 'E-'))
        else:
            raise Exception('Invalid IOB format!')
    return new_tags


def iobes_iob(tags):
    """
    IOBES -> IOBs
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if tag.split('-')[0] == 'B':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'I':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'S':
            new_tags.append(tag.replace('S-', 'B-'))
        elif tag.split('-')[0] == 'E':
            new_tags.append(tag.replace('E-', 'I-'))
        elif tag.split('-')[0] == 'O':
            new_tags.append(tag)
        else:
            raise Exception('Invalid format!')
    return new_tags


def update_tag_scheme(sentences, tag_scheme):
    """
    Check and update sentences tagging scheme to IOB2.
    Only IOB1 and IOB2 schemes are accepted.
    """
    for i, s in enumerate(sentences):
        tags = [w[-1] for w in s]
        # Check that tags are given in the IOB format
        if not iob2(tags):
            s_str = '\n'.join(' '.join(w) for w in s)
            raise Exception('Sentences should be given in IOB format! ' +
                            'Please check sentence %i:\n%s' % (i, s_str))
        if tag_scheme == 'iob':
            # If format was IOB1, we convert to IOB2
            for word, new_tag in zip(s, tags):
                word[-1] = new_tag
        elif tag_scheme == 'iobes':
            new_tags = iob_iobes(tags)
            for word, new_tag in zip(s, new_tags):
                word[-1] = new_tag
        else:
            raise Exception('Unknown tagging scheme!')
        
        
def write_conll_data(file_info, sent_info, logger=None):
    logger.info("writting data on {}".format(file_info))
    address, encoding, lang = file_info.split(";")[0], file_info.split(";")[1], file_info.split(";")[2]
    with open(address, "w", encoding=encoding) as filePtr:
        for sent in sent_info:
            for w_info in sent:
                f = 0
                for w in w_info:
                    if f:
                        filePtr.write(" ")
                    f = 1
                    filePtr.write("{}".format(w))
                filePtr.write("\n")
            filePtr.write("\n")
    return file_info

