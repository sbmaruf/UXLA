import os
import sys
import argparse
from utils_ner import read_from_path, \
                      iob2, \
                      iob_iobes, \
                      iobes_iob, \
                      update_tag_scheme


parser = argparse.ArgumentParser(description='Convert conll iob1 dataset to iob2 dataset.')
parser.add_argument("--files",             
                    nargs='*',
                    default=["./ner_data/en/eng.train", "./ner_data/en/eng.testa", "./ner_data/en/eng.testb"], 
                    help="Address of the files. Value-type: list(str)")
parser.add_argument("--encoding",            
                    default="utf-8", 
                    type=str, 
                    help="The encoding method that will be used to read the texts. Value-type: (str)")
parser.add_argument("--lang_dict_address",            
                    default="./lang_dict.txt", 
                    type=str, 
                    help="Exclude the seed value from the experiment. Value-type: (str)")
parser.add_argument("--rename",            
                    action='store_true',
                    help="Rename the language  two char short form  with standard two char short form. Value-type: (bool)")
params = parser.parse_args()


def get_lang_dict(lang_dict_address):
    _dict = {}
    with open(lang_dict_address, "r") as filePtr:
        for line in filePtr:
            lang = line.strip().split()
            assert len(lang) == 2
            _dict[lang[0]] = lang[1] 
    return _dict


# lang_dict = get_lang_dict(params.lang_dict_address)
datasets = params.files
for _file in datasets:
    sentences = read_from_path(_file, params.encoding)
    update_tag_scheme(sentences, 'iob')
    
    # prepare the new file name.
    new_file = _file+".iob2"
    if params.rename:
        for k, v in lang_dict.items():
            if k in new_file:
                new_file = new_file.replace(k, v)

    flag = 0
    with open(new_file, "w", encoding=params.encoding) as filePtr:
        for words in sentences:
            if flag:
                filePtr.write("\n")
            for word in words:
                filePtr.write(word[0]+" "+word[-1]+"\n")
            flag = 1