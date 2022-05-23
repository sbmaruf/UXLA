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
                    default=["./data/en/eng.train", "./data/en/eng.testa", "./data/en/eng.testb"], 
                    help="Address of the files. Value-type: list(str)")
parser.add_argument("--encoding",            
                    default="utf-8", 
                    type=str, 
                    help="The encoding method that will be used to read the texts. Value-type: (str)")
args = parser.parse_args()


def main():
    
    for __file in args.files:
        sentences = read_from_path(__file, args.encoding)
        output_sentences = []
        wrtPtr = open(__file+".sent", "w")
        for sentence in sentences:
            new_sent = ''
            for i, word in enumerate(sentence):
                if i:
                    new_sent = new_sent + ' '
                new_sent = new_sent + word[0]
            wrtPtr.write("{}\n".format(new_sent.strip()))
        wrtPtr.close()

if __name__ == "__main__":
    main()



# python extract_sentences.py --files "./data/de/de.train" --encoding "latin-1"
# python extract_sentences.py --files "./data/en/en.train" "./data/es/es.train" "./data/ar/ar.train" "./data/nl/nl.train" "./data/fi/fi.train"
# python extract_sentences.py --files "./data/de/de.testa" --encoding "latin-1"
# python extract_sentences.py --files "./data/en/en.testa" "./data/es/es.testa" "./data/ar/ar.testa" "./data/nl/nl.testa" "./data/fi/fi.testa"