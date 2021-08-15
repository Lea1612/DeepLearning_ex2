* Lea Setruk 
* Yoel Benabou

In order to run the following command lines, you must have a following tree:
* data
    * pos
        * dev
        * train
        * test
    * ner
        * dev
        * train
        * test
    * vocab.txt
    * wordVectors.txt
    
##Additional files

You will find three files in addition to the files you requested.

parse.py : python file containing all the functions we use for parsing the files

NNModel.py : file containing the model for parts 1 to 4

CNNModel.py : file containing the model for part 5

##PART 1 

pos: 
tagger1.py data/pos/train data/pos/dev pos

ner:
tagger1.py data/ner/train data/ner/dev ner

##PART 2
top_k.py data/vocab.txt data/wordVectors.txt

##PART 3

pos:
tagger2.py data/pos/train data/pos/dev pos data/vocab.txt data/wordVectors.txt

ner:
tagger2.py data/ner/train data/ner/dev ner data/vocab.txt data/wordVectors.txt

##PART 4

pos:
tagger3.py data/pos/train data/pos/dev pos data/vocab.txt data/wordVectors.txt

ner:
tagger3.py data/ner/train data/ner/dev ner data/vocab.txt data/wordVectors.txt

##PART 5

pos:
tagger4.py data/pos/train data/pos/dev pos data/vocab.txt data/wordVectors.txt

ner:
tagger4.py data/ner/train data/ner/dev ner data/vocab.txt data/wordVectors.txt
