from sys import argv
import numpy as np
from torch import cuda
from NNModel import NNModel
from parse import Parse


class Tagger3:
    def __init__(self, is_pos=True):
        if is_pos:
            self.hidden = 500
            self.epoch = 25
            self.batch = 1024
            self.learning_rate = 0.01
            self.test_file = "data/pos/test"

        else:
            self.hidden = 100
            self.epoch = 25
            self.batch = 1024
            self.learning_rate = 0.01
            self.test_file = "data/ner/test"


if __name__ == '__main__':
    device = 'cuda' if cuda.is_available() else 'cpu'

    train_file = argv[1]
    dev_fie = argv[2]
    corpus = argv[3]
    emb_voc = argv[4]
    emb_voc_vec = argv[5]

    parse = Parse(vocab_file=emb_voc, word_vectors_file=emb_voc_vec, has_sub_word_information=True, is_embedded=True)
    tagger3 = Tagger3(corpus == 'pos')

    print("Parsing file")

    data, tag = parse.load_input_file(train_file, True)
    val_data, val_tag = parse.load_input_file(dev_fie)

    print("Creating Neural network")
    embedded = np.array(list(parse.word_dict.values()))

    network = NNModel(output_size=parse.tags_nb,
                      emb_size=len(parse.word_id),
                      w_embed=embedded,
                      hidden_size=tagger3.hidden,
                      features=True).to(device)

    print("Start training phase")
    network = network.train_nn(network,
                               data,
                               tag,
                               batch_size=tagger3.batch,
                               validation_data=val_data,
                               validation_tag=val_tag,
                               learning_rate=tagger3.learning_rate,
                               is_pos=corpus == 'pos',
                               epoch=tagger3.epoch,
                               corpus=corpus,
                               parser=parse,
                               tagger_name='tagger3'
                               )

    network.predict_test(tagger3.test_file, 'test4.' + corpus, parse, network, tagger3.batch)
