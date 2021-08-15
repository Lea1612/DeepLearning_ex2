from sys import argv
from torch import cuda

from CNNModel import CNNModel
from parse import Parse


class Tagger4:
    def __init__(self, is_pos=True):
        if is_pos:
            self.hidden = 200
            self.epoch = 25
            self.batch = 32
            self.learning_rate = 0.001
            self.test_file = "data/pos/test"

        else:
            self.hidden = 200
            self.epoch = 25
            self.batch = 32
            self.learning_rate = 0.001
            self.test_file = "data/ner/test"

        self.filters = 40
        self.window_conv_size = 3


if __name__ == '__main__':
    device = 'cuda' if cuda.is_available() else 'cpu'

    train_file = argv[1]
    dev_fie = argv[2]
    corpus = argv[3]
    emb_voc = argv[4]
    emb_voc_vec = argv[5]

    parse = Parse(vocab_file=emb_voc, word_vectors_file=emb_voc_vec, is_embedded=True, is_cnn=True)
    tagger4 = Tagger4(corpus == 'pos')

    print("Parsing file")

    parse.load_chars(train_file)
    data, tag = parse.load_input_file(train_file, True)
    val_data, val_tag = parse.load_input_file(dev_fie)

    print("Creating Neural network : ")

    network = CNNModel(hidden_size=tagger4.hidden,
                       number_of_chars=parse.number_of_chars,
                       char_2_id=parse.char_2_id,
                       number_of_labels=parse.tags_nb,
                       word_max_len=parse.char_max_len,
                       batch_size=tagger4.batch,
                       word_vectors_file=emb_voc_vec,
                       pre_trained_vocab_size=parse.pre_trained_size,
                       embedded_char_dimension=30,
                       embedded_pre_trained_dimension=50,
                       window_size=5,
                       window_conv_size=tagger4.window_conv_size,
                       filters=tagger4.filters
                       ).to(device)

    print("Start training phase")
    network = network.train_nn(network,
                               data,
                               tag,
                               batch_size=tagger4.batch,
                               validation_data=val_data,
                               validation_tag=val_tag,
                               learning_rate=tagger4.learning_rate,
                               is_pos=corpus == 'pos',
                               epoch=tagger4.epoch,
                               corpus=corpus,
                               parser=parse,
                               tagger_name='tagger4'
                               )

    network.predict_test(tagger4.test_file, 'test5.' + corpus, parse, network, tagger4.batch)
