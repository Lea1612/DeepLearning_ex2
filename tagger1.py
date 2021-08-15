from sys import argv
from torch import cuda
from NNModel import NNModel
from parse import Parse


class Tagger1:
    def __init__(self, is_pos=True):
        if is_pos:
            self.hidden = 300
            self.epoch = 25
            self.batch = 1024
            self.learning_rate = 0.01
            self.test_file = "data/pos/test"

        else:
            self.hidden = 100
            self.epoch = 30
            self.batch = 1024
            self.learning_rate = 0.01
            self.test_file = "data/ner/test"


if __name__ == '__main__':
    device = 'cuda' if cuda.is_available() else 'cpu'

    train_file = argv[1]
    dev_fie = argv[2]
    corpus = argv[3]

    parse = Parse(has_sub_word_information=False)
    tagger1 = Tagger1(corpus == 'pos')

    emb_voc_vec = None
    isEmbedded = False

    print("Parsing file")

    data, tag = parse.load_input_file(train_file, True)
    val_data, val_tag = parse.load_input_file(dev_fie)

    print("Creating Neural network")
    embedded = None

    network = NNModel(output_size=parse.tags_nb,
                      emb_size=len(parse.word_id),
                      w_embed=embedded,
                      hidden_size=tagger1.hidden,
                      features=False).to(device)

    print("Start training phase")

    network = network.train_nn(network,
                               data,
                               tag,
                               batch_size=tagger1.batch,
                               validation_data=val_data,
                               validation_tag=val_tag,
                               learning_rate=tagger1.learning_rate,
                               is_pos=corpus == 'pos',
                               epoch=tagger1.epoch,
                               corpus=corpus,
                               parser=parse,
                               tagger_name='tagger1')

    network.predict_test(tagger1.test_file, 'test1.' + corpus, parse, network, tagger1.batch)
