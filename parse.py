import numpy as np
from torch import LongTensor, FloatTensor
from torch import cuda

device = 'cuda' if cuda.is_available() else 'cpu'

START_TAG = '<s>'
END_TAG = '</s>'
UNKNOW_WORD = 'UUUNKKK'
UNKNOW_CHAR = 'UUUNKKK_CHAR'
PADDING_CHAR = 'PAD'


class Parse(object):
    def __init__(self, vocab_file=None, word_vectors_file=None, has_sub_word_information=False, is_embedded=False,
                 lower_case=False, embedding_size=50, is_cnn=False):
        if vocab_file is not None and word_vectors_file is not None:
            self.word_dict = self.get_word_dict(vocab_file, word_vectors_file)
            self.word_id = self.get_word_id(vocab_file)
            self.pre_trained_size = len(self.word_id)
        else:
            self.word_id = {UNKNOW_WORD: 0}

        self.chars = {}
        self.char_dict = {}
        self.number_of_chars = 0
        self.id_2_tag = self.tag_2_id = None
        self.id_2_char = self.char_2_id = None
        self.tags_nb = 0
        self.has_sub_word_information = has_sub_word_information
        self.isEmbedded = is_embedded
        self.lower_case = lower_case
        self.embedding_size = embedding_size
        self.is_cnn = is_cnn
        self.char_max_len = 0

    def get_word_dict(self, vocab_file, word_vectors_file):
        word_dict = {}
        for word, word_vector in zip(open(vocab_file).read().split('\n'), open(word_vectors_file).read().split('\n')):
            if len(word_vector.split()) == 50:
                word_dict[word] = list(map(float, word_vector.split()))
        return word_dict

    def get_word_id(self, vocab_file):
        word_id_dict = {}
        for index_word, word in enumerate(open(vocab_file).read().split('\n')):
            if len(word) != 0:
                word_id_dict[word] = index_word
        return word_id_dict

    def load_input_file(self, data_file, training=False):
        data_windows = []
        tags = []
        tags_set = set()
        index = max(self.word_id.values()) + 1

        if training and len(self.word_id) < 2:
            self.word_id[START_TAG] = 1
            self.word_id[END_TAG] = 2
            index = 3
        end_id, start_id = self.get_start_end_id()
        start = [start_id, start_id]
        end = [end_id, end_id]
        max_word_size = self.char_max_len

        with open(data_file) as f:
            sentence = []
            for line in f:
                l = line.rsplit()
                if len(l) != 0:
                    word_old, tag = l

                    if not self.is_cnn:
                        word, index = self.get_word_index_to_insert(word_old, index, training)

                    if self.has_sub_word_information:
                        self.incorporate_sub_words(index, sentence, training, word, word_old)
                    elif self.is_cnn:
                        self.chars_embedding(max_word_size, sentence, word_old)
                    else:
                        sentence.append(self.word_id[word])
                    tags.append(tag)

                    if training:
                        tags_set.add(tag)

                # empty line
                else:
                    window = self.extract_window(start, sentence, end)
                    data_windows.append([item for item in window])
                    sentence = []

        if training:
            self.id_2_tag = {i: lbl for i, lbl in enumerate(tags_set)}
            self.tag_2_id = {lbl: i for i, lbl in self.id_2_tag.items()}
            self.tags_nb = len(self.id_2_tag)

        data_windows = self.transform_data_windows_to_tensors(data_windows)
        tags = self.transform_tags_to_tensors(tags)
        return data_windows, tags

    def extract_window(self, start, sentence, end):
        full_sentence = start + sentence + end
        return zip(full_sentence, full_sentence[1:], full_sentence[2:], full_sentence[3:], full_sentence[4:])

    def load_test_file(self, input_file):
        data = []
        text = []
        end_id, start_id = self.get_start_end_id()
        start = [start_id, start_id]
        end = [end_id, end_id]
        max_word_size = self.char_max_len

        with open(input_file) as f:
            sentence = []
            sentence_words = []
            for line in f:
                l = line.rsplit()
                if len(l) != 0:
                    word = word_old = l[0]

                    if self.lower_case:
                        word = word.lower()

                    if word not in self.word_id:
                        word = UNKNOW_WORD

                    if self.has_sub_word_information:
                        self.incorporate_sub_words(0, sentence, False, word, word_old)
                    elif self.is_cnn:
                        self.chars_embedding(max_word_size, sentence, word_old)
                    else:
                        sentence.append(self.word_id[word])
                    sentence_words.append(word_old)

                else:
                    window = self.extract_window(start, sentence, end)
                    data.append([item for item in window])
                    text.append(sentence_words)
                    sentence = []
                    sentence_words = []

        data = self.transform_data_windows_to_tensors(data)
        return data, text

    def transform_data_windows_to_tensors(self, data_windows):
        transform = np.array([ngram for sen in data_windows for ngram in sen])
        data_windows = LongTensor(transform)
        return data_windows

    def transform_tags_to_tensors(self, tags):
        tags = [float(self.tag_2_id[lbl]) for lbl in tags]
        tags = LongTensor(tags)
        return tags

    def transform_embedded_data_windows_to_tensors(self, data_windows):
        transform = np.array([sen for sen in data_windows])
        data_windows = FloatTensor(transform)
        return data_windows

    def incorporate_sub_words(self, index, sentence, training, word, word_old):
        pref, index = self.get_word_index_to_insert('$' + word_old[:3], index, training)
        suff, index = self.get_word_index_to_insert(word_old[-3:] + '$', index, training)
        sentence.append([self.word_id[word],
                         self.word_id[pref],
                         self.word_id[suff]])

    def chars_embedding(self, max_word_size, sentence, word):
        word_arr = []
        for c in word:
            if c in self.char_2_id:
                word_arr.append(self.char_2_id[c])
            else:
                word_arr.append(self.char_2_id[UNKNOW_CHAR])

        word_diff = max_word_size - len(word)
        if word_diff > 0:
            for i in range(word_diff):
                word_arr.append(self.char_2_id[PADDING_CHAR])

        if word in self.word_dict:
            word_index = self.word_id[word]
        else:
            word_index = self.word_id[UNKNOW_WORD]

        word_arr.append(word_index)
        sentence.append(word_arr)

    def get_start_end_id(self):
        if self.has_sub_word_information:
            start_id = [self.word_id[START_TAG], self.word_id[START_TAG], self.word_id[START_TAG]]
            end_id = [self.word_id[END_TAG], self.word_id[END_TAG], self.word_id[END_TAG]]
        elif self.is_cnn:
            start_id = [self.word_id[START_TAG] for i in range(self.char_max_len + 1)]
            end_id = [self.word_id[END_TAG] for i in range(self.char_max_len + 1)]
        else:
            start_id = self.word_id[START_TAG]
            end_id = self.word_id[END_TAG]
        return end_id, start_id

    def get_word_index_to_insert(self, word, index=0, to_train=False):
        if self.lower_case:
            word = word.lower()

        if word not in self.word_id:
            if not to_train:
                word = UNKNOW_WORD
            else:
                self.word_id[word] = index
                if self.isEmbedded:
                    self.word_dict[word] = np.random.uniform(-1, 1, self.embedding_size)
                index += 1

        return word, index

    def load_chars(self, data_file):
        self.chars = {PADDING_CHAR, UNKNOW_CHAR}
        chars_size = []
        with open(data_file) as f:
            lines = f.readlines()

        for line in lines:
            if line != "" and line != "\n":
                word, _ = line.strip().rsplit()
                self.chars.update([c for c in word])
                chars_size.append(len(word))

        self.number_of_chars = len(self.chars)
        self.char_2_id = {c: i for i, c in enumerate(self.chars)}
        self.id_2_char = {i: c for i, c in enumerate(self.chars)}
        self.char_max_len = max(chars_size)
