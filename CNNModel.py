from math import sqrt

import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import optim, no_grad, tanh, cat
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torch import cuda
import numpy as np

global_loss = []
global_accuracy = []
device = 'cuda' if cuda.is_available() else 'cpu'


class CNNModel(nn.Module):
    def __init__(self, hidden_size=100, number_of_chars=0, char_2_id=None, number_of_labels=0, word_max_len=0,
                 batch_size=32, word_vectors_file=None, pre_trained_vocab_size=0, embedded_char_dimension=0,
                 embedded_pre_trained_dimension=0, window_size=0, window_conv_size=3, filters=0):
        super().__init__()
        self.embedded_char_dimension = embedded_char_dimension
        self.embedded_pre_trained_dimension = embedded_pre_trained_dimension
        self.filters = filters
        self.word_max_len = word_max_len
        self.window_size = window_size
        self.hidden_dim = hidden_size

        self.pre_trained_embedding = nn.Embedding(pre_trained_vocab_size, self.embedded_pre_trained_dimension)
        word_vectors_values = np.loadtxt(word_vectors_file)
        self.pre_trained_embedding.weight.data.copy_(torch.from_numpy(word_vectors_values))

        self.char_embeddings = nn.Embedding(number_of_chars, self.embedded_char_dimension, padding_idx=char_2_id['PAD'])
        self.char_embeddings.weight.data.uniform_(-sqrt(3 / self.embedded_char_dimension),
                                                  sqrt(3 / self.embedded_char_dimension))

        self.conv = nn.Conv1d(in_channels=self.embedded_char_dimension,
                              out_channels=self.filters,
                              kernel_size=window_conv_size,
                              padding=2,
                              stride=1)

        self.linear = nn.Linear((embedded_pre_trained_dimension + self.filters) * self.window_size,
                                self.hidden_dim)

        self.linear2 = nn.Linear(self.hidden_dim, number_of_labels)
        self.batch_size = batch_size
        self.window_conv_size = window_conv_size

    def forward(self, x):
        words_index = x[:, :, -1]
        word_embedding = self.pre_trained_embedding(words_index)

        chars_index = x[:, :, :-1]
        char_representation = self.char_embeddings(chars_index)
        char_representation = char_representation.reshape(char_representation.size(0), self.embedded_char_dimension, self.window_size * self.word_max_len)
        char_representation = self.conv(char_representation)
        char_representation = nn.functional.max_pool1d(char_representation, kernel_size=self.word_max_len, padding=1)
        char_representation = char_representation.transpose(1, 2)

        out = cat((char_representation, word_embedding), 2)

        out = out.view(out.size(0), -1)
        out = self.linear(out)
        out = tanh(out)
        out = self.linear2(out)
        return out

    def training_model(self, model, x, y, loss_list, optimizer):
        # Sets model to TRAIN mode
        model.train()
        # Makes predictions
        y_pred = model(x)
        # Computes loss
        loss = F.cross_entropy(y_pred, y)
        # Computes gradients
        loss.backward()
        # Updates parameters and zeroes gradients
        optimizer.step()
        optimizer.zero_grad()
        loss_list.append(loss.item())

    def train_nn(self, model, train_data, tag_data, batch_size=1024, learning_rate=0.01, validation_data=None,
                 validation_tag=None, is_pos=True, epoch=10, corpus=None, parser=None, tagger_name=None):
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        dataset = TensorDataset(train_data, tag_data)
        data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

        dataset = TensorDataset(validation_data, validation_tag)
        validation_loader = DataLoader(dataset=dataset, batch_size=1024)

        loss_list = []
        for t in range(epoch):
            print(f"EPOCH: {t + 1}")
            for x, y in data_loader:
                data = x.to(device)
                tags = y.to(device)
                self.training_model(model, data, tags, loss_list, optimizer)

            self.evaluate(validation_loader, model, is_pos, parser=parser)

        self.plot(corpus, tagger_name)

        return model

    def plot(self, corpus=None, tagger_name=None):
        self.plot_accuracy(corpus=corpus, tagger_name=tagger_name)
        self.plot_loss(corpus=corpus, tagger_name=tagger_name)

    def plot_accuracy(self, corpus=None, tagger_name=None):
        plt.figure()
        plt.plot(global_accuracy)
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.savefig(tagger_name + '_acc_' + corpus + '.png')

    def plot_loss(self, corpus=None, tagger_name=None):
        plt.figure()
        plt.plot(global_loss)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.savefig(tagger_name + '_loss_' + corpus + '.png')

    def predict(self, model, data, batch_size=None):
        dataset = TensorDataset(data)
        data_loader = DataLoader(dataset=dataset, batch_size=batch_size)
        model.eval()
        preds = []

        with no_grad():
            for x in data_loader:
                x = x[0].to(device)
                out = model(x)
                preds += out.argmax(axis=1).tolist()
        return preds

    def evaluate(self, loader, model, pos=True, parser=None):
        model.eval()
        loss_list = []
        correct = 0
        total = 0

        with no_grad():
            for x, y in loader:
                x = x.to(device)
                y = y.to(device)
                out = model(x)
                loss_list.append(F.cross_entropy(out, y).item())
                if not pos:
                    for y_real, y_pred in zip(y, out.argmax(axis=1)):
                        y_real, y_pred = int(y_real), int(y_pred)

                        if not (parser.id_2_tag[y_real] == 'O' and y_pred == y_real):
                            if y_pred == y_real:
                                correct += 1
                            total += 1

                else:
                    total += len(y)
                    correct += (out.argmax(axis=1) == y).sum().item()
            global_accuracy.append(100 * correct / total)
            global_loss.append(sum(loss_list))

        print(f'Accuracy : {100 * correct / total}% loss:{sum(loss_list)}')

    def predict_test(self, file_name, file_out_name, parser, network, batch_size):
        test, text = parser.load_test_file(file_name)
        results = self.predict(network, test, batch_size=batch_size)

        with open(file_out_name, 'w') as f:
            i = 0
            for line in text:
                res = ""
                for word in line:
                    res += word + '\t' + parser.id_2_tag[results[i]] + '\n'
                    i += 1
                f.write(res + '\n')
