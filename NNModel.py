from torch.utils.data import DataLoader, TensorDataset
from torch import optim, FloatTensor, no_grad, tanh
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torch import cuda

EMBEDDING = 50
WINDOW_SIZE = 5
device = 'cuda' if cuda.is_available() else 'cpu'

global_loss = []
global_accuracy = []


class NNModel(nn.Module):
    def __init__(self, output_size, w_embed=None, emb_size=None, hidden_size=100, features=False, is_cnn=False):
        super().__init__()
        if w_embed is None:
            self.embedded = nn.Embedding(emb_size, EMBEDDING).to(device)

        else:
            w_embed = FloatTensor(w_embed)
            self.embedded = nn.Embedding.from_pretrained(w_embed, freeze=False).to(device)

        self.hidden = nn.Linear(EMBEDDING * WINDOW_SIZE, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.features = features
        self.is_cnn = is_cnn

    def forward(self, x):
        emb = EMBEDDING * WINDOW_SIZE

        if not self.features:
            x = self.embedded(x).view(-1, emb)
        else:
            x = self.embedded(x).sum(dim=2).view(-1, emb)

        x = tanh(self.hidden(x))
        x = F.dropout(x)
        x = self.out(x)

        return F.softmax(x, dim=1)

    def training_model(self, model, x, y, loss_list, optimizer):
        model.train()
        y_pred = model(x)
        loss = F.cross_entropy(y_pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss_list.append(loss.item())

    def train_nn(self, model, train_data, tag_data, batch_size=1024, learning_rate=0.01, validation_data=None,
                 validation_tag=None, is_pos=True, epoch=10, corpus=None, parser=None, tagger_name = None):
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        dataset = TensorDataset(train_data, tag_data)
        data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

        dataset = TensorDataset(validation_data, validation_tag)
        validation_loader = DataLoader(dataset=dataset, batch_size=1024)

        loss_list = []
        for t in range(epoch):
            print(f"EPOCH: {t + 1}")
            for x, y in data_loader:
                x = x.to(device)
                y = y.to(device)
                self.training_model(model, x, y, loss_list, optimizer)

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
