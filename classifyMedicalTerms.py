import torch.nn as nn
from torch.autograd import Variable
import torch
import string

all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)

#Find letter index from all_letters, e.g. "a" = 0
def letterToIndex(letter):
    return all_letters.find(letter)

#Only for demonstation
def letterToTensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letterToIndex(letter)] = 1
    return tensor

#Turn a line into a tensor
#or an array of one-hot letter vectors

def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax()

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return Variable(torch.zeros(1, self.hidden_size))

all_categories = ['Medical Term', 'Common English Term']

n_hidden = 128
n_categories = 2

rnn = RNN(n_letters, n_hidden, n_categories)
rnn.load_state_dict(torch.load('medicalTermsModel'))

# Just return an output given a line
def evaluate(line_tensor):
    hidden = rnn.initHidden()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    return output

def predict(input_line, n_predictions=1):

    output = evaluate(Variable(lineToTensor(input_line)))

    # Get top N categories
    topv, topi = output.data.topk(n_predictions, 1, True)
    predictions = []

    for i in range(n_predictions):
        value = topv[0][i]
        category_index = topi[0][i]
        predictions.append([value, all_categories[category_index]])
        if category_index == 0:
            # print('\n> %s' % input_line)
            predictions = (str(input_line), str(all_categories[category_index]))
        else:
            predictions = (str(input_line), str(all_categories[category_index]))
    return predictions
