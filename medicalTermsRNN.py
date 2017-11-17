medicalTermsFile = open('Medical_Terms.txt')
commonTermsFile = open('Common_English_Words.txt')

medicalTermsLines = medicalTermsFile.read().splitlines()
commonTermsLines = commonTermsFile.read().splitlines()

all_terms = []
medical_list = []
common_list = []

all_categories = ['Medical Term', 'Common English Term']
import string

all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)

import random
for i in range(25000):
    pickMedical_Common = random.randint(0, 20999)
    # print(pickMedical_Common)
    # 4 = pick medical term
    # 0, 1, 2, 3 = common english term

    if(pickMedical_Common > 20000):
        medicalLineNumber = random.randint(0, 98118)
        #print("medicalLineNumber:", medicalLineNumber)
        medical_list.append(medicalTermsLines[medicalLineNumber])
    else:
        commonLineNumber = random.randint(0, 19999)
        #print("commonLineNumber:", commonLineNumber)
        common_list.append(commonTermsLines[commonLineNumber])

all_terms.append(medical_list)
all_terms.append(common_list)

n_categories = 2

# print(all_terms)
#--------------------------- Data Creation done --------------------------------

import torch

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

print (letterToTensor('J'))

print(lineToTensor('Jones').size())

#-------------------------- Tensor Creation done -------------------------------

import torch.nn as nn
from torch.autograd import Variable

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

n_hidden = 128
rnn = RNN(n_letters, n_hidden, n_categories)

input = Variable(letterToTensor('A'))
hidden = Variable(torch.zeros(1, n_hidden))

output, next_hidden = rnn(input, hidden)

input = Variable(lineToTensor('Albert'))
hidden = Variable(torch.zeros(1, n_hidden))

output, next_hidden = rnn(input[0], hidden)
print(output)

#-------------------------- Network Creation done ------------------------------

def categoryFromOutput(output):
    top_n, top_i = output.data.topk(1) #Tensor out of Variable with .data
    category_i = top_i[0][0]
    return all_categories[category_i], category_i

print(categoryFromOutput(output))

def randomTrainingExample():
    category = random.randint(0, 1)
    if category == 0:
        category = 'Medical Term'
        rand = random.randint(0, len(all_terms[0]) - 1)
        line = all_terms[0][rand]
    else:
        category = 'Common English Term'
        rand = random.randint(0, len(all_terms[1]) - 1)
        line = all_terms[1][rand]
    category_tensor = Variable(torch.LongTensor([all_categories.index(category)]))
    line_tensor = Variable(lineToTensor(line))
    return category, line, category_tensor, line_tensor

for i in range(10):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    print('category =', category, '/ line =', line)

#------------------------- helper functions written ----------------------------

criterion = nn.NLLLoss()

learning_rate = 0.005

def train(category_tensor, line_tensor):
    hidden = rnn.initHidden()

    rnn.zero_grad()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    loss = criterion(output, category_tensor)
    loss.backward()

    for p in rnn.parameters():
        p.data.add_(-learning_rate, p.grad.data)

    return output, loss.data[0]

import time
import math

n_iters = 100000
print_every = 5000
plot_every = 5000

current_loss = 0
all_losses = []

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

start = time.time()

for iter in range(1, n_iters + 1):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    output, loss = train(category_tensor, line_tensor)
    current_loss += loss

    # Print iter number, loss, name and guess
    if iter % print_every == 0:
        guess, guess_i = categoryFromOutput(output)
        correct = 'correct' if guess == category else 'wrong (%s)' % category
        print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters * 100, timeSince(start), loss, line, guess, correct))

    # Add current loss avg to list of losses
    if iter % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.figure()
plt.plot(all_losses)

# Keep track of correct guesses in a confusion matrix
confusion = torch.zeros(n_categories, n_categories)
n_confusion = 10000

# Just return an output given a line
def evaluate(line_tensor):
    hidden = rnn.initHidden()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    return output

# Go through a bunch of examples and record which are correctly guessed
for i in range(n_confusion):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    output = evaluate(line_tensor)
    guess, guess_i = categoryFromOutput(output)
    category_i = all_categories.index(category)
    confusion[category_i][guess_i] += 1

# Normalize by dividing every row by its sum
for i in range(n_categories):
    confusion[i] = confusion[i] / confusion[i].sum()

# Set up plot
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(confusion.numpy())
fig.colorbar(cax)

# Set up axes
ax.set_xticklabels([''] + all_categories, rotation=90)
ax.set_yticklabels([''] + all_categories)

# Force label at every tick
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

# sphinx_gallery_thumbnail_number = 2
plt.show()

#------------------------------- training done ---------------------------------

torch.save(rnn.state_dict(), 'medicalTermsModel')

def predict(input_line, n_predictions=2):
    print('\n> %s' % input_line)
    output = evaluate(Variable(lineToTensor(input_line)))

    # Get top N categories
    topv, topi = output.data.topk(n_predictions, 1, True)
    predictions = []

    for i in range(n_predictions):
        value = topv[0][i]
        category_index = topi[0][i]
        print('(%.2f) %s' % (value, all_categories[category_index]))
        predictions.append([value, all_categories[category_index]])

predict('hypertension')
predict('tumor')
predict('diarrhea')
predict('mumps')
predict('mother')
predict('father')
predict('wimpy')
predict('kid')
predict('detection')
