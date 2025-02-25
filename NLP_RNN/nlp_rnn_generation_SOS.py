import glob
import os
import unicodedata
import string

# region preprocessing
all_letters = string.ascii_letters + " .,;'-"

"""
Since the problem we are tying to solve in this tutorial is a generation problem the training data should include an End
Of Sentence marker <EOS> to indicate that the generation is complete and we can stop predicting the next character
"""

n_letters = len(all_letters) + 2  # plus the SOS and EOS Token marker respectively


def findFiles(path): return glob.glob(path)


def unicodeToAscii(s):
    ascii_string = ''.join(c for c in unicodedata.normalize('NFD', s)
                           if unicodedata.category(c) != 'Mn'
                           and c in all_letters)
    return ascii_string


# Read file and split into lines, and return the processed lines of each file
def readLines(filename):
    with open(filename, encoding='utf-8') as input_file:
        line_list = []
        for line in input_file:
            line = unicodeToAscii(line.strip())
            line_list.append(line)

    return line_list


# Build the category_lines dictionary, a list of lines per category
category_lines = {}
all_categories = []

for filename in findFiles('../data/data_nlp_classification/names/*.txt'):
    category = os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(category)
    lines = readLines(filename)
    category_lines[category] = lines

n_categories = len(all_categories)

if n_categories == 0:
    raise RuntimeError('Data not found. Make sure that you downloaded data from'
                       ' https://download.pytorch.org/tutorial/data.zip and extract it to the current directory.')
print('# categories:', n_categories, all_categories)
print(unicodeToAscii("O'Néàl"))
# endregion

# region creating the network
import torch
import torch.nn as nn

"""
in the tutorial (https://pytorch.org/tutorials/intermediate/char_rnn_generation_tutorial.html)
there is a schematic of the network architecture, the network is used in a recurrent mode
but it is not the typical RNN unit since tanh is not applied to its recurrent unit. 
"""


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(n_categories + input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(n_categories + input_size + hidden_size, output_size)
        self.o2o = nn.Linear(hidden_size + output_size, output_size)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, category, input, hidden):
        input_combined = torch.cat((category, input, hidden), 1)
        hidden = self.i2h(input_combined)
        output = self.i2o(input_combined)
        output_combined = torch.cat((hidden, output), 1)
        output = self.o2o(output_combined)
        output = self.dropout(output)
        output = self.softmax(output)

        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)


# endregion

# region training the network
import random


# random item from a list
def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]


# get a random category and random line from that category
def randomTrainingPair():
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])

    return category, line


# One-hot vector for category
def categoryTensor(category):
    li = all_categories.index(category)
    tensor = torch.zeros(1, n_categories)
    tensor[0][li] = 1
    return tensor


# one-hot matrix of first to last letters (not including EOS) for input
def inputTensor(line):
    tensor = torch.zeros(len(line) + 1, 1, n_letters)
    tensor[0][0][n_letters - 2] = 1  # adding SOS Token
    for li in range(len(line)):
        letter = line[li]
        tensor[li + 1][0][all_letters.find(letter)] = 1

    return tensor


def inputTensorPost(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li in range(len(line)):
        letter = line[li]
        tensor[li][0][all_letters.find(letter)] = 1

    return tensor


# ''LongTensor'' of second letter to end (EOS) for target
def targetTensor(line):
    letter_indexes = [all_letters.find(line[li]) for li in range(len(line))]
    letter_indexes.append(n_letters - 1)  # EOS

    return torch.LongTensor(letter_indexes)


# Make category, input and target tensors from random category, line pair
def randomTrainingExample():
    category, line = randomTrainingPair()
    category_tensor = categoryTensor(category)
    input_line_tensor = inputTensor(line)
    target_line_tensor = targetTensor(line)

    return category_tensor, input_line_tensor, target_line_tensor


# endregion


# region training the network
import time
import math

"""
Here since we are making prediction at each time step (as opposed to the classification problem) we need to 
include the loss of each prediction in the loss function (this is called the teacher method)
there are other strategies to train the network of this kind. but for now we will just sum up the losses 
"""
# loss
criterion = nn.NLLLoss()

# parameters
learning_rate = 0.0005
n_iters = 100000
print_every = 1
plot_every = 50
hidden_units = 7745
all_losses = []
total_loss = 0  # Reset every ``plot_every`` ``iters``

# the network
rnn = RNN(n_letters, hidden_units, n_letters)
print("number of model paraeters: ", sum(p.numel() for p in rnn.parameters()))

def train(category_tensor, input_line_tensor, target_line_tensor):
    target_line_tensor.unsqueeze_(-1)
    hidden = rnn.initHidden()

    rnn.zero_grad()

    loss = torch.Tensor([0])  # you can also just simply use ``loss = 0``

    for i in range(input_line_tensor.size(0)):
        output, hidden = rnn(category_tensor, input_line_tensor[i], hidden)
        l = criterion(output, target_line_tensor[i])
        loss += l

    loss.backward()

    for p in rnn.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)

    return output, loss.item() / input_line_tensor.size(0)


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


start = time.time()

for iter in range(1, n_iters + 1):

    output, loss = train(*randomTrainingExample())
    total_loss += loss

    if iter % print_every == 0:
        print('%s (%d %d%%) %.4f' % (timeSince(start), iter, iter / n_iters * 100, loss))

    if iter % plot_every == 0:
        all_losses.append(total_loss / plot_every)
        total_loss = 0

# endregion

# region postprocessing and evaluation
import matplotlib.pyplot as plt

plt.figure()
plt.plot(all_losses)
plt.show()

max_length = 20


# Sample from a category and starting letter
def sample(category):
    with torch.no_grad():  # no need to track history in sampling
        category_tensor = categoryTensor(category)
        input_tensor = torch.zeros(1, 1, n_letters)
        input_tensor[0][0][n_letters - 2] = 1  # adding SOS Token
        input = input_tensor
        hidden = rnn.initHidden()

        output_name = ''

        for i in range(max_length):
            output, hidden = rnn(category_tensor, input[0], hidden)
            topv, topi = output.topk(1)
            topi = topi[0][0]
            if topi == n_letters - 1:
                break
            else:
                letter = all_letters[topi]
                output_name += letter
            input = inputTensorPost(letter)

        return output_name


# Get multiple samples from one category and multiple starting letters
def samples(category, number_of_samples):
    for start_letter in range(number_of_samples):
        print(category, ":", sample(category))


samples('Russian', 3)

samples('German', 3)

samples('Spanish', 3)

samples('Chinese', 3)
# endregion
