import glob
import os
import unicodedata
import string
import torch


# region data pre-processing
def find_files(data_path: str) -> list:
    """
    A function to check the available training data files, for each available language.

    Args:
        data_path: Path to the directory containing the training data files

    Returns:
        available_languages: the list of available files

    """
    language_list = []

    for data_file in glob.glob(data_path + "/*.txt"):
        language_list.append(os.path.basename(data_file))

    return language_list


train_data_path = "../data/data_nlp_classification/names"

print("Available training data for the languages:", find_files(train_data_path))

"""
since we have a case of the latin letters we can use the ASCII standard for the rest of the tutorial
this will include all the english letters lower case and upper case, and we will include also the punctuations we need
resulting in 57 characters abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ .,;'
"""
all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)

"""
since we have european names which contain nonstandard format such as ü ö or the name Ślusàrski and these names can 
only be represented with Unicode, we need a function which will turn any unicode input into a plane ASCII.
"""


def unicodeToAscii(s: str) -> str:
    """
    A function to turn the letters from unicode to ascii

    Args:
        s: input string

    Returns:
        ascii_string: the transformed string

    """

    # By using NFD normalization, you can handle and process text more consistently across different systems
    # and applications
    ascii_string = ''.join(c for c in unicodedata.normalize('NFD', s)
                           if unicodedata.category(c) != "Mn"
                           and c in all_letters
                           )
    return ascii_string


print(unicodeToAscii('Ślusàrski'))

# Build the category_lines dictionary, a list of names per language
category_lines = {}
all_categories = []


def findFiles(path): return glob.glob(path)


# Read a file and split into lines
def readLines(filename: str) -> list:
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    ascii_lines_list = [unicodeToAscii(line) for line in lines]
    return ascii_lines_list


for filename in findFiles(train_data_path + "/*.txt"):
    category = os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(category)
    lines = readLines(filename)
    category_lines[category] = lines

print(category_lines['Italian'][:5])
n_categories = len(all_categories)
"""
Now that we implemented loading and cleaning the names, next we want to store each word into a tensor 
in this case we will use a one hot vector to encode each character and the combination of these characters will in turn
make up our words stored in a 2D Tensor.
"""


# Find letter index from all_letters, e.g. "a" = 0

def letterToIndex(letter: str) -> int:
    return all_letters.find(letter)


# To demonstrate, turn a letter into a <1 x n_letters> Tensor
def letterToTensor(letter):
    one_hot_tensor = torch.zeros(1, n_letters)
    one_hot_tensor[0][letterToIndex(letter)] = 1
    return one_hot_tensor


# Turn a line into a <line_length x 1 n_letters>,
# or an array of one-hot letter vectors
def lineToTensor(line: str) -> torch.tensor:
    """
    A function which takes a string and transforms it into a 3D tensor
    [number of chars in the line][1 (batch size)][number of letters]
    Args:
        line: input line

    Returns:
        tensor: the tensor encoding the input line

    """

    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1

    return tensor


print(letterToTensor('A'))
# endregion

# region Creating the Network
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import time
import math


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        hidden = F.tanh(self.i2h(input) + self.h2h(hidden))
        output = self.h2o(hidden)
        output = self.softmax(output)

        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)


n_hidden = 128
rnn = RNN(n_letters, n_hidden, n_categories)

# trial input for the network

input = letterToTensor('A')
hidden = rnn.initHidden()
output, next_hidden = rnn(input, hidden)

print(output)

# endregion

# region training the network
import random


# create a function to retrieve the highest score of the output (make use of Tensor.topk)

def categoryFromOutput(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()

    return all_categories[category_i], category_i


print(categoryFromOutput(output))


# create functions to provide random training examples

def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]


def randomTrainingExample():
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = lineToTensor(line)

    return category, line, category_tensor, line_tensor


for i in range(10):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    print('category =', category, '/ line =', line)

# the loss criterion should be Negative log loss NLLLoss since we have a LogSoftmax function as the output layer
criterion = nn.NLLLoss()


# creating the training loop


def train(category_tensor: Tensor, line_tensor: Tensor, learning_rate: float):
    hidden = rnn.initHidden()

    # important step before each training run we must zero out the gradients otherwise they will accumulate
    rnn.zero_grad()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    loss = criterion(output, category_tensor)

    # perform the back propagation
    loss.backward()

    for p in rnn.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)

    return output, loss.item()


# running the training loop and implementing a monitoring output

# modifiable parameters
learning_rate = 0.005
n_iters = 100000
print_every = 5000
plot_every = 1000

# tracking the loss for plotting
current_loss = 0
all_losses = []


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60

    return '%dm %ds' % (m, s)


start = time.time()

for itr in range(1, n_iters + 1):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    output, loss = train(category_tensor, line_tensor, learning_rate)
    current_loss += loss

    # Print 'itr' number, loss, name and guess
    if itr % print_every == 0:
        guess, guess_i = categoryFromOutput(output)
        correct = '✓' if guess == category else '✗ (%s)' % category
        print('%d %d%% (%s) %.4f %s / %s %s' % (itr, itr / n_iters * 100, timeSince(start), loss, line, guess, correct))

        # add current loss avg to list of losses
        if itr % plot_every == 0:
            all_losses.append(current_loss / plot_every)
            current_loss = 0

# endregion

# region post-processing and plotting results
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.figure()
plt.plot(all_losses)


# implementing a confusion matrix to visualize how the network performs on each language vs another
confusion = torch.zeros(n_categories, n_categories)
n_confusion = 10000

# implement an evaluation function
def evaluate(line_tensor: Tensor):
    hidden = rnn.initHidden()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    return output

# go through a sample of examples

for i in range(n_confusion):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    output = evaluate(line_tensor)
    guess, guess_i = categoryFromOutput(output)
    category_i = all_categories.index(category)
    confusion[category_i][guess_i] +=1

    # normalize by dividing each line by its sum
for i in range(n_categories):
    confusion[i] = confusion[i] / confusion[i].sum()

fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(confusion.numpy())
fig.colorbar(cax)

ax.set_xticklabels([''] + all_categories, rotation=90)
ax.set_yticklabels([''] + all_categories)

# Force label at every tick
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

plt.show()

# endregion

# region user input

def predict(input_line, n_predictions = 3):

    print('\n> %s' % input_line)

    with torch.no_grad():
        output = evaluate(lineToTensor(input_line))

        # Get top N categories
        topv, topi = output.topk(n_predictions, 1, True)
        predictions = []

        for i in range(n_predictions):
            value = topv[0][i].item()
            category_index = topi[0][i].item()
            print('(%.2f) %s' % (value, all_categories[category_index]))
            predictions.append([value, all_categories[category_index]])


predict('Hazim ')
predict('Hazim ')
predict('Satoshi')