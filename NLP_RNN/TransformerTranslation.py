import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
import unicodedata
import re
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# ------------ #
# hyperparameters
batch_size = 16  # how many independent sequences will we process in parallel?
decoder_block_size = 10  # what is the maximum context length for predictions?
max_iters = 5
eval_interval = 1
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 64
n_head = 4
n_layer = 2
dropout = 0.4
# ------------ #

# region preprocessing
SOS_token = 0
EOS_token = 1


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # start with EOS and SOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


class LanguageDecoder:
    def __init__(self, input_lang, output_lang):
        self.input_lang = input_lang
        self.output_lang = output_lang

    def decode_input_tensor(self, tensor):
        decoded_input = self.decode_tokens(self.input_lang, tensor)

        return decoded_input

    def decode_output_tensor(self, tensor):
        decoded_output = self.decode_tokens(self.output_lang, tensor)

        return decoded_output

    def decode_tokens(self, input_lang, input_example):
        decoded_input = []
        for idx in input_example.squeeze():
            decoded_input.append(input_lang.index2word[idx.item()])

        decoded_input = " ".join(decoded_input)
        return decoded_input


# Turn a Unicode string to plain ASCII
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z!?]+", r" ", s)
    return s.strip()


def readLangs(lang1, lang2, reverse=False):
    print("Reading lines....")

    # Read the file and split into 2 lines
    lines = open('../data/data_nlp_classification/%s-%s.txt' % (lang1, lang2), encoding='utf-8'). \
        read().strip().split('\n')

    # Split every line into pairs
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs


MAX_LENGTH = 10
MAX_PREDICT_LENGTH = MAX_LENGTH - 1

eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)


def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH and \
        p[1].startswith(eng_prefixes)


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]


def prepareData(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")

    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs


@torch.no_grad()
def estimate_loss(train_dataloader, val_dataloader, model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            if split == "train":
                X, Y = next(iter(train_dataloader))
            else:
                X, Y = next(iter(val_dataloader))
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# region building the  Model
# endregion
class EncodeHead(nn.Module):
    """ one head of encoder self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # (B,T_e,C)
        q = self.query(x)  # (B,T_e,C)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * C ** -0.5  # (B, T_e, C) @ (B, C, T_e) -> (B, T_e, T_e)
        wei = F.softmax(wei, dim=-1)  # (B, T_e, T_e)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x)  # (B,T_e,C)
        out = wei @ v  # (B, T_e, T_e) @ (B, T_e, C) -> (B, T_e, C)
        return out


class DecodeHead(nn.Module):
    """ one head of decoder self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(decoder_block_size, decoder_block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # (B,T_d,C)
        q = self.query(x)  # (B,T_d,C)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * C ** -0.5  # (B, T_d, C) @ (B, C, T_d) -> (B, T_d, T_d)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # (B, T_d, T_d)
        wei = F.softmax(wei, dim=-1)  # (B, T_d, T_d)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x)  # (B,T_d,C)
        out = wei @ v  # (B, T_d, T_d) @ (B, T_d, C) -> (B, T_d, C)
        return out


class CrossAttentionHead(nn.Module):
    """ one head of cross-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoder_output, decoder_output):
        B_d, T_decoder, C_d = decoder_output.shape
        B_e, T_encoder, C_e = encoder_output.shape
        k = self.key(encoder_output)  # (B,T_encoder,C)
        q = self.query(decoder_output)  # (B,T_decoder,C)
        # compute cross attention scores ("affinities"), here we can use C_e or C_d since they must be the same
        wei = q @ k.transpose(-2,
                              -1) * C_d ** -0.5  # (B, T_decoder, C) @ (B, C, T_encoder) -> (B, T_decoder, T_encoder)
        wei = F.softmax(wei, dim=-1)  # (B, T_decoder, T_encoder)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(encoder_output)  # (B,T_encoder,C)
        out = wei @ v  # (B, T_decoder, T_encoder) @ (B, T_encoder, C) -> (B, T_decoder, C)
        return out


class MultiHeadEncoder(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([EncodeHead(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class MultiHeadDecoder(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([DecodeHead(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class MultiHeadCrossAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([CrossAttentionHead(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoder_output, decoder_output):
        out = torch.cat([h(encoder_output, decoder_output) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class EncoderBlock(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadEncoder(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class DecoderBlock(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadDecoder(n_head, head_size)
        self.ca = MultiHeadCrossAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ln3 = nn.LayerNorm(n_embd)
        self.ln4 = nn.LayerNorm(n_embd)

    def forward(self, x, encoder_output):
        x = x + self.sa(self.ln1(x))
        # TODO here try out a single layer norm for both decoder output and encoder output
        x = x + self.ca(self.ln2(encoder_output), self.ln3(x))
        x = x + self.ffwd(self.ln4(x))
        return x


class TransformerTranslation(nn.Module):

    def __init__(self, input_vocab_size, output_vocab_size, input_block_size, output_block_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.input_token_embedding_table = nn.Embedding(input_vocab_size, n_embd)
        self.input_position_embedding_table = nn.Embedding(input_block_size, n_embd)

        self.output_token_embedding_table = nn.Embedding(output_vocab_size, n_embd)
        self.output_position_embedding_table = nn.Embedding(output_block_size, n_embd)

        self.encoder_blocks = nn.Sequential(*[EncoderBlock(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.decoder_blocks = nn.ModuleList([DecoderBlock(n_embd, n_head=n_head) for _ in range(n_layer)])

        self.ln_f = nn.LayerNorm(n_embd)  # final layer norm
        self.lm_head = nn.Linear(n_embd, output_vocab_size)
        self.input_dtype = torch.int64

    def run_decoder_layers(self, x, encoder_output):
        for layer in self.decoder_blocks:
            x = layer(x, encoder_output)
        return x

    def run_decoder(self, encoder_output, batch_size, decoder_input=None):

        sos_tensor = torch.empty(batch_size, 1, device=device, dtype=self.input_dtype).fill_(SOS_token)
        if decoder_input == None:
            decoder_input = sos_tensor
        else:
            decoder_input = torch.cat((sos_tensor, decoder_input), dim=1)
        _, T_d = decoder_input.shape
        decoder_tok_emb = self.output_token_embedding_table(decoder_input)  # (B,T_d,C)
        decoder_pos_emb = self.input_position_embedding_table(torch.arange(T_d, device=device))  # (T_d,C)
        decoder_input = decoder_tok_emb + decoder_pos_emb  # (B,T_d,C)
        decoder_output = self.run_decoder_layers(decoder_input, encoder_output)  # (B,T_d,C)

        x = self.ln_f(decoder_output)  # (B,T_d,C)
        logits = self.lm_head(x)  # (B,T_d, output_vocab_size)

        return logits

    def postprocess_prediction(self, logits):
        # apply softmax to get probabilities
        logits = logits[:, -1, :]  # becomes (B, output_vocab_size)
        # apply softmax to get probabilities
        probs = F.softmax(logits, dim=-1)  # (B, output_vocab_size)
        _, idx_next = probs.topk(1)
        return idx_next

    def forward(self, idx, targets=None):
        B, T_e = idx.shape

        # idx and targets are both (B,T_e) tensor of integers
        encoder_tok_emb = self.input_token_embedding_table(idx)  # (B,T_e,C)
        encoder_pos_emb = self.input_position_embedding_table(torch.arange(T_e, device=device))  # (T_e,C)
        encoder_input = encoder_tok_emb + encoder_pos_emb  # (B,T_e,C)
        encoder_output = self.encoder_blocks(encoder_input)  # (B,T_e,C)

        if targets is None:
            logits = self.run_decoder(encoder_output, B)
            targets = self.postprocess_prediction(logits)

            for index in range(MAX_PREDICT_LENGTH):
                logits = self.run_decoder(encoder_output, B, targets)
                idx_next = self.postprocess_prediction(logits)
                targets = torch.cat((targets, idx_next), dim=1)
                if idx_next.item() == EOS_token:
                    return targets
            return targets


        else:
            _, T_d = targets.shape

            decoder_input = targets[:, :-1]  # take all the tokens in the sequence but the last one

            """
            Here we have the input for the decoder consisting of (<SOS>, T_d -1) since the last word in the sequence 
            should usually be the <EOS> and it should be predicted (in this case we have not trimmed the trailing tokens
            after the <EOS>)
            """

            logits = self.run_decoder(decoder_input, encoder_output, B)

            B, T_d, C = logits.shape
            logits = logits.view(B * T_d, C)
            targets = targets.view(B * T_d)
            loss = F.cross_entropy(logits, targets)

        return logits, loss


# region Training

def indexFromSentance(lang, sentance):
    return [lang.word2index[word] for word in sentance.split(' ')]


def tensorFromSentance(lang, sentance):
    indexes = indexFromSentance(lang, sentance)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(1, -1)


def get_dataloader(batch_size, val_split):
    input_lang, output_lang, pairs = prepareData('eng', 'fra', True)

    n = len(pairs)
    input_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)
    target_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)

    for idx, (inp, tgt) in enumerate(pairs):
        inp_ids = indexFromSentance(input_lang, inp)
        tgt_ids = indexFromSentance(output_lang, tgt)
        inp_ids.append(EOS_token)
        tgt_ids.append(EOS_token)

        input_ids[idx, :len(inp_ids)] = inp_ids
        target_ids[idx, :len(tgt_ids)] = tgt_ids

    data_train, data_val, labels_train, labels_val = train_test_split(input_ids, target_ids, test_size=val_split,
                                                                      random_state=42)

    train_data = TensorDataset(torch.LongTensor(data_train).to(device),
                               torch.LongTensor(labels_train).to(device))

    val_data = TensorDataset(torch.LongTensor(data_val).to(device),
                             torch.LongTensor(labels_val).to(device))

    train_sampler = RandomSampler(data_val)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    val_dataloader = DataLoader(val_data, sampler=train_sampler, batch_size=batch_size)

    return input_lang, output_lang, train_dataloader, val_dataloader


def train(model, optimizer, dataloader, val_dataloader, epochs: int):
    val_losses = []
    train_losses = []
    for epoch in range(1, epochs + 1):
        if epoch % eval_interval == 0 or epoch == max_iters - 1:
            losses = estimate_loss(dataloader, val_dataloader, model)
            val_losses.append((losses['val']))
            train_losses.append((losses['train']))
            print(f"epoch {epoch}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        for data in dataloader:
            input_tensor, target_tensor = data
            logits, loss = model(input_tensor, target_tensor)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    plt.plot(val_losses, 'g', train_losses, 'r')
    plt.title("epoch VS loss")
    plt.show()


def generate_random(model, dataloader, samples, decoder):
    with torch.no_grad():
        model.eval()
        for sample in range(samples):
            input_tensor, output_tensor = random.choice(dataloader.dataset)
            input_tensor = trim_trailing_zeros(input_tensor)
            predicted_tensor = model(input_tensor)

            input_sentence = decoder.decode_input_tensor(input_tensor)
            output_sentence = decoder.decode_output_tensor(output_tensor)
            predicted_sentence = decoder.decode_output_tensor(predicted_tensor)

            print(f"<-----------------------{sample}------------------------>")
            print("input sentence: ", input_sentence)
            print("output sentence:", output_sentence)
            print("predicted sentence:", predicted_sentence)
            print("<------------------------------------------------>")
        model.train()


def trim_trailing_zeros(tensor):
    # Flatten the tensor to handle it as a 1D array
    flattened = tensor.flatten()

    # Find the index of the last non-zero element
    non_zero_indices = (flattened != 0).nonzero(as_tuple=True)[0]

    if len(non_zero_indices) == 0:
        # If all elements are zero, return an empty tensor
        return torch.tensor([], device=tensor.device)

    last_non_zero_index = non_zero_indices[-1]

    # Slice the tensor up to the last non-zero element
    trimmed_tensor = flattened[:last_non_zero_index + 1]

    # Reshape the tensor back to its original shape (excluding trailing zeros)
    return trimmed_tensor.view(1, -1)  # the original tensor is 2D


if __name__ == "__main__":
    input_lang, output_lang, train_dataloader, val_dataloader = get_dataloader(batch_size, 0.2)
    translation_model = TransformerTranslation(input_lang.n_words, output_lang.n_words, MAX_LENGTH, MAX_LENGTH)
    language_decoder = LanguageDecoder(input_lang, output_lang)
    translation_model.to(device)

    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(translation_model.parameters(), lr=learning_rate)
    # train the model or load existing
    translation_model.load_state_dict(torch.load('saved_models/transformer_translator_model_v1.pth'))
    # train(translation_model, optimizer, train_dataloader, val_dataloader, 15)
    # torch.save(translation_model.state_dict(), "saved_models/transformer_translator_model_v1.pth")

    generate_random(translation_model, train_dataloader, 20, language_decoder)
    print(1)
