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
    """A class to handle vocabulary and tokenization for a language."""

    def __init__(self, name):
        """
        Initialize the language object.

        Args:
            name (str): The name of the language.
        """
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # start with EOS and SOS

    def addSentence(self, sentence):
        """
        Tokenize a sentence and add each word to the vocabulary.

        Args:
            sentence (str): The sentence to tokenize and add.
        """
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        """
        Add a single word to the vocabulary.

        Args:
            word (str): The word to add.
        """
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


class LanguageDecoder:
    """
    A class to decode tensors back into human-readable sentences."""

    def __init__(self, input_lang, output_lang):
        """Initialize the decoder.

        Args:
            input_lang (Lang): The input language object.
            output_lang (Lang): The output language object.
        """
        self.input_lang = input_lang
        self.output_lang = output_lang

    def decode_input_tensor(self, tensor):
        """
        Decode an input tensor into a sentence.

        Args:
            tensor (torch.Tensor): The input tensor to decode.

        Returns:
            str: The decoded sentence.
        """
        decoded_input = self.decode_tokens(self.input_lang, tensor)
        return decoded_input

    def decode_output_tensor(self, tensor):
        """
        Decode an output tensor into a sentence.

        Args:
            tensor (torch.Tensor): The output tensor to decode.

        Returns:
            str: The decoded sentence.
        """
        decoded_output = self.decode_tokens(self.output_lang, tensor)
        return decoded_output

    def decode_tokens(self, input_lang, input_example):
        """
        Decode a tensor into a sentence using the given language.

        Args:
            input_lang (Lang): The language object.
            input_example (torch.Tensor): The tensor to decode.

        Returns:
            str: The decoded sentence.
        """
        decoded_input = []
        for idx in input_example.squeeze():
            decoded_input.append(input_lang.index2word[idx.item()])
        decoded_input = " ".join(decoded_input)
        return decoded_input


def unicodeToAscii(s):
    """
    Convert a Unicode string to plain ASCII.

    Args:
        s (str): The Unicode string to convert.

    Returns:
        str: The ASCII string.
    """
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalizeString(s):
    """
    Normalize a string by lowercasing, trimming, and removing non-letter characters.

    Args:
        s (str): The string to normalize.

    Returns:
        str: The normalized string.
    """
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z!?]+", r" ", s)
    return s.strip()


def readLangs(lang1, lang2, reverse=False):
    """
    Read a dataset of sentence pairs and create Lang instances.

    Args:
        lang1 (str): The first language.
        lang2 (str): The second language.
        reverse (bool): Whether to reverse the language pairs.

    Returns:
        tuple: A tuple containing the input language, output language, and pairs.
    """
    print("Reading lines....")
    lines = open('../data/data_nlp_classification/%s-%s.txt' % (lang1, lang2), encoding='utf-8'). \
        read().strip().split('\n')
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
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
    """
    Filter sentence pairs based on length and prefix criteria.

    Args:
        p (list): A pair of sentences.

    Returns:
        bool: Whether the pair passes the filter.
    """
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH and \
        p[1].startswith(eng_prefixes)


def filterPairs(pairs):
    """
    Filter a list of sentence pairs.

    Args:
        pairs (list): A list of sentence pairs.

    Returns:
        list: The filtered list of sentence pairs.
    """
    return [pair for pair in pairs if filterPair(pair)]


def prepareData(lang1, lang2, reverse=False):
    """
    Prepare the dataset by reading, filtering, and tokenizing sentence pairs.

    Args:
        lang1 (str): The first language.
        lang2 (str): The second language.
        reverse (bool): Whether to reverse the language pairs.

    Returns:
        tuple: A tuple containing the input language, output language, and pairs.
    """
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
    """
    Estimate the model's loss on the training and validation datasets.

    Args:
        train_dataloader (DataLoader): The training data loader.
        val_dataloader (DataLoader): The validation data loader.
        model (nn.Module): The model to evaluate.

    Returns:
        dict: A dictionary containing the average loss for training and validation.
    """
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


# region building the Model
class EncodeHead(nn.Module):
    """One head of encoder self-attention."""

    def __init__(self, head_size):
        """Initialize the encoder head.

        Args:
            head_size (int): The size of the attention head.
        """
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """Forward pass for the encoder head.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        B, T, C = x.shape
        k = self.key(x)  # (B,T_e,C)
        q = self.query(x)  # (B,T_e,C)
        wei = q @ k.transpose(-2, -1) * C ** -0.5  # (B, T_e, C) @ (B, C, T_e) -> (B, T_e, T_e)
        wei = F.softmax(wei, dim=-1)  # (B, T_e, T_e)
        wei = self.dropout(wei)
        v = self.value(x)  # (B,T_e,C)
        out = wei @ v  # (B, T_e, T_e) @ (B, T_e, C) -> (B, T_e, C)
        return out


class DecodeHead(nn.Module):
    """One head of decoder self-attention."""

    def __init__(self, head_size):
        """
        Initialize the decoder head.

        Args:
            head_size (int): The size of the attention head.
        """
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(decoder_block_size, decoder_block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass for the decoder head.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        B, T, C = x.shape
        k = self.key(x)  # (B,T_d,C)
        q = self.query(x)  # (B,T_d,C)
        wei = q @ k.transpose(-2, -1) * C ** -0.5  # (B, T_d, C) @ (B, C, T_d) -> (B, T_d, T_d)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # (B, T_d, T_d)
        wei = F.softmax(wei, dim=-1)  # (B, T_d, T_d)
        wei = self.dropout(wei)
        v = self.value(x)  # (B,T_d,C)
        out = wei @ v  # (B, T_d, T_d) @ (B, T_d, C) -> (B, T_d, C)
        return out


class CrossAttentionHead(nn.Module):
    """
    One head of cross-attention."""

    def __init__(self, head_size):
        """
        Initialize the cross-attention head.

        Args:
            head_size (int): The size of the attention head.
        """
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoder_output, decoder_output):
        """
        Forward pass for the cross-attention head.

        Args:
            encoder_output (torch.Tensor): The encoder output tensor.
            decoder_output (torch.Tensor): The decoder output tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        B_d, T_decoder, C_d = decoder_output.shape
        B_e, T_encoder, C_e = encoder_output.shape
        k = self.key(encoder_output)  # (B,T_encoder,C)
        q = self.query(decoder_output)  # (B,T_decoder,C)
        wei = q @ k.transpose(-2,
                              -1) * C_d ** -0.5  # (B, T_decoder, C) @ (B, C, T_encoder) -> (B, T_decoder, T_encoder)
        wei = F.softmax(wei, dim=-1)  # (B, T_decoder, T_encoder)
        wei = self.dropout(wei)
        v = self.value(encoder_output)  # (B,T_encoder,C)
        out = wei @ v  # (B, T_decoder, T_encoder) @ (B, T_encoder, C) -> (B, T_decoder, C)
        return out


class MultiHeadEncoder(nn.Module):
    """Multiple heads of self-attention in parallel."""

    def __init__(self, num_heads, head_size):
        """
        Initialize the multi-head encoder.

        Args:
            num_heads (int): The number of attention heads.
            head_size (int): The size of each attention head.
        """
        super().__init__()
        self.heads = nn.ModuleList([EncodeHead(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass for the multi-head encoder.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class MultiHeadDecoder(nn.Module):
    """Multiple heads of self-attention in parallel."""

    def __init__(self, num_heads, head_size):
        """
        Initialize the multi-head decoder.

        Args:
            num_heads (int): The number of attention heads.
            head_size (int): The size of each attention head.
        """
        super().__init__()
        self.heads = nn.ModuleList([DecodeHead(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass for the multi-head decoder.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class MultiHeadCrossAttention(nn.Module):
    """Multiple heads of cross-attention in parallel."""

    def __init__(self, num_heads, head_size):
        """
        Initialize the multi-head cross-attention.

        Args:
            num_heads (int): The number of attention heads.
            head_size (int): The size of each attention head.
        """
        super().__init__()
        self.heads = nn.ModuleList([CrossAttentionHead(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoder_output, decoder_output):
        """
        Forward pass for the multi-head cross-attention.

        Args:
            encoder_output (torch.Tensor): The encoder output tensor.
            decoder_output (torch.Tensor): The decoder output tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        out = torch.cat([h(encoder_output, decoder_output) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    """A simple linear layer followed by a non-linearity."""

    def __init__(self, n_embd):
        """
        Initialize the feedforward network.

        Args:
            n_embd (int): The embedding dimension.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        """
        Forward pass for the feedforward network.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        return self.net(x)


class EncoderBlock(nn.Module):
    """Transformer block: communication followed by computation."""

    def __init__(self, n_embd, n_head):
        """
        Initialize the encoder block.

        Args:
            n_embd (int): The embedding dimension.
            n_head (int): The number of attention heads.
        """
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadEncoder(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        """
        Forward pass for the encoder block.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class DecoderBlock(nn.Module):
    """Transformer block: communication followed by computation."""

    def __init__(self, n_embd, n_head):
        """
        Initialize the decoder block.

        Args:
            n_embd (int): The embedding dimension.
            n_head (int): The number of attention heads.
        """
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
        """
        Forward pass for the decoder block.

        Args:
            x (torch.Tensor): The input tensor.
            encoder_output (torch.Tensor): The encoder output tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        x = x + self.sa(self.ln1(x))
        x = x + self.ca(self.ln2(encoder_output), self.ln3(x))
        x = x + self.ffwd(self.ln4(x))
        return x


class TransformerTranslation(nn.Module):
    """A transformer-based model for sequence-to-sequence translation."""

    def __init__(self, input_vocab_size, output_vocab_size, input_block_size, output_block_size):
        """
        Initialize the transformer model.

        Args:
            input_vocab_size (int): The size of the input vocabulary.
            output_vocab_size (int): The size of the output vocabulary.
            input_block_size (int): The maximum input sequence length.
            output_block_size (int): The maximum output sequence length.
        """
        super().__init__()
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
        """
        Run the decoder layers.

        Args:
            x (torch.Tensor): The input tensor.
            encoder_output (torch.Tensor): The encoder output tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        for layer in self.decoder_blocks:
            x = layer(x, encoder_output)
        return x

    def run_decoder(self, encoder_output, batch_size, decoder_input=None):
        """
        Run the decoder to generate output sequences.

        Args:
            encoder_output (torch.Tensor): The encoder output tensor.
            batch_size (int): The batch size.
            decoder_input (torch.Tensor, optional): The decoder input tensor.

        Returns:
            torch.Tensor: The output logits.
        """
        sos_tensor = torch.empty(batch_size, 1, device=device, dtype=self.input_dtype).fill_(SOS_token)
        if decoder_input is None:
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
        """
        Convert model logits into predicted token indices.

        Args:
            logits (torch.Tensor): The output logits.

        Returns:
            torch.Tensor: The predicted next token id.
        """
        logits = logits[:, -1, :]  # becomes (B, output_vocab_size)
        probs = F.softmax(logits, dim=-1)  # (B, output_vocab_size)
        _, idx_next = probs.topk(1)
        return idx_next

    def forward(self, idx, targets=None):
        """
        Forward pass for the transformer model. This function supports two modes:
        1. **Training Mode**: When `targets` are provided, the model computes the loss by comparing
           the predicted output with the ground truth.
        2. **Prediction Mode**: When `targets` are not provided, the model generates output sequences
           token by token until the end-of-sequence (EOS) token is predicted or the maximum sequence
           length is reached.

        Args:
            idx (torch.Tensor): The input tensor of shape (B, T_e), where B is the batch size and
                                T_e is the length of the input sequence. Each element is an integer
                                representing a token index from the input vocabulary.
            targets (torch.Tensor, optional): The target tensor of shape (B, T_d), where T_d is the
                                              length of the target sequence. Each element is an integer
                                              representing a token index from the output vocabulary.
                                              If `None`, the model operates in prediction mode.

        Returns:
            - If `targets` is provided (training mode):
                - logits (torch.Tensor): The output logits of shape (B * T_d, output_vocab_size),
                                         where output_vocab_size is the size of the output vocabulary.
                - loss (torch.Tensor): The computed cross-entropy loss between the predicted logits
                                       and the ground truth targets.
            - If `targets` is not provided (prediction mode):
                - predicted_tokens (torch.Tensor): The predicted output sequence of shape (B, T_pred),
                                                   where T_pred is the length of the predicted sequence.
                                                   The sequence is generated token by token until the
                                                   EOS token is predicted or the maximum sequence length
                                                   is reached.

        Workflow:
        ---------
        1. **Input Encoding**:
           - The input tensor `idx` is passed through the input token embedding table to get token
             embeddings of shape (B, T_e, C), where C is the embedding dimension.
           - Positional embeddings are added to the token embeddings to incorporate positional
             information.
           - The combined embeddings are passed through the encoder blocks to produce the encoder
             output of shape (B, T_e, C).

        2. **Training Mode**:
           - If `targets` are provided, the model operates in training mode.
           - The target sequence is trimmed to exclude the last token, as the model predicts the next
             token at each step.
           - The decoder processes the trimmed target sequence along with the encoder output to
             produce logits of shape (B, T_d, output_vocab_size).
           - The logits are reshaped to (B * T_d, output_vocab_size), and the targets are reshaped to
             (B * T_d).
           - The cross-entropy loss is computed between the logits and the targets.

        3. **Prediction Mode**:
           - If `targets` are not provided, the model operates in prediction mode.
           - The decoder generates the output sequence token by token, starting with the start-of-sequence
             (SOS) token.
           - At each step, the model predicts the next token using the previously predicted tokens and
             the encoder output.
           - The process continues until the EOS token is predicted or the maximum sequence length is
             reached.
           - The final predicted sequence is returned as a tensor of shape (B, T_pred).
        """
        B, T_e = idx.shape
        # Input encoding
        encoder_tok_emb = self.input_token_embedding_table(idx)  # (B, T_e, C)
        encoder_pos_emb = self.input_position_embedding_table(torch.arange(T_e, device=device))  # (T_e, C)
        encoder_input = encoder_tok_emb + encoder_pos_emb  # (B, T_e, C)
        encoder_output = self.encoder_blocks(encoder_input)  # (B, T_e, C)

        if targets is None:
            # Prediction mode
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
            # Training mode
            _, T_d = targets.shape
            decoder_input = targets[:, :-1]  # Exclude the last token for teacher forcing
            logits = self.run_decoder(decoder_input, encoder_output, B)
            B, T_d, C = logits.shape
            logits = logits.view(B * T_d, C)
            targets = targets.view(B * T_d)
            loss = F.cross_entropy(logits, targets)
            return logits, loss

# region Training
def indexFromSentance(lang, sentance):
    """
    Convert a sentence into a list of token indices.

    Args:
        lang (Lang): The language object.
        sentance (str): The sentence to convert.

    Returns:
        list: A list of token indices.
    """
    return [lang.word2index[word] for word in sentance.split(' ')]


def tensorFromSentance(lang, sentance):
    """
    Convert a sentence into a tensor of token indices.

    Args:
        lang (Lang): The language object.
        sentance (str): The sentence to convert.

    Returns:
        torch.Tensor: The tensor of token indices.
    """
    indexes = indexFromSentance(lang, sentance)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(1, -1)


def get_dataloader(batch_size, val_split):
    """
    Prepare data loaders for training and validation.

    Args:
        batch_size (int): The batch size.
        val_split (float): The fraction of data to use for validation.

    Returns:
        tuple: A tuple containing the input language, output language, and data loaders.
    """
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
    """
    Train the model.

    Args:
        model (nn.Module): The model to train.
        optimizer (torch.optim.Optimizer): The optimizer.
        dataloader (DataLoader): The training data loader.
        val_dataloader (DataLoader): The validation data loader.
        epochs (int): The number of epochs to train.
    """
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
    """Generate translations for random samples from the dataset.

    Args:
        model (nn.Module): The trained model.
        dataloader (DataLoader): The data loader.
        samples (int): The number of samples to generate.
        decoder (LanguageDecoder): The decoder for converting tensors to sentences.
    """
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
    """Remove trailing zeros from a tensor.

    Args:
        tensor (torch.Tensor): The tensor to trim.

    Returns:
        torch.Tensor: The trimmed tensor.
    """
    flattened = tensor.flatten()
    non_zero_indices = (flattened != 0).nonzero(as_tuple=True)[0]
    if len(non_zero_indices) == 0:
        return torch.tensor([], device=tensor.device)
    last_non_zero_index = non_zero_indices[-1]
    trimmed_tensor = flattened[:last_non_zero_index + 1]
    return trimmed_tensor.view(1, -1)


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
    # torch.save(translation_model.state_dict(), "saved_models/transformer_translator_m
    generate_random(translation_model, train_dataloader, 20, language_decoder)
    print(1)
