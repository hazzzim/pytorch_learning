import torch
import torch.nn as nn
from torch.nn import functional as F

# ------------ #
# hyperparameters
batch_size = 16  # how many independent sequences will we process in parallel?
encoder_block_size = 256  # maximum context length for the input
decoder_block_size = 256  # what is the maximum context length for predictions?
max_iters = 5
eval_interval = 100
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 64
n_head = 4
n_layer = 6
dropout = 0.0
# ------------ #

torch.manual_seed(1337)
# region preprocessing
SOS_token = 0
EOS_token = 1

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('../data/transformer/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]  # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l])  # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))  # first 90% will be trained, rest val
train_data = data[:n]
val_data = data[n:]


# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


# endregion

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


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

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.input_token_embedding_table = nn.Embedding(input_vocab_size, n_embd)
        self.input_position_embedding_table = nn.Embedding(input_block_size, n_embd)

        self.output_token_embedding_table = nn.Embedding(output_vocab_size, n_embd)
        self.output_position_embedding_table = nn.Embedding(output_block_size, n_embd)

        self.encoder_blocks = nn.Sequential(*[EncoderBlock(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.decoder_blocks = nn.Sequential(*[DecoderBlock(n_embd, n_head=n_head) for _ in range(n_layer)])

        self.ln_f = nn.LayerNorm(n_embd)  # final layer norm
        self.lm_head = nn.Linear(n_embd, output_vocab_size)

    def forward(self, idx, targets=None):
        B, T_e = idx.shape

        # idx and targets are both (B,T_e) tensor of integers
        encoder_tok_emb = self.input_token_embedding_table(idx)  # (B,T_e,C)
        encoder_pos_emb = self.input_position_embedding_table(torch.arange(T_e, device=device))  # (T_e,C)
        encoder_input = encoder_tok_emb + encoder_pos_emb  # (B,T_e,C)
        encoder_output = self.encoder_blocks(encoder_input)  # (B,T_e,C)

        if targets is None:
            # TODO add a loop for the SOS as an input until the EOS is reached (break at max length)
            loss = None
        else:
            _, T_d = targets.shape

            decoder_input = targets[:, :-1]  # take all the tokens in the sequence but the last one

            """
            Here we have the input for the decoder consisting of (<SOS>, T_d -1) since the last word in the sequence 
            should be the <EOS> and it should be predicted
            """

            sos_tensor = torch.empty(batch_size, 1, device=device).fill_(SOS_token)
            decoder_input = torch.cat((sos_tensor, decoder_input), dim=1)

            decoder_tok_emb = self.output_token_embedding_table(decoder_input)  # (B,T_d,C)
            decoder_pos_emb = self.input_position_embedding_table(torch.arange(T_d, device=device))  # (T_d,C)
            decoder_input = decoder_tok_emb + decoder_pos_emb  # (B,T_d,C)
            decoder_output = self.decoder_blocks(decoder_input, encoder_output)  # (B,T_d,C)

            x = self.ln_f(decoder_output)  # (B,T_d,C)
            logits = self.lm_head(x)  # (B,T_d, output_vocab_size)

            B, T_d, C = logits.shape
            logits = logits.view(B * T_d, C)
            targets = targets.view(B * T_d)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


model = TransformerTranslation()
m = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters()) / 1e6, 'M parameters')

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=2000)[0].tolist()))
