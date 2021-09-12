import torch
import torch.nn as nn

from torchtext.data import Field, BucketIterator
from torchtext import data

import numpy as np

import random
import math
import time

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Sorting key for examples
def example_key(e):
    return len(e.src)

# Create a torchtext dataset from two source arrays
def create_dataset(src, trg, f=(data.Field(), data.Field())):
    fields = [('src', f[0]), ('trg', f[1])]
    examples = []
    for s, t in zip(src, trg):
        s = s.replace("\n", "").split(" ")
        t = t.replace("\n", "").split(" ")
        examples.append(data.Example.fromlist([s, t], fields))
    dataset = data.Dataset(examples, fields)
    dataset.sort_key = example_key
    return dataset

# Tokenizer
def tokenize(text):
    return text.split(" ")

class Encoder(nn.Module):
    def __init__(self,
                 input_dim,
                 hid_dim,
                 n_layers,
                 n_heads,
                 pf_dim,
                 dropout,
                 device,
                 max_length=100):
        super().__init__()

        self.device = device

        self.tok_embedding = nn.Embedding(input_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)

        self.layers = nn.ModuleList([EncoderLayer(hid_dim,
                                                  n_heads,
                                                  pf_dim,
                                                  dropout,
                                                  device)
                                     for _ in range(n_layers)])

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

    def forward(self, src, src_mask):
        batch_size = src.shape[0]
        src_len = src.shape[1]

        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)

        src = self.dropout((self.tok_embedding(src) * self.scale) + self.pos_embedding(pos))

        for layer in self.layers:
            src = layer(src, src_mask)

        return src


class EncoderLayer(nn.Module):
    def __init__(self,
                 hid_dim,
                 n_heads,
                 pf_dim,
                 dropout,
                 device):
        super().__init__()

        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, pf_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask):
        # self attention
        fsrc, _ = self.self_attention(src, src, src, src_mask)

        # dropout, residual connection and layer norm
        src = self.self_attn_layer_norm(src + self.dropout(fsrc))

        # positionwise feedforward
        fsrc = self.positionwise_feedforward(src)

        # dropout, residual and layer norm
        return self.ff_layer_norm(src + self.dropout(fsrc))


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()

        assert hid_dim % n_heads == 0

        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads

        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)

        self.fc_o = nn.Linear(hid_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

    def forward(self, query, key, value, mask=None):

        batch_size = query.shape[0]

        fc_query = self.fc_q(query)
        fc_key = self.fc_k(key)
        fc_value = self.fc_v(value)

        fc_query = fc_query.view(batch_size, -1, self.n_heads,
                   self.head_dim).permute(0, 2, 1, 3)
        fc_key = fc_key.view(batch_size, -1, self.n_heads,
                   self.head_dim).permute(0, 2, 1, 3)
        fc_value = fc_value.view(batch_size, -1, self.n_heads,
                   self.head_dim).permute(0, 2, 1, 3)

        energy = torch.matmul(fc_query, fc_key.permute(0, 1, 3, 2)) / self.scale

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        attention = torch.softmax(energy, dim=-1)

        x = torch.matmul(self.dropout(attention), fc_value)

        x = x.permute(0, 2, 1, 3).contiguous()

        x = x.view(batch_size, -1, self.hid_dim)

        x = self.fc_o(x)

        return x, attention


class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()

        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(torch.relu(self.fc_1(x)))
        return self.fc_2(x)


class Decoder(nn.Module):
    def __init__(self,
                 output_dim,
                 hid_dim,
                 n_layers,
                 n_heads,
                 pf_dim,
                 dropout,
                 device,
                 max_length=100):
        super().__init__()

        self.device = device

        self.tok_embedding = nn.Embedding(output_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)

        self.layers = nn.ModuleList([DecoderLayer(hid_dim,
                                                  n_heads,
                                                  pf_dim,
                                                  dropout,
                                                  device)
                                     for _ in range(n_layers)])

        self.fc_out = nn.Linear(hid_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

    def forward(self, trg, enc_src, trg_mask, src_mask):

        batch_size = trg.shape[0]
        trg_len = trg.shape[1]

        pos = torch.arange(0, trg_len).unsqueeze(
            0).repeat(batch_size, 1).to(self.device)

        trg = self.dropout(
            (self.tok_embedding(trg) * self.scale) + self.pos_embedding(pos))

        for layer in self.layers:
            trg, attention = layer(trg, enc_src, trg_mask, src_mask)

        output = self.fc_out(trg)

        return output, attention


class DecoderLayer(nn.Module):
    def __init__(self,
                 hid_dim,
                 n_heads,
                 pf_dim,
                 dropout,
                 device):
        super().__init__()

        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.enc_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(
            hid_dim, n_heads, dropout, device)
        self.encoder_attention = MultiHeadAttentionLayer(
            hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, pf_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        # self attention
        strg, _ = self.self_attention(trg, trg, trg, trg_mask)

        # dropout, residual connection and layer norm
        trg = self.self_attn_layer_norm(trg + self.dropout(strg))

        # encoder attention
        strg, attention = self.encoder_attention(
            trg, enc_src, enc_src, src_mask)

        # dropout, residual connection and layer norm
        trg = self.enc_attn_layer_norm(trg + self.dropout(strg))

        # positionwise feedforward
        strg = self.positionwise_feedforward(trg)

        # dropout, residual and layer norm
        trg = self.ff_layer_norm(trg + self.dropout(strg))

        return trg, attention


class Seq2Seq(nn.Module):
    def __init__(self,
                 encoder,
                 decoder,
                 src_pad_idx,
                 trg_pad_idx,
                 device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(self, src):
        return (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)

    def make_trg_mask(self, trg):
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(torch.ones(
            (trg_len, trg_len), device=self.device)).bool()
        return trg_pad_mask & trg_sub_mask

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)

        enc_src = self.encoder(src, src_mask)

        output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)

        return output, attention

def translate_sentence(sentence, src_field, trg_field, model, device, max_len=50):

    model.eval()

    if isinstance(sentence, str):
        tokens = sentence.lower().split(" ")
    else:
        tokens = [token.lower() for token in sentence]

    tokens = [src_field.init_token] + tokens + [src_field.eos_token]

    src_indexes = [src_field.vocab.stoi[token] for token in tokens]

    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)

    src_mask = model.make_src_mask(src_tensor)

    with torch.no_grad():
        enc_src = model.encoder(src_tensor, src_mask)

    trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]

    for _ in range(max_len):

        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)

        trg_mask = model.make_trg_mask(trg_tensor)

        with torch.no_grad():
            output, attention = model.decoder(
                trg_tensor, enc_src, trg_mask, src_mask)

        pred_token = output.argmax(2)[:, -1].item()

        trg_indexes.append(pred_token)

        if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
            break

    trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]

    return trg_tokens[1:], attention

# Hyperparameters
HID_DIM = 256
ENC_LAYERS = 3
DEC_LAYERS = 3
ENC_HEADS = 8
DEC_HEADS = 8
ENC_PF_DIM = 512
DEC_PF_DIM = 512
ENC_DROPOUT = 0.1
DEC_DROPOUT = 0.1
if __name__ == "__main__":
      
    # Fields
    src_lang = Field(tokenize=tokenize,
                init_token='<sos>',
                eos_token='<eos>',
                lower=True,
                batch_first=True)

    trg_lang = Field(tokenize=tokenize,
                init_token='<sos>',
                eos_token='<eos>',
                lower=True,
                batch_first=True)

    # Language to translate from
    lang = "es"

    # Load Training set
    src_train = open("corpuses/%s-en/train/train.%s" % (lang, lang), 
                    "r", encoding="utf-8").readlines()[:-200000]
    tgt_train = open("corpuses/%s-en/train/train.en" % (lang),
                    "r", encoding="utf-8").readlines()[:-200000]
    train_data = create_dataset(src_train, tgt_train, (src_lang, trg_lang))

    # Load Validating set
    src_val = open("corpuses/%s-en/validate/validate.%s" % (lang, lang), 
                   "r", encoding="utf-8").readlines()[:5000]
    tgt_val = open("corpuses/%s-en/validate/validate.en" % (lang), 
                   "r", encoding="utf-8").readlines()[:5000]
    val_data = create_dataset(src_val, tgt_val, (src_lang, trg_lang))

    # Compile vocabulary from training set
    src_lang.build_vocab(train_data, min_freq=2)
    trg_lang.build_vocab(train_data, min_freq=2)

    # Print vocab size
    print(f"{lang} vocab size: {len(src_lang.vocab)}")
    print(f"en vocab size: {len(trg_lang.vocab)}")

    # Batch size (sample taken from dataset)
    BATCH_SIZE = 16

    # Create bucketiterators for datasets
    train_iterator, valid_iterator = BucketIterator.splits(
        (train_data, val_data),
        batch_size=BATCH_SIZE,
        device=device)



    INPUT_DIM = len(src_lang.vocab)
    OUTPUT_DIM = len(trg_lang.vocab)


    enc = Encoder(INPUT_DIM,
                  HID_DIM,
                  ENC_LAYERS,
                  ENC_HEADS,
                  ENC_PF_DIM,
                  ENC_DROPOUT,
                  device)

    dec = Decoder(OUTPUT_DIM,
                  HID_DIM,
                  DEC_LAYERS,
                  DEC_HEADS,
                  DEC_PF_DIM,
                  DEC_DROPOUT,
                  device)

    SRC_PAD_IDX = src_lang.vocab.stoi[src_lang.pad_token]
    TRG_PAD_IDX = trg_lang.vocab.stoi[trg_lang.pad_token]

    model = Seq2Seq(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, device).to(device)


    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


    print(f'The model has {count_parameters(model):,} trainable parameters')


    def initialize_weights(m):
        if hasattr(m, 'weight') and m.weight.dim() > 1:
            nn.init.xavier_uniform_(m.weight.data)


    model.apply(initialize_weights)

    LEARNING_RATE = 0.0005

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)


    def train(model, iterator, optimizer, criterion, clip):

        model.train()

        epoch_loss = 0

        for i, batch in enumerate(iterator):

            if i % 5 == 0:
                print(f'TRAINING: {i}/{len(iterator)}', end="\r")

            src = batch.src
            trg = batch.trg

            optimizer.zero_grad()

            output, _ = model(src, trg[:, :-1])

            output_dim = output.shape[-1]

            output = output.contiguous().view(-1, output_dim)
            trg = trg[:, 1:].contiguous().view(-1)

            loss = criterion(output, trg)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

            optimizer.step()

            epoch_loss += loss.item()

        return epoch_loss / len(iterator)


    def evaluate(model, iterator, criterion):

        model.eval()

        epoch_loss = 0

        with torch.no_grad():

            for i, batch in enumerate(iterator):
                if i % 5 == 0:
                    print(f'EVALUATING: {i}/{len(iterator)}', end="\r")

                src = batch.src
                trg = batch.trg

                output, _ = model(src, trg[:, :-1])

                output_dim = output.shape[-1]

                output = output.contiguous().view(-1, output_dim)
                trg = trg[:, 1:].contiguous().view(-1)

                loss = criterion(output, trg)

                epoch_loss += loss.item()

        return epoch_loss / len(iterator)


    def epoch_time(start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs


    N_EPOCHS = 30
    CLIP = 1

    best_valid_loss = float('inf')

    train_model = True

    if train_model:
        for epoch in range(N_EPOCHS):

            start_time = time.time()

            train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
            valid_loss = evaluate(model, valid_iterator, criterion)

            end_time = time.time()

            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), f"model.{lang}-en.pt")

            print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
            print(
                f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
            print(
                f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')


    torch.cuda.empty_cache()
    model.load_state_dict(torch.load(f"model.{lang}-en.pt"))



    # MODEL TESTING
    src_test = open("corpuses/%s-en/test/test.%s" % (lang, lang), 
                    "r", encoding="utf-8").readlines()
    tgt_test = open("corpuses/%s-en/test/test.en" % (lang), 
                    "r", encoding="utf-8").readlines()

    f = open(f"models/{lang}-en/neural/output.{lang}-en.txt", "w", encoding="utf-8")
    for i in range(len(src_test)):
        if i % 5 == 0:
            print(f"SAVING TEST DATA: {i}/{len(src_test)}", end="\r")
        src = src_test[i]
        translation, attention = translate_sentence(src, src_lang, trg_lang, model, device)
        outtext = " ".join(translation).replace("<eos>", "") + "\n"
        f.write(outtext)
