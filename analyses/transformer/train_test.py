from importlib import reload
import torch
import transformer
reload(transformer)
from transformer import RhythmicSongformer, Songformer

from torch import nn, optim


bird = 'wh09pk88'

data = torch.load(f'data/{bird}_torch.pt')

N = len(data.keys())

train_idx = torch.randperm(N)[:int(N*0.6)]
test_idx = torch.randperm(N)[int(N*0.6):]

all_songs_tokens_flattened = torch.cat([data[i][0] for i in range(len(data))])
n_tokens = all_songs_tokens_flattened.argmax().item() + 1  # +1 since indexing is from 0
max_length = 25

model = Songformer(ntokens=n_tokens, n_heads=8, num_layers=2, context_length=max_length)

optimizer = optim.Adam(model.parameters(), lr=0.0003)

n_iters = 250
batch_size = 50
losses = []
for i in range(n_iters):
    idx = torch.randint(len(train_idx), (batch_size, ))
    songs = [data[song_idx.item()] for song_idx in idx]

    # song[0] is the token tensor; song[1] is the duration tensor

    min_song_length = min([len(song[0]) for song in songs])
    max_length_in_iter = min(max_length, min_song_length) - 1  # minus one because the sequence has to contain one more element to be predicted
    tokens      = []
    next_tokens = []
    durations   = []
    for song in songs:
        start_idx = torch.randint(low=0, high=len(song[0])-max_length_in_iter, size=(1,)).item()
        next_idx   = start_idx + max_length_in_iter
        tokens.append(song[0][start_idx:next_idx])
        next_tokens.append(song[0][next_idx])
        durations.append(song[1][start_idx:next_idx])

    tokens = torch.stack(tokens).permute(1, 0)  # in this case, permute transposes the axes
    next_tokens = torch.stack(next_tokens)
    durations = torch.stack(durations).permute(1, 0)

    loss = model.train(tokens, next_tokens) # cross-entropy loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(loss.item())
    losses.append(loss.item())

# Evaluation
x = torch.tensor([1,1,1])
z = model.forward(x)
h = model.predict(z)
