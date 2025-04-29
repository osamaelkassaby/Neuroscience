import numpy as np

text = "I love deep network"

words = text.split()
vocab = list(set(words))
word_to_ix = {w: i for i, w in enumerate(vocab)}
ix_to_word = {i: w for w, i in word_to_ix.items()}

X_seq = [word_to_ix[w] for w in words[:-1]]
Y_seq = word_to_ix[words[-1]]

def one_hot(idx, size):
    vec = np.zeros(size)
    vec[idx] = 1
    return vec

vocab_size = len(vocab)
hidden_size = 10
lr = 0.1

Wxh = np.random.randn(hidden_size, vocab_size) * 0.01
Whh = np.random.randn(hidden_size, hidden_size) * 0.01
Why = np.random.randn(vocab_size, hidden_size) * 0.01
bh = np.zeros((hidden_size, 1))
by = np.zeros((vocab_size, 1))

for epoch in range(500):
    h_prev = np.zeros((hidden_size, 1))
    inputs = [one_hot(ix, vocab_size).reshape(-1, 1) for ix in X_seq]
    
    hs = {}
    hs[-1] = h_prev
    for t in range(3):
        hs[t] = np.tanh(np.dot(Wxh, inputs[t]) + np.dot(Whh, hs[t-1]) + bh)

    y = np.dot(Why, hs[2]) + by
    probs = np.exp(y) / np.sum(np.exp(y))  

    loss = -np.log(probs[Y_seq])
    
    dy = probs
    dy[Y_seq] -= 1  
    dWhy = np.dot(dy, hs[2].T)
    dby = dy

    dh = np.dot(Why.T, dy)
    for t in reversed(range(3)):
        dh_raw = (1 - hs[t] ** 2) * dh
        dWxh = np.dot(dh_raw, inputs[t].T)
        dWhh = np.dot(dh_raw, hs[t-1].T)
        dbh = dh_raw

        Wxh -= lr * dWxh
        Whh -= lr * dWhh
        Why -= lr * dWhy
        bh -= lr * dbh
        by -= lr * dby
        dh = np.dot(Whh.T, dh_raw)

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

h = np.zeros((hidden_size, 1))
for word in X_seq:
    x = one_hot(word, vocab_size).reshape(-1, 1)
    h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)

y = np.dot(Why, h) + by
pred = np.argmax(y)
print("Prediction:", ix_to_word[pred])
