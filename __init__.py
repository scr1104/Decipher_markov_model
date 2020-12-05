import numpy as np
import matplotlib.pyplot as plt
import string
import random
import re
import requests
import os
import textwrap
from cypher import ngram_3

# create substitution cypher
letters1 = list(string.ascii_lowercase)
letters2 = list(string.ascii_lowercase)

true_mapping = {}

# the language model
regex = re.compile('[^a-zA-Z]')

original_message = '''I then lounged down the street and found,
as I expected, that there was a mews in a lane which runs down
by one wall of the garden. I lent the ostlers a hand in rubbing
down their horses, and received in exchange twopence, a glass of
half-and-half, two fills of shag tobacco, and as much information
as I could desire about Miss Adler, to say nothing of half a dozen
other people in the neighbourhood in whom I was not in the least
interested, but whose biographies I was compelled to listen to.
'''

random.shuffle(letters2)
for k, v in zip(letters1, letters2):
    true_mapping[k] = v

# create a markov model using Moby Dick
if not os.path.exists('moby_dick.txt'):
    print("Downloading moby dick...")
    r = requests.get('https://lazyprogrammer.me/course_files/moby_dick.txt')
    with open('moby_dick.txt', 'w') as f:
        f.write(r.content.decode())

regex = re.compile('[^a-zA-Z]')
for line in open("moby_dick.txt"):
    line = line.rstrip()

    if line:
        line = regex.sub(" ", line)
        tokens = line.lower().split()

        for token in tokens:
            ch0 = token[0]
            ngram_3.update_pi(ch0)

            for ch1 in token[1:]:
                ngram_3.update_transition(ch0, ch1)
                ch0 = ch1

ngram_3.pi /= ngram_3.pi.sum()
ngram_3.M /= ngram_3.M.sum(axis=1, keepdims=True)

encoded_msg = ngram_3.encode(original_message, true_mapping)

# run an evolutionary algorithm to decode the message
dna_pool = []
for _ in range(400):
    dna = list(string.ascii_lowercase)
    random.shuffle(dna)
    dna_pool.append(dna)

# actually run the algorithm
iter_num = 1000
scores = np.zeros(iter_num)
best_dna = None
best_map = None
best_score = float('-inf')
for i in range(iter_num):
    if i > 0:
        dna_pool = ngram_3.evolve_offspring(dna_pool, 5)

    dna2score = {}
    for dna in dna_pool:
        current_map = {}
        for k, v in zip(letters1, dna):
            current_map[k] = v

        decoded_msg = ngram_3.decode(encoded_msg, current_map)
        score = ngram_3.get_sequence_prob(decoded_msg)

        dna2score["".join(dna)] = score

        if score > best_score:
            best_score = score
            best_dna = dna
            best_map = current_map

    scores[i] = np.mean(list(dna2score.values()))
    sorted_dna = sorted(dna2score.items(), key=lambda x: x[1], reverse=True)
    dna_pool = [list(k) for k, v in sorted_dna[:5]]

    if i % 200 == 0:
        print("iter:", i, "score:", scores[i], "best so far:", best_score)

decoded_msg = ngram_3.decode(encoded_msg, best_map)

print("LL of decoded message:", ngram_3.get_sequence_prob(decoded_msg))
print("LL of original message:", ngram_3.get_sequence_prob(regex.sub(" ", original_message.lower())))

for true, v in true_mapping.items():
    pred = best_map[v]
    if true != pred:
        print("true: %s, pred: %s" % (true, pred))

print("Decoded message:\n", textwrap.fill(decoded_msg))
print("\nTrue message:\n", textwrap.fill(original_message))

plt.plot(scores)
plt.show()
