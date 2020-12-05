import numpy as np
import re

regex = re.compile('[^a-zA-Z]')
M = np.ones((26, 26))
pi = np.zeros(26)


def update_transition(ch1, ch2):
    ch1 = ord(ch1) - 97
    ch2 = ord(ch2) - 97
    M[ch1, ch2] += 1
    return M


def update_pi(ch):
    ch = ord(ch) - 97
    pi[ch] += 1
    return pi


def get_word_prob(word):
    i = ord(word[0]) - 97
    logp = np.log(pi[i])

    for ch in range(1, len(word)-2):
        j1 = ord(word[ch]) - 97
        j2 = ord(word[ch+1]) - 97
        logp += np.log(M[i, j1])
        logp += np.log(M[j1, j2])
        i = j1

    return logp


def get_sequence_prob(words):
    if type(words) == str:
        words = words.split()

    logp = 0
    for word in words:
        logp += get_word_prob(word)

    return logp


def encode(original_msg, mapping):
    original_msg = original_msg.lower()
    original_msg = regex.sub(" ", original_msg)

    coded_msg = []
    for ch in original_msg:
        coded_ch = ch
        if ch in mapping:
            coded_ch = mapping[ch]
        coded_msg.append(coded_ch)

    return "".join(coded_msg)


def decode(encoded_msg, mapping):
    decoded_msg = []
    for ch in encoded_msg:
        decoded_ch = ch
        if ch in mapping:
            decoded_ch = mapping[ch]
        decoded_msg.append(decoded_ch)

    return "".join(decoded_msg)


def evolve_offspring(dna_pool, n_children):
    offspring = []

    for dna in dna_pool:
        for _ in range(n_children):
            copy = dna.copy()
            j = np.random.randint(len(copy))
            k = np.random.randint(len(copy))

            tmp = copy[j]
            copy[j] = copy[k]
            copy[k] = tmp
            offspring.append(copy)

    return offspring + dna_pool
