#!/usr/bin/env python
import argparse
import os
from collections import Counter
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd
from scipy.sparse.linalg import svds

from data import load_docs
from plot import emb_scatter, heatmap


def ppmi(pxy, py, px=None):
    xdim, ydim = pxy.shape
    assert py.shape == (1, ydim)
    if px is None:
        pmi = np.log(pxy) - np.log(py)
    else:
        assert px.shape == (xdim, 1)
        pmi = np.log(pxy) - np.log(px) - np.log(py)
    pmi[pmi < 0] = 0
    return pmi


def make_matrices(unigrams, bigrams, w2i):
    assert (len(unigrams) == len(w2i))
    py = np.zeros((1, len(unigrams)))
    pxy = np.zeros((len(bigrams), len(unigrams)))
    for word, i in w2i.items():
        py[0, i] = unigrams[word]
    for i, title in enumerate(tqdm(bigrams)):
        for word, prob in bigrams[title].items():
            pxy[i, w2i[word]] = prob
    return pxy, py


def write_vectors(vectors, titles, path, gensim=True):
    assert (vectors.shape[0] == len(titles))
    with open(path, 'w') as f:
        if gensim:
            print(len(titles), vectors.shape[1], file=f)
        for i, title in enumerate(titles):
            vector = vectors[i]
            title = '_'.join(title.split())  # must be one word
            line = ' '.join((str(title),) + tuple(str(val) for val in vector))
            print(line, file=f)


def main(args):
    print(f'Loading data from `{args.data}`...')
    docs = load_docs(args.data)
    titles = tuple(docs.keys())[1:]

    all_text = ''
    for title in titles:
        if not isinstance(docs[title], str):
            continue
        if args.lower:
            docs[title] = docs[title].lower()
        all_text += ' ' + docs[title]
    vocab = Counter(all_text.split())
    if args.num_words is None:
        args.num_words = len(vocab)
    counts = vocab.most_common(args.num_words)
    vocab = [word for word, _ in counts]

    print(f'Num documents: {len(titles):,}')
    print(f'Num words: {len(vocab):,}')

    total = sum(count for _, count in counts)
    unigrams = dict((word, count/total) for word, count in counts)

    bigrams = dict()
    for title in titles:
        text = (word for word in docs[title].split() if word in vocab)
        word_counts = Counter(text)
        total = sum(count for _, count in word_counts.items())
        bigrams[title] = dict((word, count/total) for word, count in word_counts.items())

    w2i = dict((word, i) for i, word in enumerate(unigrams.keys()))

    print('Making PPMI matrix...')
    pxy, py = make_matrices(unigrams, bigrams, w2i)
    if args.ppmi:
        mat = ppmi(pxy, py)
    else:
        mat = pxy
    U, s, V = svds(mat, k=args.dim)

    print('Saving results...')
    write_vectors(U, titles, args.outpath)
    emb_scatter(U, titles, model_name='wikitext-2', tsne=args.no_tsne, perplexity=args.perplexity)
    heatmap(U, 'plots/U.pdf')
    heatmap(mat, 'plots/mat.pdf')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', default='data/wikitext-2-raw.docs.json')
    parser.add_argument('--outpath', default='vec/doc.vec.txt')
    parser.add_argument('--lower', action='store_true')
    parser.add_argument('--ppmi', action='store_true')
    parser.add_argument('--no-tsne', action='store_false')
    parser.add_argument('--num-docs', type=int, default=1000)
    parser.add_argument('--num-words', type=int, default=None)
    parser.add_argument('--dim', type=int, default=100)
    parser.add_argument('--perplexity', type=int, default=30)

    args = parser.parse_args()

    main(args)
