#!/usr/bin/env python
import argparse
import os

from gensim.models import KeyedVectors


def main(args):
    path = os.path.join(args.vecs)
    vectors = KeyedVectors.load_word2vec_format(path, binary=False)
    plays = vectors.vocab.keys()
    for play in sorted(plays):
        similar = vectors.most_similar(positive=play, topn=10)
        print(play.replace('_', ' '))
        for word, val in similar:
            print('\t', word.replace('_', ' '), val)
        print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--vecs', default='vec/doc.vec.txt')

    args = parser.parse_args()

    main(args)
