#!/usr/bin/env python
import argparse
import os
import json
import csv
from tqdm import tqdm
from collections import defaultdict


COLLUMNS = {
    'id':      0,
    'play':    1,
    'speaker': 4,
    'text':    5
}


def main(args):
    vocab_path = os.path.join(args.indir, 'vocab.txt')
    with open(vocab_path) as f:
        vocab = [line.strip() for line in f.readlines()]

    text_path = os.path.join(args.indir, 'will_play_text.csv')
    length = sum(1 for _ in open(text_path))
    docs = defaultdict(str)
    with open(text_path) as f:
        reader = csv.reader(f, delimiter=';')
        for row in tqdm(reader, total=length):
            doc = row[COLLUMNS[args.doc]]
            line = [word.lower() for word in row[COLLUMNS['text']].split() if word.lower() in vocab]
            docs[doc] += ' ' + ' '.join(line)

    # Save documents as json for easy loading.
    args.outpath = args.outpath + '.json' if not args.outpath.endswith('.json') else args.outpath
    with open(args.outpath, 'w') as f:
        json.dump(docs, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('indir')
    parser.add_argument('outpath')
    parser.add_argument('--doc', default='play', choices=['play', 'speaker'])
    args = parser.parse_args()

    main(args)
