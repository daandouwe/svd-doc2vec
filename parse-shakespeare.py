#!/usr/bin/env python
import os
import sys
import json
import csv
from tqdm import tqdm


COL = {
    'id':      0,
    'play':    1,
    'speaker': 4,
    'text':    5
}


def main(indir, outpath):
    vocab_path = os.path.join(indir, 'vocab.txt')
    with open(vocab_path) as f:
        vocab = [line.strip() for line in f.readlines()]

    play_path = os.path.join(indir, 'play_names.txt')
    with open(play_path) as f:
        plays = [line.strip() for line in f.readlines()]

    text_path = os.path.join(indir, 'will_play_text.csv')
    length = sum(1 for _ in open(text_path))
    docs = {play: ' ' for play in plays}
    with open(text_path) as f:
        reader = csv.reader(f, delimiter=';')
        for row in tqdm(reader, total=length):
            play = row[COL['play']]
            line = [word.lower() for word in row[COL['text']].split() if word.lower() in vocab]
            docs[play] += ' ' + ' '.join(line)

    # Save documents as json for easy loading.
    outpath = outpath + '.json' if not outpath.endswith('.json') else outpath
    with open(outpath, 'w') as f:
        json.dump(docs, f)


if __name__ == '__main__':
    if len(sys.argv) > 2:
        indir = sys.argv[1]
        outpath = sys.argv[2]
        main(indir, outpath)
    else:
        exit('Specify paths.')
