#!/usr/bin/env python
import sys
import json


def main(inpath, outpath):
    docs = {}
    with open(inpath, 'r') as f:
        topic = None
        docs[topic] = []
        for line in f:
            line = line.strip()
            # Remove empty lines.
            if not line:
                continue
            # Remove all sub headers.
            if line.startswith('= = '):
                continue
            # New document found.
            if line.startswith('= ') and line.endswith(' ='):
                # Close old topic (turn back into string).
                docs[topic] = '\n'.join(docs[topic])
                # Open new topic.
                topic = line[2:-2]  # remove `= ` from both sides
                docs[topic] = []
            # Otherwise we are still in the same document.
            else:
                docs[topic].append(line)
        # Save documents as json for easy loading.
        outpath = outpath + '.json' if not outpath.endswith('.json') else outpath
        with open(outpath, 'w') as f:
            json.dump(docs, f)


if __name__ == '__main__':
    if len(sys.argv) > 2:
        inpath = sys.argv[1]
        outpath = sys.argv[2]
        main(inpath, outpath)
    else:
        exit('Specify path')
