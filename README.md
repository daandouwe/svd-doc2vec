# Doc2vec with PPMI-SVD


## Setup
We use the [WikiText dataset](https://einstein.ai/research/the-wikitext-long-term-dependency-language-modeling-dataset).

To extract documents from WikiText and save as json file, run:
```bash
cd data
./parse-docs.py wikitext-2-raw/wiki.train.tokens wikitext-2-raw.docs.json
```

## Usage
In the project terminal, run
```bash
./main.py --data wikitext-2-raw.docs.json --lower --num-words 1000 --dim 10
```
for a small test.

Plots results are saved in folder `vec` and plots in folder `plots`.

## Requirements
```
numpy
scipy
tqdm
matplotlib
sklearn
bokeh
```
