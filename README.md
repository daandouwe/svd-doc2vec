# Doc2vec with PPMI-SVD


## Setup
We use the [WikiText dataset](https://einstein.ai/research/the-wikitext-long-term-dependency-language-modeling-dataset).

To extract documents from WikiText and save as json file, run:
```bash
mkdir data
./parse-wikitext.py wikitext-2-raw/wiki.train.raw data/wikitext-2-raw.docs.json
```

## Usage
In the project terminal, run
```bash
./main.py --data data/wikitext-2-raw.docs.json --outpath vec/wikitext-2-raw.vec.txt \
    --lower --num-words 1000 --dim 10
```
for a quick demo. Plots are saved in the folder `plots`.

To rank the documents based on the vectors, use:
```bash
./rank.py vec/wikitext-2-raw.vec.txt > wikitext-2-raw.ranking.txt
```


## Requirements
```
numpy
scipy
tqdm
matplotlib
sklearn
bokeh
```
