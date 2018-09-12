import json


def clean(title):
    title = title.replace('( ', '(')
    title = title.replace(' )', ')')
    title = title.replace(' @-@ ', '-')
    title = title.replace(' :', ':')
    title = title.replace(' ;', ';')
    title = title.replace(' ,', ',')
    title = title.replace(' .', '.')
    title = title.replace(' !', '!')
    title = title.replace(' ?', '?')
    title = title.replace('" ', '"')
    title = title.replace(' "', '"')
    title = title.replace("' ", "'")
    title = title.replace(" '", "'")
    return title


def load_docs(path):
    with open(path) as f:
        docs = json.load(f)
    return dict((clean(title), text) for title, text in docs.items())
