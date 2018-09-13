import os

from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from bokeh.models import ColumnDataSource, LabelSet
from bokeh.plotting import figure, save, output_file
from bokeh.palettes import d3
import matplotlib.pyplot as plt


def heatmap(mat, path):
    fig, ax = plt.subplots()
    ax.imshow(mat)
    plt.savefig(path)


def emb_scatter(data, names, model_name, tsne=True, perplexity=30.0, k=20):
    """t-SNE plot of embeddings and coloring with K-means clustering.

    Uses t-SNE with given perplexity to reduce the dimension of the
    vectors in data to 2, plots these in a bokeh 2d scatter plot,
    and colors them with k colors using K-means clustering of the
    originial vectors. The colored dots are tagged with labels from
    the list names.

    Args:
        data (np.Array): the word embeddings shape [num_vectors, embedding_dim]
        names (list): num_vectors words same order as data
        perplexity (float): perplexity for t-SNE
        k (int): number of clusters to find by K-means
    """
    # Find clusters with kmeans.
    print('Finding clusters...')
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(data)
    klabels = kmeans.labels_

    # Get a tsne fit.
    if tsne:
        print('Fitting t-SNE...')
        tsne = TSNE(n_components=2, perplexity=perplexity)
        emb = tsne.fit_transform(data)
    else:
        emb = data

    # Plot the t-SNE of the embeddings with bokeh,
    # source: https://github.com/oxford-cs-deepnlp-2017/practical-1
    fig = figure(tools='pan,wheel_zoom,reset,save',
               toolbar_location='above',
               title='T-SNE for most common words')

    # Set colormap as a list.
    colormap = d3['Category20'][k]
    colors = [colormap[i] for i in klabels]

    source = ColumnDataSource(data=dict(x1=emb[:,0],
                                        x2=emb[:,1],
                                        names=names,
                                        colors=colors))

    fig.scatter(x='x1', y='x2', size=8, source=source, color='colors')

    labels = LabelSet(x='x1', y='x2', text='names', y_offset=6,
                      text_font_size='8pt', text_color='#555555',
                      source=source, text_align='center')
    fig.add_layout(labels)

    output_file(os.path.join('plots', f'{model_name}.tsne.html'))
    save(fig)
