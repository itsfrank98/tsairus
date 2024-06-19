from os.path import join
from node_classification.graph_embeddings.node2vec import Node2VecEmbedder
from utils import embeddings_pca


def reduce_dimension(lab, model_dir, ne_dim, train_df, edge_path=None, epochs=None, n_of_walks=10, p=1, prev_models=[],
                     q=4, walk_length=10):
    """
    This function applies one of the node dimensionality reduction techniques and generate the feature vectors for
    training the decision tree.
    Args:
        :param prev_models: List containing the node2vec models learned on the previous network snapshots. It is used
         for performing graph smoothing. At each network snap, the embedding for a node learned on that snap will be
         summed to the embeddings for that same node in the previous snaps. If you don't want to perform it, leave this
         list empty
        :param lab: Label, can be either "spat" or "rel".
        :param model_dir: Directory where the models will be saved.
        :param ne_dim: Dimension of the embeddings to create.
        :param train_df: Dataframe with the training data. The IDs will be used.
        :param edge_path: Path to the list of edges used by the node embedding technique
        :param epochs: Epochs for training the node embedding model.
        :param n_of_walks: Number of walks that the n2v model will do.
        :param p: n2v's hyperparameter p.
        :param q: n2v's hyperparameter q.
        :param walk_length: Length of the walks that the n2v model will do.
    Returns:
        train_set: Array containing the node embeddings, which will be used for training the decision tree.
        train_set_labels: Labels of the training vectors.
    """
    train_set = []
    train_set_labels = []
    model_path = join(model_dir, "n2v.h5")
    weighted = False
    directed = True
    if lab == "spat":
        weighted = True
        directed = False
    n2v = Node2VecEmbedder(path_to_edges=edge_path, weighted=weighted, directed=directed, n_of_walks=n_of_walks,
                           walk_length=walk_length, embedding_size=ne_dim, p=p, q=q, epochs=epochs,
                           model_path=model_path).learn_n2v_embeddings()
    embeddings_pca(n2v, dst_dir=model_dir)
    mod = n2v.wv
    train_set_ids = [i for i in train_df['id'] if str(i) in mod.index_to_key]  # we use this cycle to keep the order of the users as they appear in the df. The same applies for the next line
    for i in train_set_ids:
        embedding_current_node = mod[str(i)]
        for m in prev_models:
            if str(i) in m.key_to_index:
                embedding_current_node = embedding_current_node + m[str(i)]
        train_set.append(embedding_current_node)
        train_set_labels.append(train_df[train_df.id == i]['label'].values[0])
    return train_set, train_set_labels
