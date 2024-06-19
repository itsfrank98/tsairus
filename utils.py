import matplotlib.pyplot as plt
import pickle
import seaborn as sn
from os.path import join
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix


def save_to_pickle(name, c):
    with open(name, 'wb') as f:
        pickle.dump(c, f)


def load_from_pickle(name):
    with open(name, 'rb') as f:
        return pickle.load(f)


def embeddings_pca(emb_model, dst_dir):
    vectors = emb_model.wv.vectors
    k2i = emb_model.wv.key_to_index
    pca = PCA(n_components=2, random_state=42)
    pca_embs = pca.fit_transform(vectors)
    d = {}
    for k in k2i:
        d[k] = pca_embs[k2i[k]]
    save_to_pickle(join(dst_dir, "reduced_embs.pkl"), d)


def write_edgelist_to_file(dst, edges, spatial=False):
    with open(dst, 'w') as f:
        for l in edges:
            if not spatial:
                f.write("{}\t{}\n".format(l[0], l[1]))
            else:
                f.write("{}\t{}\t{}\n".format(l[0], l[1], l[2]))


def is_square(m):
    return m.shape[0] == m.shape[1]


def plot_confusion_matrix(y_true, y_pred):
    mat = confusion_matrix(y_true=y_true, y_pred=y_pred)
    plt.figure(figsize=(6, 4))
    ax = plt.subplot()
    sn.heatmap(mat, annot=True, cmap="CMRmap", linewidths=0.5, cbar=False, fmt="d", ax=ax)

    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(['0', '1'])
    ax.yaxis.set_ticklabels(['0', '1'])

    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    plt.show()
