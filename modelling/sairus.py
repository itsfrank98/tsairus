import gensim.models
import numpy as np
import torch
from gensim.models import Word2Vec
from modelling.ae import AE
from modelling.mlp import MLP
from modelling.text_preprocessing import TextPreprocessing
from modelling.word_embedding import WordEmb
from node_classification.random_forest import *
from os import makedirs
from os.path import exists, join
from node_classification.reduce_dimension import reduce_dimension
from sklearn.metrics import classification_report
from torch.nn import MSELoss
from torch.optim import Adam
from utils import load_from_pickle, plot_confusion_matrix
np.random.seed(123)


def keyedvectors_to_vec(mod: gensim.models.KeyedVectors, token_dict, emb_size=300):
    d = {}
    for u in token_dict:
        list_temp = []
        if token_dict[u]:
            for w in token_dict[u]:
                try:
                    embed_vector = mod[w]
                    list_temp.append(embed_vector)
                except KeyError:
                    list_temp.append(np.zeros(emb_size))
            list_temp = np.array(list_temp)
            list_temp = np.sum(list_temp, axis=0)
            d[u] = list_temp
            # list_tot = np.asarray(list_tot)
    return d


def learn_mlp(ae_dang, ae_safe, content_embs, field_label, model_dir, train_df, tree_rel, tree_spat, train_y, we_dim,
              rel_dim, spat_dim, n2v_rel, n2v_spat, weights=None, lr=0.0003, batch_size=128, epochs_mlp=50):
    """
    Train the MLP aimed at fusing the models
    Args:
    :param ae_dang: Dangerous autoencoder model
    :param ae_safe: Safe autoencoder model
    :param content_embs: torch tensor containing the word embeddings of the normalized content posted by the users
    :param model_dir:
    :param train_df: Dataframe with the IDs of the users in the training set
    :param tree_rel: Relational decision tree
    :param tree_spat: Spatial decision tree
    :param train_y: Train labels
    :param we_dim: dimension of the word embeddings
    :param rel_dim: dimension of relational node embeddings
    :param spat_dim: dimension of spatial node embeddings
    :param n2v_rel: relational node2vec model
    :param n2v_spat: spatial node2vec model
    :param weights: training weights
    :param lr: learning rate for the mlp
    :param batch_size: batch size for the mlp
    :param epochs_mlp: number of epochs for the mlp
    Returns: The learned MLP
    """
    #"""
    train_x = torch.zeros((content_embs.shape[0], 7))
    prediction_dang = ae_dang.predict(content_embs)
    prediction_safe = ae_safe.predict(content_embs)

    loss = MSELoss()
    prediction_loss_dang = []
    prediction_loss_safe = []
    for i in range(content_embs.shape[0]):
        prediction_loss_dang.append(loss(content_embs[i], prediction_dang[i]))
        prediction_loss_safe.append(loss(content_embs[i], prediction_safe[i]))
    labels = [1 if i < j else 0 for i, j in zip(prediction_loss_dang, prediction_loss_safe)]
    train_x[:, 0] = torch.tensor(prediction_loss_dang, dtype=torch.float32)
    train_x[:, 1] = torch.tensor(prediction_loss_safe, dtype=torch.float32)
    train_x[:, 2] = torch.tensor(labels, dtype=torch.float32)

    cmi = 1.0
    print("Computing social part")
    rel_part = predict_relational_part(df=train_df, field_label=field_label, tree=tree_rel, n2v=n2v_rel, cmi=cmi)
    train_x[:, 3], train_x[:, 4] = rel_part[:, 0], rel_part[:, 1]

    print("Computing spatial part")
    spat_part = predict_relational_part(df=train_df, field_label=field_label, tree=tree_spat, n2v=n2v_spat, cmi=cmi)
    train_x[:, 5], train_x[:, 6] = spat_part[:, 0], spat_part[:, 1]

    #save_to_pickle(join(dir, "x_train.pkl"), train_x)
    #"""
    name = "mlp_{}_rel_{}_spat_{}.pkl".format(we_dim, rel_dim, spat_dim)

    #train_x = load_from_pickle(join(dir, "x_train.pkl"))

    mlp = MLP(train_x=train_x, train_y=train_y, model_path=join(model_dir, name), weights=weights,
              batch_size=batch_size, epochs=epochs_mlp)
    optim = Adam(mlp.parameters(), lr=lr, weight_decay=1e-3)
    mlp.train_mlp(optim)
    preds = mlp(train_x)
    y_p = []
    for p in preds:
        if p[0] < p[1]:
            y_p.append(1)
        else:
            y_p.append(0)


def predict_relational_part(df, tree, field_label, n2v=None, cmi=0.5, pmi=None, prev_models=[]):
    train_x = torch.zeros(len(df), 2)

    for index, row in df.iterrows():
        id = row['id']
        try:
            forest_input = np.expand_dims(n2v.wv[str(id)], axis=0)
            for mod in prev_models:
                if str(id) in mod.key_to_index:
                    forest_input = forest_input + mod[str(id)]
                # forest_input = forest_input / len(prev_models)
            pr, conf = test_random_forest(test_set=forest_input, cls=tree)
        except KeyError:
            if not pmi:
                pmi = row[field_label]
            pr, conf = pmi, cmi
        train_x[index, 0] = torch.tensor(pr, dtype=torch.float64)
        train_x[index, 1] = torch.tensor(conf, dtype=torch.float64)
    return train_x


def train(emb_dim_rel, emb_dim_spat, field_id, field_label, model_dir, train_df, word_emb_size, users_emb_dict,
          eps_embs_rel=None, eps_embs_spat=None, mlp_batch_size=128, mlp_epochs=50, mlp_lr=0.0003, path_rel=None,
          path_spat=None, prev_n2v_rel=[], prev_n2v_spat=[], weights=None):
    """
    Builds and trains the independent modules that analyze content, social relationships and spatial relationships, and
    then fuses them with the MLP
    :param field_id: Name of the field containing the id
    :param model_dir: Directory where the models will be saved
    :param emb_dim_rel: Dimension of the relational node embeddings to learn
    :param emb_dim_spat: Dimension of the spatial node embeddings to learn
    :param train_df: Dataframe with the posts used for the MLP training
    :param word_emb_size: Dimension of the word embeddings to create
    :param users_emb_dict: Dictionary having as keys the user ids and as values their associated word embedding vector
    :param eps_embs_rel: Epochs for training the relational node embedding model
    :param eps_embs_spat: Epochs for training the spatial node embedding model
    :param mlp_batch_size: Batch size to use when learning the MLP
    :param mlp_epochs: Epochs for training the MLP
    :param mlp_lr: Learning rate for the model fusion MLP
    :param path_rel: Path to the file stating the social relationships among the users
    :param path_spat: Path to the file stating the spatial relationships among the users
    :param prev_n2v_rel: List containing the relational n2v models learned in the previous splits, for smoothing
    :param prev_n2v_spat: List containing the spatial n2v models learned in the previous splits, for smoothing
    :param weights: Training weights to give to the classes in order to compensate for data imbalance

    :return: Nothing, the learned mlp will be saved in the file "mlp.h5" and put in the model directory
    """
    y_train = list(train_df[field_label])
    #"""
    dang_posts_ids = list(train_df.loc[train_df[field_label] == 1][field_id])
    safe_posts_ids = list(train_df.loc[train_df[field_label] == 0][field_id])

    posts_embs = np.array(list(users_emb_dict.values()))
    keys = list(users_emb_dict.keys())

    dang_users_ar = np.array([users_emb_dict[k] for k in keys if k in dang_posts_ids])
    safe_users_ar = np.array([users_emb_dict[k] for k in keys if k in safe_posts_ids])
    posts_embs = torch.tensor(posts_embs, dtype=torch.float32)

    ################# TRAIN AND LOAD SAFE AND DANGEROUS AUTOENCODER ####################
    dang_ae_name = join(model_dir, "autoencoderdang_{}.pkl".format(word_emb_size))
    safe_ae_name = join(model_dir, "autoencodersafe_{}.pkl".format(word_emb_size))

    if not exists(dang_ae_name):
        dang_ae = AE(X_train=dang_users_ar, epochs=200, batch_size=64, lr=0.003, name=dang_ae_name)
        dang_ae.train_autoencoder_content()
    else:
        dang_ae = load_from_pickle(dang_ae_name)
    if not exists(safe_ae_name):
        safe_ae = AE(X_train=safe_users_ar, epochs=200, batch_size=64, lr=0.002, name=safe_ae_name)
        safe_ae.train_autoencoder_content()
    else:
        safe_ae = load_from_pickle(safe_ae_name)

    ################# TRAIN OR LOAD DECISION TREES ####################
    model_dir_rel = join(model_dir, "node_embeddings", "rel")
    model_dir_spat = join(model_dir, "node_embeddings", "spat")
    try:
        makedirs(model_dir_rel, exist_ok=False)
        makedirs(model_dir_spat, exist_ok=False)
    except OSError:
        pass
    rel_forest_path = join(model_dir_rel, "forest_{}_{}.h5".format(emb_dim_rel, word_emb_size))
    spat_forest_path = join(model_dir_spat, "forest_{}_{}.h5".format(emb_dim_spat, word_emb_size))

    x_rel, y_rel = reduce_dimension(model_dir=model_dir_rel, edge_path=path_rel, lab="rel",
                                    ne_dim=emb_dim_rel, train_df=train_df, epochs=eps_embs_rel,
                                    prev_models=prev_n2v_rel)
    train_random_forest(train_set=x_rel, dst_dir=rel_forest_path, train_set_labels=y_rel, name="rel")
    tree_rel = load_from_pickle(rel_forest_path)

    x_spat, y_spat = reduce_dimension(model_dir=model_dir_spat, edge_path=path_spat, lab="spat", ne_dim=emb_dim_spat,
                                      train_df=train_df, epochs=eps_embs_spat, prev_models=prev_n2v_spat)
    train_random_forest(train_set=x_spat, dst_dir=spat_forest_path, train_set_labels=y_spat, name="spat")
    tree_spat = load_from_pickle(spat_forest_path)

    # WE CAN NOW OBTAIN THE TRAINING SET FOR THE MLP
    n2v_rel = Word2Vec.load(join(model_dir_rel, "n2v.h5"))
    n2v_spat = Word2Vec.load(join(model_dir_spat, "n2v.h5"))
    #"""
    print("Learning MLP...\n")
    #dang_ae = safe_ae = posts_embs = id2idx_rel = id2idx_spat = x_rel = x_spat = tree_rel = tree_spat = n2v_rel = n2v_spat = None

    learn_mlp(ae_dang=dang_ae, ae_safe=safe_ae, content_embs=posts_embs, field_label=field_label, model_dir=model_dir,
              epochs_mlp=mlp_epochs, train_df=train_df, tree_rel=tree_rel, tree_spat=tree_spat, train_y=y_train,
              n2v_rel=n2v_rel, n2v_spat=n2v_spat, weights=weights, we_dim=word_emb_size, rel_dim=emb_dim_rel,
              spat_dim=emb_dim_spat, lr=mlp_lr, batch_size=mlp_batch_size)


def get_testset_dtree(idx, n2v):
    """
    provide the embedding of the node corresponding to the idx
    """
    idx = str(idx)
    mod = n2v.wv
    test_set = mod.vectors[mod.key_to_index[idx]]
    test_set = np.expand_dims(test_set, axis=0)

    return test_set


def predict_textual_part(posts_embs, ae_dang, ae_safe):
    pred_dang = ae_dang.predict(posts_embs)
    pred_safe = ae_safe.predict(posts_embs)
    loss = MSELoss()
    pred_loss_dang = []
    pred_loss_safe = []
    for i in range(posts_embs.shape[0]):
        pred_loss_dang.append(loss(posts_embs[i], pred_dang[i]))
        pred_loss_safe.append(loss(posts_embs[i], pred_safe[i]))

    labels = [1 if i < j else 0 for i, j in zip(pred_loss_dang, pred_loss_safe)]
    pred_loss_dang = torch.tensor(pred_loss_dang, dtype=torch.float32)
    pred_loss_safe = torch.tensor(pred_loss_safe, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.float32)
    return pred_loss_dang, pred_loss_safe, labels


def test(ae_dang, ae_safe, df, df_train, field_id, field_text, field_label, forest_rel, forest_spat, mlp: MLP, n2v_rel,
         n2v_spat, w2v_model):
    tok = TextPreprocessing()
    posts = tok.token_dict(df, field_text=field_text, field_id=field_id)
    test_set = torch.zeros(len(posts), 7)
    if type(w2v_model) == WordEmb:
        posts_embs_dict = w2v_model.text_to_vec(posts)
    else:
        tok = TextPreprocessing()
        token_dict = tok.token_dict(df=df, field_text=field_text, field_id=field_id)
        posts_embs_dict = keyedvectors_to_vec(w2v_model, token_dict)
    posts_embs = torch.tensor(list(posts_embs_dict.values()), dtype=torch.float32)
    pl_dang, pl_safe, labels = predict_textual_part(posts_embs, ae_dang, ae_safe)
    test_set[:, 0] = pl_dang
    test_set[:, 1] = pl_safe
    test_set[:, 2] = labels

    # At test time, if we meet an instance that doesn't have information about relationships or closeness, we will
    # replace the decision tree prediction with the most frequent label in the training set, and the confidence = 0.5
    pmi = df_train[field_label].value_counts().argmax()
    cmi = 0.5

    social_part = predict_relational_part(df=df, field_label=field_label, tree=forest_rel, n2v=n2v_rel, cmi=cmi,
                                          pmi=pmi)
    test_set[:, 3], test_set[:, 4] = social_part[:, 0], social_part[:, 1]
    spatial_part = predict_relational_part(df=df, field_label=field_label, tree=forest_spat, n2v=n2v_spat, cmi=cmi,
                                           pmi=pmi)
    test_set[:, 5], test_set[:, 6] = spatial_part[:, 0], spatial_part[:, 1]

    y_pred = mlp.test(test_set)
    y_true = np.array(df[field_label])
    plot_confusion_matrix(y_true=y_true, y_pred=y_pred)
    print(classification_report(y_true=y_true, y_pred=y_pred))


