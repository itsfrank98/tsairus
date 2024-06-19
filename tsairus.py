import os
import json
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import classification_report
from modelling.sairus import keyedvectors_to_vec, predict_textual_part, predict_relational_part
from tqdm import tqdm
from gensim.models import KeyedVectors, Word2Vec
from modelling.sairus import train
from modelling.text_preprocessing import TextPreprocessing
from os.path import exists, join
from utils import save_to_pickle, load_from_pickle, plot_confusion_matrix

seed = 123
np.random.seed(seed)


def create_unique_dataframes(dataset_dir, field_id, field_text, field_label):
    """
    The dataset is divided in splits and each split contains files having for each row a tweet. This function puts all
    together by concatenating the tweets associated to a user and creating a dataframe for each split
    """
    for split in os.listdir(dataset_dir):
        dangerous = []
        #inv_matches = {v: k for k, v in d_matches.items()}
        d_users = {}
        g = []
        for f in os.listdir(join(dataset_dir, split, "new_tweets")):
            with open(join(dataset_dir, split, "new_tweets", f), 'r') as f:
                for line in f.readlines():
                    uid, text = line.split("\t")
                    try:
                        if uid in d_users:
                            if text.strip():    # check if the text contains something or is empty
                                d_users[uid] += " " + text.strip()
                        else:
                            if text.strip():
                                d_users[uid] = text.strip()
                    except KeyError:
                        g.append(uid)

        df = pd.DataFrame(d_users.items(), columns=[field_id, field_text])
        labels = [0] * len(df)

        # I retrieve the dangerous users
        for file in os.listdir(join(dataset_dir, split, "new_dang")):
            with open(join(dataset_dir, split, "new_dang", file), 'r') as f2:
                for line in f2.readlines():
                    node = json.loads(line)['node']
                    dangerous.append(str(node))

        # I scan the dataframe and if a user is dangerous I set the corresponding element in labels to 1
        for i, r in df.iterrows():
            id = str(r['id'])
            if id in dangerous:
                labels[i] = 1

        df.insert(1, field_label, labels)
        df.to_csv(join(dataset_dir, split, "dataframe_origids.csv"))


def main_train(content_df, epochs_rel, epochs_spat, field_id, field_text, field_label, mlp_batch_size, mlp_lr,
               ne_dim_rel, ne_dim_spat, path, social_net_name, spatial_net_name, smoothing, w2v_mod, word_emb_size):
    prev_n2v_rel = []
    prev_n2v_spat = []

    for split in tqdm(os.listdir(path)[:-1]):
        path_rel = join(path, split, social_net_name)
        path_spat = join(path, split, spatial_net_name)
        models_dir = join(path, split, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir, exist_ok=True)
        df = pd.read_csv(join(path, split, content_df))
        if not exists(join(path, split, "uembs.pkl")):
            tok = TextPreprocessing()
            token_dict = tok.token_dict(df=df, field_text=field_text, field_id=field_id)
            users_embs_dict = keyedvectors_to_vec(w2v_mod, token_dict, word_emb_size)
            save_to_pickle(join(path, split, "uembs.pkl"), users_embs_dict)
        users_embs_dict = load_from_pickle(join(path, split, "uembs.pkl"))
        nz = len(df[df[field_label] == 1])
        pos_weight = len(df) / nz
        neg_weight = len(df) / (2 * (len(df) - nz))
        #pos_weight = neg_weight = 1.0
        train(train_df=df, model_dir=models_dir, field_id=field_id, path_rel=path_rel, path_spat=path_spat,
              word_emb_size=word_emb_size, emb_dim_spat=ne_dim_spat, emb_dim_rel=ne_dim_rel, eps_embs_spat=epochs_spat,
              eps_embs_rel=epochs_rel, users_emb_dict=users_embs_dict, prev_n2v_rel=prev_n2v_rel,
              prev_n2v_spat=prev_n2v_spat, mlp_lr=mlp_lr, mlp_batch_size=mlp_batch_size,
              weights=torch.tensor([neg_weight, pos_weight]), field_label=field_label)
        if smoothing:
            n2v_model = Word2Vec.load(join(path, split, "models", "node_embeddings", "rel", "n2v.h5"))
            # I load the n2v model learned in the current split and append it to the list of models for the previous splits
            prev_n2v_rel.append(n2v_model.wv)
            n2v_model = Word2Vec.load(join(path, split, "models", "node_embeddings", "spat", "n2v.h5"))
            prev_n2v_spat.append(n2v_model.wv)


def main_test(dir, w2v, field_id, field_text, field_label, wemb_size, rel_size, spat_size, modality, smoothing=False):
    splits = os.listdir(dir)
    df_test = pd.read_csv(join(dir, splits[-1], "dataframe_origids.csv"))
    tok = TextPreprocessing()
    token_dict = tok.token_dict(df_test, field_text=field_text, field_id=field_id)
    posts_embs_dict = keyedvectors_to_vec(mod=w2v, token_dict=token_dict, emb_size=wemb_size)
    dict_preds = {}     # Dictionary that will keep track of the predictions made for each user in the different training splits
    prev_n2v_rel = prev_n2v_spat = []
    for en, split in enumerate(splits[:-1]):
        model_dir = join(dir, split, "models")
        df = pd.read_csv(join(dir, split, "dataframe_origids.csv"))
        ids = list(df[field_id])
        ae_dang = load_from_pickle(join(model_dir, "autoencoderdang_{}.pkl".format(wemb_size)))
        ae_safe = load_from_pickle(join(model_dir, "autoencodersafe_{}.pkl".format(wemb_size)))
        n2v_rel = load_from_pickle(join(model_dir, "node_embeddings", "rel", "n2v.h5"))
        n2v_spat = load_from_pickle(join(model_dir, "node_embeddings", "spat", "n2v.h5"))
        forest_rel = load_from_pickle(join(model_dir, "node_embeddings", "rel", "forest_{}_{}.h5".format(rel_size, wemb_size)))
        forest_spat = load_from_pickle(
            join(model_dir, "node_embeddings", "spat", "forest_{}_{}.h5".format(spat_size, wemb_size)))
        mlp = load_from_pickle(join(model_dir, "mlp_{}_rel_{}_spat_{}.pkl".format(wemb_size, rel_size, spat_size)))
        for k in token_dict:
            if k in ids:
                test_ar = torch.zeros(7)
                textual_part = predict_textual_part(torch.reshape(torch.tensor(posts_embs_dict[k], dtype=torch.float32),
                                                                  (1, -1)), ae_dang, ae_safe)
                test_ar[0], test_ar[1], test_ar[2] = textual_part[0][0], textual_part[1][0], textual_part[2][0]
                rel_part = predict_relational_part(df=df[df.id == k].reset_index(), pmi=textual_part[2][0],
                                                   tree=forest_rel, n2v=n2v_rel, prev_models=prev_n2v_rel,
                                                   field_label=field_label)

                test_ar[3], test_ar[4] = rel_part[0, 0], rel_part[0, 1]
                spat_part = predict_relational_part(df=df[df.id == k].reset_index(), pmi=textual_part[2][0],
                                                    tree=forest_spat, n2v=n2v_spat, prev_models=prev_n2v_spat,
                                                    field_label=field_label)
                test_ar[5], test_ar[6] = spat_part[0, 0], spat_part[0, 1]
                pred = mlp.test(torch.reshape(test_ar, (1, -1)))[0]
                value = 1
                if modality == "linear":
                    value = en + 1
                elif modality == "quadratic":
                    value = (en + 1) ** 2
                if k not in dict_preds:
                    dict_preds[k] = {0: 0, 1: 0}
                dict_preds[k][pred] += value
        if smoothing:
            prev_n2v_rel.append(n2v_rel.wv)
            prev_n2v_spat.append(n2v_spat.wv)
    pred = np.zeros(len(dict_preds))
    y_true = np.zeros(len(dict_preds))
    for i, el in enumerate(dict_preds):
        if dict_preds[el][1] >= dict_preds[el][0]:
            pred[i] = 1
        y_true[i] = df_test[df_test.id == el][field_label].values[0]
    plot_confusion_matrix(y_true=y_true, y_pred=pred)
    print(classification_report(y_true=y_true, y_pred=pred))


if __name__ == "__main__":
    with open("parameters.yaml", 'r') as params_file:
        params = yaml.safe_load(params_file)
        params = params["params"]
    df_name = params["df_name"]
    epochs_rel = params["epochs_rel"]
    epochs_spat = params["epochs_spat"]
    field_id = params["field_id"]
    field_text = params["field_text"]
    field_label = params["field_label"]
    mlp_batch_size = int(params["mlp_batch_size"])
    mlp_lr = float(params["mlp_lr"])
    ne_dim_rel = int(params["ne_dim_rel"])
    ne_dim_spat = int(params["ne_dim_spat"])
    social_net_name = params["social_net_name"]
    spatial_net_name = params["spatial_net_name"]
    dir = params["dir"]
    w2v_path = params["w2v_path"]
    smooth = bool(params["smooth"])
    word_emb_size = int(params["word_emb_size"])
    #create_unique_dataframes(dataset_dir, field_id, field_text)

    w2v_mod = KeyedVectors.load_word2vec_format(w2v_path, binary=True)
    main_train(content_df=df_name, epochs_rel=epochs_rel, epochs_spat=epochs_spat, field_id=field_id,
               field_text=field_text, field_label=field_label, mlp_batch_size=mlp_batch_size, mlp_lr=mlp_lr,
               ne_dim_rel=ne_dim_rel, ne_dim_spat=ne_dim_spat, path=dir, social_net_name=social_net_name,
               spatial_net_name=spatial_net_name, smoothing=smooth, w2v_mod=w2v_mod, word_emb_size=word_emb_size)
    for mod in ["uniform", "linear", "quadratic"]:
        print(mod)
        main_test(dir=dir, w2v=w2v_mod, field_text=field_text, field_id=field_id, field_label=field_label,
                  wemb_size=300, rel_size=128, spat_size=128, smoothing=smooth, modality=mod)
