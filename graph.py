import os
import re
from tqdm import tqdm
from os.path import join
import json
import numpy as np
import torch
import pickle
import random
from torch_geometric.data import Data
import string

def save_emb(mode, name, pubs, save_path):
    # build mapping of paper & index-id
    mapping = dict()
    for idx, pid in enumerate(pubs):
        mapping[idx] = pid

    # load paper embedding
    with open(join('paper_emb', mode, name, 'ptext_emb.pkl'), 'rb') as f:
        ptext_emb = pickle.load(f)

    # init node feature matrix(n * dim_size)
    ft = dict()
    for pidx_1 in mapping:
        pid_1 = mapping[pidx_1]
        ft[pidx_1] = torch.from_numpy(ptext_emb[pid_1])

    feats_file_path = join(save_path, 'feats_p.npy')
    np.save(feats_file_path, ft)

def gen_relations(name, mode, target):
    """
    Generates a mapping from paper IDs to related entities (authors, organizations, or venues) based on the specified target.
    Args:
        target (str): The type of relation to generate. Must be one of 'author', 'org', or 'venue'.
    Returns:
        dict: A dictionary mapping paper IDs (str) to a list of related entity IDs (str), depending on the target.
    Raises:
        FileNotFoundError: If the corresponding relation file does not exist.
        ValueError: If the target is not one of the supported types.
    IMPORTANT:
        The function expects the existence of files named 'paper_author.txt', 'paper_org.txt',
        or 'paper_venue.txt'in the directory specified by 'dirpath'.
        Each file should contain tab-separated pairs of paper and entity IDs.
    """

    temp = set()
    paper_info = dict()
    info_paper = dict()

    if target == 'author':
        filename = "paper_author.txt"
    elif target == 'org':
        filename = "paper_org.txt"
    elif target == 'venue':
        filename = "paper_venue.txt"

    dirpath =  join('relations', mode, name)
    with open(join(dirpath, filename), 'r', encoding='utf-8') as f:
        for line in f:
            temp.add(line)

    for line in temp:
        toks = line.strip().split("\t")
        if len(toks) == 2:
            p, a = toks[0], toks[1]
            if p not in paper_info:
                paper_info[p] = []
            paper_info[p].append(a)

    temp.clear()
    return paper_info

def save_label_pubs(mode, name, raw_pubs, save_path):
    """
    Processes and saves publication labels based on the specified mode.
    In "train" mode, assigns integer labels to each publication ID grouped by author,
    saves the label dictionary as a NumPy file, and returns a list of all publication IDs.
    In other modes, returns a list of publication IDs for the given name.
    Args:
        mode (str): Operation mode, either "train" or another value.
        name (str): Key to select a subset from raw_pubs.
        raw_pubs (dict): Nested dictionary containing publication data, structured as raw_pubs[name][author_id] = [pub_ids].
    Returns:
        list: List of publication IDs for the specified name.
    """
    os.makedirs(save_path, exist_ok=True)

    if mode in ["train", "valid"]:
        label_dict = {}
        pubs = []
        ilabel = 0
        for aid in raw_pubs[name]:
            pubs.extend(raw_pubs[name][aid])  
            for pid in raw_pubs[name][aid]:
                label_dict[pid] = ilabel
            ilabel += 1
        file_path = join(save_path, "p_label.npy")
        np.save(file_path, label_dict)

    else:
        pubs = []
        for pid in raw_pubs[name]:
            pubs.append(pid)

    return pubs

def save_graph(name, pubs, save_path, mode):

    """
    Generates and saves graph-based relationships and outlier information for a set of publications.
    This function computes relationships between publications based on shared authors, organizations, and venues.
    It writes adjacency and attribute information to 'adj_attr.txt' and marks outlier papers (those without coauthors
    and co-organizations) in 'rel_cp.txt' within the specified save path.
    Args:
        name (str): The identifier or name used to generate relationships.
        pubs (iterable): A collection of publication IDs to be included in the graph.
        save_path (str): The directory path where output files will be saved.
        mode (str): The mode or context for generating relationships (passed to `gen_relations`).
    Output Files:
        adj_attr.txt: Contains tab-separated values for each pair of publications with their relationship attributes.
        rel_cp.txt: Contains indices of publications identified as outliers (without coauthors and co-organizations).
   """

    # init node mapping & edge mapping
    paper_dict = {pid: idx for idx, pid in enumerate(pubs)}

    cp_a, cp_o = set(), set()   # Outliers

    paper_rel_ath = gen_relations(name, mode, 'author')
    paper_rel_org = gen_relations(name, mode, 'org')
    paper_rel_ven = gen_relations(name, mode, 'venue')

    for pid in paper_dict:
        if pid not in paper_rel_ath:
            cp_a.add(paper_dict[pid])

    for pid in paper_dict:
        if pid not in paper_rel_org:
            cp_o.add(paper_dict[pid])

    # mark paper w/o coauthor and coorg as outlier
    cp = cp_a & cp_o

    with open(join(save_path, 'adj_attr.txt'), 'w') as f:

        for p1 in paper_dict:
            p1_idx = paper_dict[p1]
            for p2 in paper_dict:
                p2_idx = paper_dict[p2]
                if p1 != p2:
                    co_aths, co_orgs, co_vens = 0, 0 ,0
                    org_attr, org_attr_jaccard, org_jaccard2, ven_attr, ven_attr_jaccard, venue_jaccard2 = 0, 0, 0, 0, 0, 0
                    org_idf_sum, org_idf_sum1, org_idf_sum2, ven_idf_sum, ven_idf_sum1, ven_idf_sum2, co_org_idf, co_ven_idf = 0, 0, 0, 0, 0, 0, 0, 0
                    co_org_idf_2, co_ven_idf_2 = 0, 0
                    if p1 in paper_rel_ath:
                        for k in paper_rel_ath[p1]:
                            if p2 in paper_rel_ath:
                                if k in paper_rel_ath[p2]:
                                    co_aths += 1

                    if p1 in paper_rel_org:
                        for k in paper_rel_org[p1]:
                            if p2 in paper_rel_org:
                                if k in paper_rel_org[p2]:
                                    co_orgs += 1

                    if p1 in paper_rel_ven:
                        for k in paper_rel_ven[p1]:
                            if p2 in paper_rel_ven:
                                if k in paper_rel_ven[p2]:
                                    co_vens += 1

                    if co_orgs > 0:
                        all_words_p1 = len(paper_rel_org[p1])
                        all_words_p2 = len(paper_rel_org[p2])
                        org_attr_jaccard = co_orgs/(all_words_p1+all_words_p2-co_orgs)

                    if co_vens>0:
                        all_words_p1 = len(paper_rel_ven[p1])
                        all_words_p2 = len(paper_rel_ven[p2])
                        ven_attr_jaccard = co_vens / (all_words_p1+all_words_p2-co_vens)

                    if (co_aths + co_orgs) > 0:
                        f.write(f'{p1_idx}\t{p2_idx}\t{co_aths}\t'
                                f'{co_orgs}\t{org_attr_jaccard}\t'
                                f'{co_vens}\t{ven_attr_jaccard}\n')


    f.close()

    with open(join(save_path, 'rel_cp.txt'), 'w') as out_f:
        for i in cp:
            out_f.write(f'{i}\n')

    out_f.close()

def build_graph():
    """
    Processes datasets for different modes ('train', 'valid', 'test') by loading corresponding JSON files,
    iterating through each publication, and generating graph-related data.
    For each mode:
        - Loads the dataset from a JSON file.
        - Iterates over each publication name in the dataset.
        - Creates a directory for saving graph data if it does not exist.
        - Calls functions to save labeled publications, graph structures, and embeddings for each publication.
        - Creates directories and writes files to disk under the 'graph' directory for each mode and publication.
    """

    for mode in ["train", "valid", "test"]:
        
        print("preprocess dataset: ", mode)
        with open(join(os.path.dirname(__file__), f"data_{mode}.json"), "r", encoding="utf-8") as f:
                raw_pubs = json.load(f)[f'data_{mode}'][0]

        for name in tqdm(raw_pubs):

            save_path = join('graph', mode, name)
            os.makedirs(save_path, exist_ok=True)
            pubs = save_label_pubs(mode, name, raw_pubs, save_path)
            save_graph(name, pubs, save_path, mode)
            save_emb(mode, name, pubs, save_path)

def load_graph(name, mode='train', rel_on='aov', th_a=0, th_o=0.5, th_v=2, p_v=0.9):
    """
    Args:
        name(str): author
        th_a(int): threshold of coA
        th_o(float): threshold of coO
        th_v(int): threshold of coV
    Returns:
        label(list): true label
        ft_tensor(tensor): node feature
        data(Pyg Graph Data): graph
    """
    data_path = 'graph' 
    datapath = join(data_path, mode, name)

    # Load label
    if mode in ["train", "valid"]:
        p_label = np.load(join(datapath, 'p_label.npy'), allow_pickle=True)
        p_label_list = []
        for pid in p_label.item():
            p_label_list.append(p_label.item()[pid])
        label = torch.LongTensor(p_label_list)

    else:
        label = []

    # Load node feature
    feats = np.load(join(datapath, 'feats_p.npy'), allow_pickle=True)
    ft_list = []
    for idx in feats.item():
        ft_list.append(feats.item()[idx])
    ft_tensor = torch.stack(ft_list) # size: N * feature dimension

    # Load edge
    temp = set()
    with open(join(datapath, 'adj_attr.txt'), 'r', encoding='utf-8') as f:
        for line in f:
            temp.add(line)

    srcs, dsts, value, attr = [], [], [], []
    for line in temp:
        toks = line.strip().split("\t")
        if len(toks) == 7:
            src, dst = int(toks[0]), int(toks[1])
            val_a, val_o, val_v = int(toks[2]), int(toks[3]), int(toks[5])
            attr_o, attr_v = float(toks[4]), float(toks[6])
        else:
            print('read adj_attr ERROR!\n')

        if rel_on == 'a':
            if val_a > th_a:
                srcs.append(src)
                dsts.append(dst)
                value.append(val_a)
                attr.append(val_a)
        elif rel_on == 'o':
            if val_o > th_o:
                srcs.append(src)
                dsts.append(dst)
                value.append(val_o)
                attr.append(val_o)
        elif rel_on == 'v':
            if val_v > th_v:
                srcs.append(src)
                dsts.append(dst)
                value.append(val_v)
                attr.append(val_v)
        elif rel_on == 'aov':
            prob_v = random.random()
            if (prob_v >= p_v):
                val_v = val_v
            else:
                val_v = 0

            if attr_o >= th_o:
                val_o = val_o
            else:
                val_o = 0

            if (val_a > th_a) and (val_o > th_o) and (val_v > th_v): #a, o, v
                srcs.append(src)
                dsts.append(dst)
                value.append(val_a+val_o+val_v)
                attr.append([float(val_a), float(attr_o), float(attr_v)])
            elif (val_a > th_a) and (val_o > th_o) and (val_v <= th_v): #a, o
                srcs.append(src)
                dsts.append(dst)
                value.append(val_a+val_o)
                attr.append([float(val_a), float(attr_o), 0])
            elif (val_a > th_a) and (val_o <= th_o) and (val_v > th_v): #a, v
                srcs.append(src)
                dsts.append(dst)
                value.append(val_a+val_v)
                attr.append([float(val_a), 0, float(attr_v)])
            elif (val_a > th_a) and (val_o <= th_o) and (val_v <= th_v): #a
                srcs.append(src)
                dsts.append(dst)
                value.append(val_a)
                attr.append([float(val_a), 0, 0])
            elif (val_a <= th_a) and (val_o > th_o) and (val_v > th_v): #o, v
                srcs.append(src)
                dsts.append(dst)
                value.append(val_o+val_v)
                attr.append([0, float(attr_o), float(attr_v)])
            elif (val_a <= th_a) and (val_o > th_o) and (val_v <= th_v): #o
                srcs.append(src)
                dsts.append(dst)
                value.append(val_o)
                attr.append([0, float(attr_o), 0])
            elif (val_a <= th_a) and (val_o <= th_o) and (val_v > th_v): #v
                srcs.append(src)
                dsts.append(dst)
                value.append(val_v)
                attr.append([0, 0, float(attr_v)])

        else:
            print('wrong relation set\n')
            break

    temp.clear()

    # Build graph
    edge_index = torch.cat([torch.tensor(srcs).unsqueeze(0), torch.tensor(dsts).unsqueeze(0)], dim=0)
    edge_attr = torch.tensor(attr, dtype=torch.float32)
    edge_weight = torch.tensor(value, dtype=torch.float32)
    data = Data(edge_index=edge_index, edge_attr=edge_attr, edge_weight=edge_weight)

    return label, ft_tensor, data