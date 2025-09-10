import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import random
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import pairwise_distances
from torch_geometric.nn import GAE
from load import load_dataset
from graph import load_graph
from os.path import join, dirname
import os
import codecs
import json
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

def pairwise_f1(gt_labels, pred_labels, *, return_stats=False):
    """
    Pairwise F1 for SND.
    GT positive: same author.
    Pred positive: same predicted cluster AND neither label is -1 (noise).
    Pairs with any noise are treated as predicted NEGATIVE.
    """
    print(gt_labels)
    print(pred_labels)
    gt = np.asarray(gt_labels)
    pr = np.asarray(pred_labels)
    assert gt.shape == pr.shape and gt.ndim == 1

    n = gt.shape[0]
    # upper-triangular pairs (i < j)
    i, j = np.triu_indices(n, k=1)

    # ground-truth: same author?
    y_true = (gt[i] == gt[j])

    # predicted: same non-noise cluster?
    same = (pr[i] == pr[j])
    non_noise = (pr[i] != -1) & (pr[j] != -1)
    y_pred = same & non_noise

    # counts
    tp = np.sum(y_true & y_pred)
    fp = np.sum(~y_true & y_pred)
    fn = np.sum(y_true & ~y_pred)

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0.0

    if return_stats:
        total_pairs = len(i)
        pos_rate = y_true.mean()  # how imbalanced the task is
        return f1, dict(tp=int(tp), fp=int(fp), fn=int(fn),
                        total_pairs=int(total_pairs), pos_rate=float(pos_rate),
                        prec=float(prec), rec=float(rec))
    return f1



def tanimoto(p, q):
    c = [v for v in p if v in q]
    return float(len(c) / (len(p) + len(q) - len(c)))

def save_results(names, pubs, results, mode):
    output = {}
    for name in names:
        output[name] = []
        name_pubs = []
        if mode == 'train':
            for aid in pubs[name]:
                name_pubs.extend(pubs[name][aid])
        else:
            for pid in pubs[name]:
                name_pubs.append(pid)

        for i in set(results[name]):
            oneauthor = []
            for idx, j in enumerate(results[name]):
                if i == j:
                    oneauthor.append(name_pubs[idx])
            output[name].append(oneauthor)
    
    result_dir = 'out'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    result_path = join(result_dir, f'res.json')   
    with codecs.open(result_path, 'w', encoding='utf-8') as wf:
        json.dump(output, wf, ensure_ascii=False, indent=4)    
    return result_path

def generate_pair(pubs, name, outlier, mode):
    dirpath = join('relations', mode, name)
    paper_org = {}
    paper_conf = {}
    paper_author = {}
    paper_word = {}

    temp = set()
    with open(dirpath + "/paper_org.txt", encoding='utf-8') as pafile:
        for line in pafile:
            temp.add(line)
    for line in temp:
        toks = line.strip().split("\t")
        if len(toks) == 2:
            p, a = toks[0], toks[1]
            if p not in paper_org:
                paper_org[p] = []
            paper_org[p].append(a)
    temp.clear()

    with open(dirpath + "/paper_venue.txt", encoding='utf-8') as pafile:
        for line in pafile:
            temp.add(line)
    for line in temp:
        toks = line.strip().split("\t")
        if len(toks) == 2:
            p, a = toks[0], toks[1]
            if p not in paper_conf:
                paper_conf[p] = []
            paper_conf[p] = a
    temp.clear()

    with open(dirpath + "/paper_author.txt", encoding='utf-8') as pafile:
        for line in pafile:
            temp.add(line)
    for line in temp:
        toks = line.strip().split("\t")
        if len(toks) == 2:
            p, a = toks[0], toks[1]
            if p not in paper_author:
                paper_author[p] = []
            paper_author[p].append(a)
    temp.clear()

    with open(dirpath + "/paper_title.txt", encoding='utf-8') as pafile:
        for line in pafile:
            temp.add(line)
    for line in temp:
        toks = line.strip().split("\t")
        if len(toks) == 2:
            p, a = toks[0], toks[1]
            if p not in paper_word:
                paper_word[p] = []
            paper_word[p].append(a)
    temp.clear()

    paper_paper = np.zeros((len(pubs), len(pubs)))
    for i, pid in enumerate(pubs):
        if i not in outlier:
            continue
        for j, pjd in enumerate(pubs):
            if j == i:
                continue
            ca = 0
            cv = 0
            co = 0
            ct = 0

            if pid in paper_author and pjd in paper_author:
                ca = len(set(paper_author[pid]) & set(paper_author[pjd])) * 1.5
            if pid in paper_conf and pjd in paper_conf and 'null' not in paper_conf[pid]:
                cv = tanimoto(set(paper_conf[pid]), set(paper_conf[pjd])) * 1.0
            if pid in paper_org and pjd in paper_org:
                co = tanimoto(set(paper_org[pid]), set(paper_org[pjd])) * 1.0
            if pid in paper_word and pjd in paper_word:
                ct = len(set(paper_word[pid]) & set(paper_word[pjd])) * 0.33

            paper_paper[i][j] = ca + cv + co + ct
            
    return paper_paper

seed = 0
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ATTGNN(nn.Module):

    ''' 
    The graph attention network (GAT) model. This is just the encoder part of the GAE!
    '''

    def __init__(self, layer_shape):
        super(ATTGNN, self).__init__()
        self.layer_shape = layer_shape
        self.conv1 = GATConv(layer_shape[0], layer_shape[1], heads=1)
        self.conv2 = GATConv(layer_shape[1], layer_shape[2], heads=1)
        self.clas_layer = nn.Parameter(torch.FloatTensor(layer_shape[-2], layer_shape[-1]))
        self.bias = nn.Parameter(torch.FloatTensor(1, layer_shape[-1]))

        nn.init.xavier_uniform_(self.clas_layer.data, gain=1.414)
        nn.init.xavier_uniform_(self.bias.data, gain=1.414)
    
    def forward(self, ft_list, adj_tensor, edge_attr=None):
        ''' This part gives us the embeddings H'' '''
        x_list = self.conv1(ft_list, adj_tensor, edge_attr)
        x_list = self.non_linear(x_list)
        x_list = self.dropout_ft(x_list, 0.5)
        x_list = self.conv2(x_list, adj_tensor, edge_attr)
        ''' This is the FCC layer for later clustering '''
        logits = torch.mm(x_list, self.clas_layer)+self.bias

        embd = x_list
        return logits, embd
        
    def non_linear(self, x_list):
        y_list = F.elu(x_list)
        return y_list
    
    def dropout_ft(self, x_list, dropout):
        y_list = F.dropout(x_list, dropout, training=self.training)
        return y_list

class BONDTrainer:
    def __init__(self) -> None:
        pass

    def onehot_encoder(self, label_list):
        """
        Transform label list to one-hot matrix.
        Arg:
            label_list: e.g. [0, 0, 1]
        Return:
            onehot_mat: e.g. [[1, 0], [1, 0], [0, 1]]
        """
        if isinstance(label_list, np.ndarray):
            labels_arr = label_list
        else:
            try:
                labels_arr = np.array(label_list.cpu().detach().numpy())
            except:
                labels_arr = np.array(label_list)
        
        num_classes = max(labels_arr) + 1
        onehot_mat = np.zeros((len(labels_arr), num_classes+1))

        for i in range(len(labels_arr)):
            onehot_mat[i, labels_arr[i]] = 1

        return onehot_mat
    
    def matx2list(self, adj):
        """
        Transform matrix to list.
        """
        adj_preds = []
        for i in adj:
            if isinstance(i, np.ndarray):
                temp = i
            else:
                temp = i.cpu().detach().numpy()
            for idx, j in enumerate(temp):
                if j == 1: 
                    adj_preds.append(idx)
                    break
                if idx == len(temp)-1:
                    adj_preds.append(-1)

        return adj_preds

    def post_match(self, pred, pubs, name, mode):
        """
        Post-match outliers.
        Args:
            pred(list): prediction e.g. [0, 0, -1, 1]
            pubs(list): paper-ids
            name(str): author name
            mode(str): train/valid/test
        Return:
            pred(list): after post-match e.g. [0, 0, 0, 1] 
        """
        #1 outlier from dbscan labels
        outlier = set()
        for i in range(len(pred)):
            if pred[i] == -1:
                outlier.add(i)

        #2 outlier from building graphs (relational)
        datapath = join('graph', mode, name)
        with open(join(datapath, 'rel_cp.txt'), 'r') as f:
            rel_outlier = [int(x) for x in f.read().split('\n')[:-1]] 

        for i in rel_outlier:
            outlier.add(i)
        
        print(f"post matching {len(outlier)} outliers")
        paper_pair = generate_pair(pubs, name, outlier, mode)
        paper_pair1 = paper_pair.copy()
        
        K = len(set(pred))

        for i in range(len(pred)):
            if i not in outlier:
                continue
            j = np.argmax(paper_pair[i])
            while j in outlier:
                paper_pair[i][j] = -1
                last_j = j
                j = np.argmax(paper_pair[i])
                if j == last_j:
                    break

            if paper_pair[i][j] >= 1.5:
                pred[i] = pred[j]
            else:
                pred[i] = K
                K = K + 1

        for ii, i in enumerate(outlier):
            for jj, j in enumerate(outlier):
                if jj <= ii:
                    continue
                else:
                    if paper_pair1[i][j] >= 1.5:
                        pred[j] = pred[i]
        return pred

    def fit(self, datatype):
        names, pubs = load_dataset(datatype)
        results = {}

        for name in names:
            print("training:", name)
            results[name] = []

            # ==== Load data ====
            label, ft_list, data = load_graph(name)
            num_cluster = int(ft_list.shape[0])
            layer_shape = []
            input_layer_shape = ft_list.shape[1]
            hidden_layer_shape = [256, 512]
            output_layer_shape = num_cluster #adjust output-layer size of FC layer.
            
            layer_shape.append(input_layer_shape)
            layer_shape.extend(hidden_layer_shape)
            layer_shape.append(output_layer_shape)
            
            # get the list of pid(paper-id)
            name_pubs = []
            if datatype == 'train':
                for aid in pubs[name]:
                    name_pubs.extend(pubs[name][aid])
            else:
                for pid in pubs[name]:
                    name_pubs.append(pid)

            # ==== Init model ====
            model = GAE(ATTGNN(layer_shape))
            ft_list = ft_list.float()
            ft_list = ft_list.to(device)
            data = data.to(device)
            model.to(device)
            
            optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=5e-4)

            cluster_losses, recon_losses, total_losses, f1s = [], [], [], []

            for epoch in range(50):
                model.train()
                optimizer.zero_grad()
                logits, embd = model.encode(ft_list, data.edge_index, data.edge_attr)

                # DBSCAN clustering
                dis = pairwise_distances(embd.cpu().detach().numpy(), metric='cosine')
                db_label = DBSCAN(eps=0.1, min_samples=5, metric='precomputed').fit_predict(dis)

                # Labels for loss
                class_matrix = torch.from_numpy(self.onehot_encoder(db_label)).to(device)
                local_label = torch.mm(class_matrix, class_matrix.t()).float().to(device)
                global_label = torch.matmul(logits, logits.t())

                # Losses
                loss_cluster = F.binary_cross_entropy_with_logits(global_label, local_label)
                loss_recon = model.recon_loss(embd, data.edge_index)
                loss_train = 0.5 * loss_cluster + 0.5 * loss_recon

                # === Log running losses ===
                cluster_losses.append(loss_cluster.item())
                recon_losses.append(loss_recon.item())
                total_losses.append(loss_train.item())

                # === Running F1 ===
                gt_labels = label.cpu().numpy()
                f1 = pairwise_f1(gt_labels, db_label)
                f1s.append(f1)
                
                if epoch % 5 == 0:
                    print(
                        f'epoch: {epoch:3d} | cluster loss: {loss_cluster.item():.4f} '
                        f'recon loss: {loss_recon.item():.4f} | total loss: {loss_train.item():.4f} '
                        f'| F1: {f1:.3f}'
                    )

                loss_train.backward()
                optimizer.step()

                
                        
            # ==== Evaluate ====
            # ================= Plot after training =================
            plt.figure(figsize=(12,4))

            plt.subplot(1,2,1)
            plt.plot(cluster_losses, label="Cluster loss")
            plt.plot(recon_losses, label="Reconstruction loss")
            plt.plot(total_losses, label="Total loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.title(f"Loss curves - {name}")

            plt.subplot(1,2,2)
            plt.plot(f1s, label="Pairwise F1")
            plt.xlabel("Epoch")
            plt.ylabel("F1-score")
            plt.legend()
            plt.title(f"Clustering F1 vs GT - {name}")

            plt.tight_layout()
            plt.show()
            with torch.no_grad():
                model.eval()
                logits, embd = model.encode(ft_list, data.edge_index, data.edge_attr)
                ''' Get the predicted matrix from C@C.t() '''
                global_label = torch.matmul(logits, logits.t())
                ''' Get the pairwise distances in embd space '''
                lc_dis = pairwise_distances(embd.cpu().detach().numpy(), metric='cosine')
                ''' Fit a DBSCAN on local distances (from embd)'''
                local_label = DBSCAN(eps=0.5, min_samples=5, metric='precomputed').fit_predict(lc_dis) 
                ''' Get the pairwise distances in logits space '''
                gl_dis = pairwise_distances(global_label.cpu().detach().numpy(), metric='cosine')
                ''' Fit a dbscan on global distances (from logits) '''
                global_label = DBSCAN(eps=0.5, min_samples=5, metric='precomputed').fit_predict(gl_dis) 
                pred = []           
                # change to one-hot form
                class_matrix = torch.from_numpy(self.onehot_encoder(local_label))
                # get N * N matrix
                local_label = torch.mm(class_matrix, class_matrix.t())
                pred = self.matx2list(local_label)
                pred = self.post_match(pred, name_pubs, name, datatype)
                # Save results
                results[name] = pred

        result_path = save_results(names, pubs, results)
        print("Done! Results saved:", result_path)