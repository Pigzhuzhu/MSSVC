#CUDA_VISIBLE_DEVICES=0 /remote-home/mzhong/anaconda3/bin/python train.py --subgraph_size 10 --batch_size 200
import argparse, os
import math

import matplotlib.pyplot as plt
import nni.retiarii.nn.pytorch
import torch
import random
import torch.nn as nn
import numpy as np
from torch_geometric.datasets import Planetoid,WikiCS, Coauthor
import torch_geometric.transforms as T
from torch_geometric.utils import dropout_adj
from utils_mp import Subgraph, preprocess
from subgcon import SugbCon
from model import Encoder, Scorer, Pool
from time import perf_counter as t
from eval import label_classification
from sklearn.manifold import TSNE



def get_parser():
    parser = argparse.ArgumentParser(description='Description: Script to run our model.')
    parser.add_argument('--dataset', help='Cora, Citeseer or Pubmed. Default=Cora', default='Cora')
    parser.add_argument('--batch_size', type=int, help='batch size', default=300)
    parser.add_argument('--subgraph_size', type=int, help='subgraph size', default=20)
    parser.add_argument('--n_order', type=int, help='order of neighbor nodes', default=10)
    parser.add_argument('--hidden_size', type=int, help='hidden size', default=1024)
    return parser



if __name__ == '__main__':
    parser = get_parser()
    try:
        args = parser.parse_args()
    except:
        exit()
    print(args)
    # Loading data
    data = Planetoid(root='./dataset/' + args.dataset, name=args.dataset)
    #data = WikiCS(root='./dataset/' + args.dataset)
    #data = Coauthor(root='./dataset/' + args.dataset, name=args.dataset)
    num_classes = data.num_classes
    data = data[0]
    num_node = data.x.size(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Setting up the subgraph extractor
    ppr_path = './subgraph/' + args.dataset
    subgraph = Subgraph(data.x, data.edge_index, ppr_path, args.subgraph_size, args.n_order)
    subgraph.build()
    # Setting up the model and optimizer
    model = SugbCon(
        hidden_channels=args.hidden_size, encoder=Encoder(data.num_features, args.hidden_size),
        pool=Pool(in_channels=args.hidden_size),
        scorer=Scorer(args.hidden_size)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


    def drop_feature(x, drop_prob):
        drop_mask = torch.empty(
            (x.size(1),),
            dtype=torch.float32,
            device=x.device).uniform_(0, 1) < drop_prob
        x = x.clone()
        x[:, drop_mask] = 0

        return x

    def train(epoch):
        # Model training
        model.train()
        optimizer.zero_grad()
        sample_idx = random.sample(range(data.x.size(0)), args.batch_size)
        batch, index = subgraph.search(sample_idx)
        '''edge_index_1 = dropout_adj(batch.edge_index, p=0.1)[0]
        edge_index_2 = dropout_adj(batch.edge_index, p=0.2)[0]
        x_1 = drop_feature(batch.x, 0.2)
        x_2 = drop_feature(batch.x, 0.2)
        z1 = model(x_1.cuda(), edge_index_1.cuda())
        z2 = model(x_2.cuda(), edge_index_2.cuda())'''
        edge_index_1 = batch.edge_index
        x_1 = batch.x

        edge_index_2 = dropout_adj(batch.edge_index, p=1.0)[0]
        x_2 = drop_feature(batch.x, 1.0)
        z1 = model(x_1.cuda(), edge_index_1.cuda())
        z2 = model(x_2.cuda(), edge_index_2.cuda())
        '''z, summary = model(batch.x.cuda(), batch.edge_index.cuda(), batch.batch.cuda(), index.cuda())
        loss = model.loss(z, summary)
        loss.backward()'''
        loss = model.loss1(z1, z2, batch_size=0)
        #loss = model.margin_loss(z1,z2)
        #loss = model.loss2(z1,z2)
        #test(model, data.x.cuda(), data.edge_index.cuda(), data.y, final=True)
        loss.backward()
        optimizer.step()
        return loss.item()


    def test(model, x, edge_index, y, final=False):
        model.eval()
        z = model(x, edge_index)
        z = z.cuda()
        y = y.cuda()
        label_classification(z, y, ratio=0.1)
    def visualize(h,color):
        z = TSNE(n_components=2).fit_transform(h.detach().cpu().numpy())
        plt.figure(figsize=(8,8))
        plt.xticks([])
        plt.yticks([])
        plt.scatter(z[:,0],z[:,1],s=5,c=color,cmap="rainbow")
        plt.show()


    start = t()
    prev = start
    for epoch in range(200):
        loss = train(epoch)
        now = t()
        print(f'(T) | Epoch={epoch:03d}, Positive_loss={loss:.4f},'
              f'this epoch {now - prev:.4f}, total {now - start:.4f}')
    model.eval()
    h = model(data.x.cuda(),data.edge_index.cuda())
    visualize(h,color=data.y)
    '''for i in range(50):
        test(model, data.x.cuda(), data.edge_index.cuda(), data.y, final=True)'''