import os
import torch
from typing import List


class DataParser():
    def __init__(self):
        super().__init__()
        self.dataset_name = 'bbbp'
        self.dataset_dir = './datasets'
        self.task = None
        self.random_split: bool = True
        self.data_split_ratio: List = [0.8, 0.1, 0.1]   # the ratio of training, validation and testing set for random split
        self.seed = 1


class GATParser():# hyper-parameter for gat model
    def __init__(self):
        super().__init__()
        self.gat_dropout = 0.6    # dropout in gat layer
        self.gat_heads = 10         # multi-head
        self.gat_hidden = 10        # the hidden units for each head
        self.gat_concate = True    # the concatenation of the multi-head feature
        self.num_gat_layer = 3


class ModelParser():
    def __init__(self):
        super().__init__()
        self.device: int = 0
        self.model_name: str = 'gcn'
        self.checkpoint: str = './checkpoint'
        self.concate: bool = False                     # whether to concate the gnn features before mlp
        self.latent_dim: List[int] = [128, 128, 128]   # the hidden units for each gnn layer
        self.readout: 'str' = 'max'                    # the graph pooling method
        self.mlp_hidden: List[int] = []                # the hidden units for mlp classifier
        self.gnn_dropout: float = 0.0                  # the dropout after gnn layers
        self.dropout: float = 0.5                      # the dropout after mlp layers
        self.adj_normlize: bool = True                 # the edge_weight normalization for gcn conv
        self.emb_normlize: bool = False                # the l2 normalization after gnn layer
        self.enable_prot = True                        # whether to enable prototype training
        self.num_prototypes_per_class = 5              # the num_prototypes_per_class
        self.gat_dropout = 0.6  # dropout in gat layer
        self.gat_heads = 10  # multi-head
        self.gat_hidden = 10  # the hidden units for each head
        self.gat_concate = True  # the concatenation of the multi-head feature
        self.num_gat_layer = 3


    def process_args(self) -> None:
        # self.device = torch.device('cpu')
        if torch.cuda.is_available():
            self.device = torch.device('cuda', self.device_id)
        else:
            pass


class MCTSParser(DataParser, ModelParser):
    rollout: int = 10                         # the rollout number
    high2low: bool = False                    # expand children with different node degree ranking method
    c_puct: float = 5                         # the exploration hyper-parameter
    min_atoms: int = 5                        # for the synthetic dataset, change the minimal atoms to 5.
    max_atoms: int = 10
    expand_atoms: int = 10                     # # of atoms to expand children

    def process_args(self) -> None:
        self.explain_model_path = os.path.join(self.checkpoint,
                                               self.dataset_name,
                                               f"{self.model_name}_best.pth")


class RewardParser():
    def __init__(self):
        super().__init__()
        self.reward_method: str = 'mc_l_shapley'                         # Liberal, gnn_score, mc_shapley, l_shapley， mc_l_shapley
        self.local_raduis: int = 4                                       # (n-1) hops neighbors for l_shapley
        self.subgraph_building_method: str = 'zero_filling'
        self.sample_num: int = 100                                       # sample time for monte carlo approximation


class TrainParser():
    def __init__(self):
        super().__init__()
        self.learning_rate = 0.005
        self.batch_size = 24
        self.weight_decay = 0.0
        self.max_epochs = 800
        self.save_epoch = 10
        self.early_stopping = 80
        self.last_layer_optimizer_lr = 1e-4            # the learning rate of the last layer
        self.joint_optimizer_lrs = {'features': 1e-4,
                       'add_on_layers': 3e-3,
                       'prototype_vectors': 3e-3}      # the learning rates of the joint training optimizer
        self.warm_epochs = 10                          # the number of warm epochs
        self.proj_epochs = 100                         # the epoch to start mcts
        self.sampling_epochs = 100                     # the epoch to start sampling edges
        self.nearest_graphs = 10                       # number of graphs in projection


data_args = DataParser()
model_args = ModelParser()
mcts_args = MCTSParser()
reward_args = RewardParser()
train_args = TrainParser()

import torch
import random
import numpy as np
random_seed = 1234
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
