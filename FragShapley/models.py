import lightning as L
import torch
import torch.optim as optim
import torch.nn.functional as F

from torch_geometric.nn.pool import global_mean_pool
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

class GCNBase(L.LightningModule):
    """
    Base Class for GCN

    Will be used later to build the Regressor and Classifier variant
    """
    def __init__(self, 
                 node_feat_in: int = 9, 
                 n_conv_layers: int = 3, 
                 n_fc_layers: int = 1, 
                 node_feat_hidden: int = 64, 
                 fc_hidden: int =128,
                 dropout: float = 0.2, 
                 lr: float = 1e-4, 
                 pooling_layer = global_mean_pool,
                 ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.node_feat_in = node_feat_in
        self.n_conv_layers = n_conv_layers
        self.n_fc_layers = n_fc_layers
        self.node_feat_hidden = node_feat_hidden
        self.fc_hidden = fc_hidden

        # set up convolutional layers
        self.conv_layers = torch.nn.ModuleList()
        self.conv_layers.append(GCNConv(node_feat_in, node_feat_hidden)) # initial layer
        for _ in range(n_conv_layers-1): # remaining layers
            self.conv_layers.append(GCNConv(node_feat_hidden, node_feat_hidden))
        # pooling operation
        self.pooling_layer = pooling_layer
        # fully connected layers
        self.fc_layers = torch.nn.ModuleList()
        self.fc_layers.append(torch.nn.Linear(node_feat_hidden, fc_hidden))
        for _ in range(n_fc_layers-1):
            self.fc_layers.append(torch.nn.Linear(fc_hidden, fc_hidden))
        # final layer
        self.out_layer = torch.nn.Linear(fc_hidden, 1)
        self.dropout = torch.nn.Dropout(p=dropout)
        # for optimizer
        self.lr = lr
    
    def _to_latent(self, graph):
        """
        Helper function which sends the graph to latent space thourgh all convolutional
        layers, the pooling layer, and the final linear layers
        """
        # trough the conv layers
        l = F.relu(self.conv_layers[0](graph.x.float(), graph.edge_index))
        for layer in self.conv_layers[1:]:
            l = F.relu(layer(l, graph.edge_index))
        # pooling
        l = self.pooling_layer(l, graph.batch)
        # fc layers
        for layer in self.fc_layers:
            l = F.relu(layer(l))
            l = self.dropout(l)
        return l
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def get_empty_graph(self, n_nodes=1):
        """
        Returns an "empty" graph which is compatible with the current architecture

        An "empty" graph in this case is a graph with n_nodes nodes and no edges
        """
        eg = Data(x=torch.zeros(size=(n_nodes, self.node_feat_in), dtype=torch.float),
                edge_index=torch.zeros(size=(2, 0), dtype=torch.long))
        return eg

class GCNRegressor(GCNBase):
    """
    Implements a GCN Regressor based on GCNBase
    """
    def __init__(self, 
                 node_feat_in: int = 9, 
                 n_conv_layers: int = 3, 
                 n_fc_layers: int = 1, 
                 node_feat_hidden: int = 64, 
                 fc_hidden: int =128,
                 dropout: float = 0.2, 
                 lr: float = 1e-4, 
                 pooling_layer = global_mean_pool,
                 ) -> None:
        super().__init__(node_feat_in=node_feat_in,
                         n_conv_layers=n_conv_layers,
                         n_fc_layers=n_fc_layers,
                         node_feat_hidden=node_feat_hidden,
                         fc_hidden=fc_hidden,
                         dropout=dropout,
                         lr=lr,
                         pooling_layer=pooling_layer)
        self.loss_fn = torch.nn.MSELoss()
        self.name = 'GCNRegressor'
    
    def training_step(self, batch, batch_idx):
        """
        Implements the training step for lightning
        """
        graph, y = batch
        l = self._to_latent(graph) # latent representation
        l = self.out_layer(l) # to final outcome
        loss = self.loss_fn(torch.squeeze(l, 1), y.float())
        self.log('train_loss', loss, batch_size=len(batch))
        return loss
    
    def predict_step(self, batch, batch_idx):
        """
        Implements the predict step for lightning

        similar to training_step
        """
        graph, _ = batch # we do not care about the label here
        l = self._to_latent(graph) # latent representation
        l = self.out_layer(l) # to final outcome
        return l

class GCNClassifier(GCNBase):
    """
    Implements a GCN Classifier based on GCNBase
    Note: Only works for binary classification
    """
    def __init__(self, 
                 node_feat_in: int = 9, 
                 n_conv_layers: int = 3, 
                 n_fc_layers: int = 1, 
                 node_feat_hidden: int = 64, 
                 fc_hidden: int =128,
                 dropout: float = 0.2, 
                 lr: float = 1e-4, 
                 pooling_layer = global_mean_pool,
                 ) -> None:
        super().__init__(node_feat_in=node_feat_in,
                         n_conv_layers=n_conv_layers,
                         n_fc_layers=n_fc_layers,
                         node_feat_hidden=node_feat_hidden,
                         fc_hidden=fc_hidden,
                         dropout=dropout,
                         lr=lr,
                         pooling_layer=pooling_layer)
        self.loss_fn = torch.nn.BCEWithLogitsLoss()
        self.name = 'GCNClassifier'
    
    def training_step(self, batch, batch_idx):
        """
        Implements the training step for lightning
        """
        graph, y = batch
        l = self._to_latent(graph) # to latent
        l = self.out_layer(l) # here we have logits
        loss = self.loss_fn(torch.squeeze(l, 1), y.float()) # use logits for the loss
        self.log('train_loss', loss, batch_size=len(batch))
        return loss
    
    def predict_step(self, batch, batch_idx):
        graph, _ = batch
        l = self._to_latent(graph) # to latent
        l = self.out_layer(l) # here we have logits
        return torch.nn.functional.sigmoid(l) # convert to probabilities using Sigmoid
