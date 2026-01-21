import lightning as L
import torch
import torch.optim as optim
import torch.nn.functional as F

from torch_geometric.nn.pool import global_mean_pool
from torch_geometric.nn import GCNConv, NNConv
from torch_geometric.data import Data

class MPNNBase(L.LightningModule):
    def __init__(self, 
                 node_feat_in: int = 9, 
                 node_feat_hidden: int = 64, 
                 edge_feat_in: int = 3,  
                 edge_feat_hidden: int = 32, 
                 n_fc_layers: int = 1, 
                 fc_hidden: int = 128, 
                 message_passing_steps: int = 3, 
                 dropout: float = 0.2, 
                 lr: float = 1e-4,
                 pooling_layer = global_mean_pool,
                 ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.node_feat_in = node_feat_in
        self.node_feat_hidden = node_feat_hidden
        self.edge_feat_in = edge_feat_in
        self.edge_feat_hidden = edge_feat_hidden
        self.n_fc_layers = n_fc_layers
        self.fc_hidden = fc_hidden
        self.dropout_rate = dropout

        # project node features to hidden dimension
        self.projection_node_features = torch.nn.Sequential(torch.nn.Linear(self.node_feat_in, self.node_feat_hidden),
                                                            torch.nn.ReLU())

        # network for message passing
        mp_network = torch.nn.Sequential(torch.nn.Linear(self.edge_feat_in, self.edge_feat_hidden),
                                         torch.nn.ReLU(),
                                         torch.nn.Linear(self.edge_feat_hidden, self.node_feat_hidden * self.node_feat_hidden))
        self.message_passing_steps = message_passing_steps
        self.mpnn_layer = NNConv(in_channels=self.node_feat_hidden,
                                 out_channels=self.node_feat_hidden,
                                 nn=mp_network,
                                 aggr='add')
        
        self.gru = torch.nn.GRU(self.node_feat_hidden, self.node_feat_hidden)
        self.pooling_layer = pooling_layer

        # fully connected layers
        self.fc_layers = torch.nn.ModuleList()
        self.fc_layers.append(torch.nn.Linear(self.node_feat_hidden, self.fc_hidden))
        for _ in range(n_fc_layers-1):
            self.fc_layers.append(torch.nn.Linear(self.fc_hidden, self.fc_hidden))

        # final layer
        self.dropout = torch.nn.Dropout(p=self.dropout_rate)
        self.out_layer = torch.nn.Linear(self.fc_hidden, 1)
        
        # for optimizer
        self.lr = lr

    def _to_latent(self, graph):
        """
        Helper function which sends the graph to latent space thourgh all message passing steps,
        the pooling layer, and the final linear layers
        """

        # node features projected to hidden dimension
        node_features = self.projection_node_features(graph.x.float())
        hidden_features = node_features.unsqueeze(0)

        # now the message passing
        for _ in range(self.message_passing_steps):
            node_features = F.relu(self.mpnn_layer(node_features, graph.edge_index, graph.edge_attr.float()))
            node_features, hidden_features = self.gru(node_features.unsqueeze(0), hidden_features)
            node_features = node_features.squeeze(0)

        # pooling
        out = self.pooling_layer(node_features, graph.batch)

        # fully connected layers
        for layer in self.fc_layers:
            out = F.relu(layer(out))
            out = self.dropout(out)
        return out
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
    def get_empty_graph(self, n_nodes=1):
        """
        Returns an "empty" graph which is compatible with the current architecture

        An "empty" graph in this case is a graph with n_nodes nodes and no edges
        """
        eg = Data(x=torch.zeros(size=(n_nodes, self.node_feat_in), dtype=torch.float),
                 edge_index=torch.zeros(size=(2, 0), dtype=torch.long),
                 edge_attr=torch.zeros(size=(0, self.edge_feat_in), dtype=torch.float),
                 )
        return eg
        
class MPNNRegressor(MPNNBase):
    def __init__(self, 
                 node_feat_in = 9, 
                 node_feat_hidden = 64, 
                 edge_feat_in = 3, 
                 edge_feat_hidden = 32, 
                 n_fc_layers = 1, 
                 fc_hidden = 128, 
                 message_passing_steps = 3, 
                 dropout = 0.2, 
                 lr = 0.0001, 
                 pooling_layer=global_mean_pool):
        super().__init__(node_feat_in=node_feat_in, 
                         node_feat_hidden=node_feat_hidden, 
                         edge_feat_in=edge_feat_in, 
                         edge_feat_hidden=edge_feat_hidden, 
                         n_fc_layers=n_fc_layers, 
                         fc_hidden=fc_hidden, 
                         message_passing_steps=message_passing_steps, 
                         dropout=dropout, 
                         lr=lr, 
                         pooling_layer=pooling_layer)
        self.loss_fn = torch.nn.MSELoss()
        self.name = 'MPNNRegressor'

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
    
    def forward(self, x, edge_index, edge_attr):
        """
        Helper function which sends the graph to latent space thourgh all message passing steps,
        the pooling layer, and the final linear layers
        """

        # node features projected to hidden dimension
        node_features = self.projection_node_features(x.float())
        hidden_features = node_features.unsqueeze(0)

        # now the message passing
        for _ in range(self.message_passing_steps):
            node_features = F.relu(self.mpnn_layer(node_features, edge_index, edge_attr.float()))
            node_features, hidden_features = self.gru(node_features.unsqueeze(0), hidden_features)
            node_features = node_features.squeeze(0)

        # pooling
        out = self.pooling_layer(node_features, batch=None)

        # fully connected layers
        for layer in self.fc_layers:
            out = F.relu(layer(out))
            out = self.dropout(out)
        return out
    
class MPNNClassifier(MPNNBase):
    def __init__(self, 
                 node_feat_in = 9, 
                 node_feat_hidden = 64, 
                 edge_feat_in = 3, 
                 edge_feat_hidden = 32, 
                 n_fc_layers = 1, 
                 fc_hidden = 128, 
                 message_passing_steps = 3, 
                 dropout = 0.2, 
                 lr = 0.0001, 
                 pooling_layer=global_mean_pool):
        super().__init__(node_feat_in=node_feat_in, 
                         node_feat_hidden=node_feat_hidden, 
                         edge_feat_in=edge_feat_in, 
                         edge_feat_hidden=edge_feat_hidden, 
                         n_fc_layers=n_fc_layers, 
                         fc_hidden=fc_hidden, 
                         message_passing_steps=message_passing_steps, 
                         dropout=dropout, 
                         lr=lr, 
                         pooling_layer=pooling_layer)
        self.loss_fn = torch.nn.BCEWithLogitsLoss()
        self.name = 'MPNNClassifier'

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
    
    def forward(self, x, edge_index):
        # trough the conv layers
        l = F.relu(self.conv_layers[0](x.float(), edge_index))
        for layer in self.conv_layers[1:]:
            l = F.relu(layer(l, edge_index))
        # pooling
        l = self.pooling_layer(l, batch=None)
        # fc layers
        for layer in self.fc_layers:
            l = F.relu(layer(l))
            l = self.dropout(l)
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
    
    def forward(self, x, edge_index):
        # trough the conv layers
        l = F.relu(self.conv_layers[0](x.float(), edge_index))
        for layer in self.conv_layers[1:]:
            l = F.relu(layer(l, edge_index))
        # pooling
        l = self.pooling_layer(l, batch=None)
        # fc layers
        for layer in self.fc_layers:
            l = F.relu(layer(l))
            l = self.dropout(l)
        return l
