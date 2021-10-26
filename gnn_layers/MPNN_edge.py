import torch
from utils.nn_layers import mlp, central_encoder
from torch_geometric.nn.conv import MessagePassing

class MPNN_edge(MessagePassing):
    
    def __init__(self,
                 d_in,
                 d_ef,
                 d_degree,
                 degree_as_tag,
                 retain_features,
                 d_msg,
                 d_up,
                 d_h,
                 activation_name,
                 bn,
                 aggr='add',
                 aggr_fn='general',
                 eps=0,
                 train_eps=False,
                 flow='source_to_target',
                 **kwargs):
        super(MPNN_edge, self).__init__(aggr=aggr,flow=flow)
        d_msg = d_in if d_msg is None else d_msg
        self.aggr = aggr
        self.aggr_fn = aggr_fn
        self.degree_as_tag = degree_as_tag
        self.retain_features = retain_features
        if degree_as_tag:
            d_in = d_in + d_degree if retain_features else d_degree
        # INPUT_DIMS
        if aggr_fn == 'gin':
            # dummy variable for self loop edge features
            self.central_node_edge_encoder = central_encoder(kwargs['edge_embedding'], d_ef, extend=kwargs['extend_dims'])
            d_ef = self.central_node_edge_encoder.d_out
            self.initial_eps = eps
            if train_eps:
                self.eps = torch.nn.Parameter(torch.Tensor([eps]))
            else:
                self.register_buffer('eps', torch.Tensor([eps]))
            self.eps.data.fill_(self.initial_eps)
            self.msg_fn = None
            update_input_dim = d_in + d_ef
        elif aggr_fn == 'general':
            msg_input_dim = 2 * d_in + d_ef
            self.msg_fn = mlp(
                            msg_input_dim,
                            d_msg,
                            d_h,
                            activation_name,
                            bn)
            update_input_dim = d_in + d_msg
        else:
            raise NotImplementedError("Aggregation function {} is not currently supported.".format(aggr_fn))
        self.update_fn = mlp(
                            update_input_dim,
                            d_up,
                            d_h,
                            activation_name,
                            bn)
        return

    def forward(self, x, edge_index, **kwargs):
        # prepare input features
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        degrees = kwargs['degrees']
        degrees = degrees.unsqueeze(-1) if degrees.dim() == 1 else degrees
        if self.degree_as_tag:
            x = torch.cat([x, degrees], -1) if self.retain_features else degrees
        edge_features = kwargs['edge_features']
        edge_features = edge_features.unsqueeze(-1) if edge_features.dim() == 1 else edge_features
        n_nodes = x.shape[0]
        if self.aggr_fn == 'gin':
            edge_features_ii, edge_features = self.central_node_edge_encoder(edge_features, n_nodes)
            self_msg = torch.cat((x, edge_features_ii), -1)
            out = self.update_fn((1 + self.eps) * self_msg +
                                 self.propagate(edge_index=edge_index, x=x, edge_features=edge_features))
        elif self.aggr_fn == 'general':
                out = self.update_fn(torch.cat((x, self.propagate(edge_index=edge_index, x=x,  edge_features=edge_features)), -1))
        return out
    def message(self, x_i, x_j, edge_features):
        if self.aggr_fn == 'gin':
            msg_j = torch.cat((x_j, edge_features), -1)
        elif self.aggr_fn == 'general':
            msg_j = torch.cat((x_i, x_j, edge_features), -1)
            msg_j = self.msg_fn(msg_j)
        else:
            raise NotImplementedError("Aggregation function {} is not currently supported.".format(self.aggr_fn))
        return msg_j
    def __repr__(self):
        return '{}(msg_fn = {}, update_fn = {})'.format(self.__class__.__name__, self.msg_fn, self.update_fn)


