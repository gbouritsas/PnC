import torch
import sys
import numpy as np


def encode_features(graphs, degree_encoding=None, id_encoding=None, **kwargs):
    '''
        Encodes categorical variables such as structural identifiers and degree features.
    '''
    encoding_kwargs = {}
    if hasattr(graphs[0], 'identifiers'):
        encoder_ids, d_id = None, [1]*graphs[0].identifiers.shape[1]
        if id_encoding is not None:
            id_encoding_fn = getattr(sys.modules[__name__], id_encoding)
            ids = [graph.identifiers for graph in graphs]
            if 'ids' not in kwargs:
                encoding_kwargs['ids'] = {}
            else:
                encoding_kwargs['ids'] = kwargs['ids']
            encoder_ids = id_encoding_fn(ids, **(encoding_kwargs['ids']))
            encoded_ids = encoder_ids.fit(ids)
            d_id = encoder_ids.d
    else:
#         encoder_ids, d_id = None, None
        d_id = None

    if hasattr(graphs[0], 'degrees'):
        encoder_degrees, d_degree = None, [1]
        if degree_encoding is not None: 
            degree_encoding_fn = getattr(sys.modules[__name__], degree_encoding)
            degrees = [graph.degrees.unsqueeze(1) for graph in graphs]
            if 'degrees' not in kwargs:
                encoding_kwargs['degrees'] = {}
            else:
                encoding_kwargs['degrees'] = kwargs['degrees']
            encoder_degrees = degree_encoding_fn(degrees, **(encoding_kwargs['degrees']))
            encoded_degrees = encoder_degrees.fit(degrees)
            d_degree = encoder_degrees.d
    else:
#         encoder_degrees, d_degree = None, None
        d_degree = None
        
    for g, graph in enumerate(graphs):
        if id_encoding is not None and hasattr(graph, 'identifiers'):
            setattr(graph, 'identifiers', encoded_ids[g])
        if degree_encoding is not None and hasattr(graph, 'degrees'):
            setattr(graph, 'degrees', encoded_degrees[g])
                            
    return graphs, d_degree, d_id


class one_hot_unique:
    
    def __init__(self, tensor_list, **kwargs):
        tensor_list = torch.cat(tensor_list, 0)
        self.d = list()
        self.corrs = dict()
        for col in range(tensor_list.shape[1]):
            uniques, corrs = np.unique(tensor_list[:, col], return_inverse=True, axis=0)
            self.d.append(len(uniques))
            self.corrs[col] = corrs
        return       
            
    def fit(self, tensor_list):
        pointer = 0
        encoded_tensors = list()
        for tensor in tensor_list:
            n = tensor.shape[0]
            for col in range(tensor.shape[1]):
                translated = torch.LongTensor(self.corrs[col][pointer:pointer+n]).unsqueeze(1)
                encoded = torch.cat((encoded, translated), 1) if col > 0 else translated
            encoded_tensors.append(encoded)
            pointer += n
        return encoded_tensors
        

class one_hot_max:
    
    def __init__(self, tensor_list, **kwargs):
        tensor_list = torch.cat(tensor_list,0)
        self.d = [int(tensor_list[:,i].max()+1) for i in range(tensor_list.shape[1])]
    
    def fit(self, tensor_list):
        return tensor_list