import scipy.special as special
import numpy as np
import math
import torch

def torch_log_factorial(n, base=2.0):
    n = n.float()
    return torch.lgamma(n+1)/math.log(base)


def torch_log_binom(n, k, base=2.0):
    n = n.float()
    k = k.float()
    mask = n.detach() >= k.detach()
    n = mask * n
    k = mask * k
    a = (torch.lgamma(n + 1) - torch.lgamma((n - k) + 1) - torch.lgamma(k + 1))/math.log(base)
    return a * mask

def torch_log_arrangement(n, k, base=2.0):
    n = n.float()
    k = k.float()
    mask = n.detach() >= k.detach()
    n = mask * n
    k = mask * k
    a = (torch.lgamma(n + 1) - torch.lgamma((n - k) + 1))/math.log(base)
    return a * mask


def torch_factorial(n):
    n = n.double()
    return torch.exp(torch.lgamma(n+1))


def torch_binom(n, k):
    # double precision to avoid rounding errors
    n = n.double()
    k = k.double()
    mask = n.detach() >= k.detach()
    n = mask * n
    k = mask * k
    a = torch.lgamma(n + 1) - torch.lgamma((n - k) + 1) - torch.lgamma(k + 1)
    return torch.exp(a) * mask


# modified from: https://stackoverflow.com/questions/14455634/find-the-index-of-a-given-combination-of-natural-numbers-among-those-returned

def combinations_index(comb, n, torch_enabled=True):
    """Return the index of combination (length == k)

    The combination argument should be a sorted sequence (i | i∈{0…n-1})"""
    
    k= len(comb)
    index= 0
    for offset, item in enumerate(comb):
        item += 1
        index+= special.comb(int(n-item), int(k-offset)) if not torch_enabled \
                else torch.round(torch_binom(n-item, torch.tensor(k-offset, device=n.device)))
        
    result = int(np.ceil(special.comb(int(n), int(k))-int(index)-1)) if not torch_enabled\
                else torch.ceil(torch.round(torch_binom(n, torch.tensor(k, device=n.device)))-index-1).long()
        
    return result

def combinations_comb(index, n , k, torch_enabled=True):
    """Select the 'index'th combination of k over n
    Result is a tuple (i | i∈{0…n-1}) of length k

    Note that if index ≥ binomial_coefficient(n,k)
    then the result is almost always invalid"""
    result= []
    for item, n_temp in enumerate(range(int(n), -1, -1)):
        n_temp = torch.tensor(n_temp, device=n.device) if torch_enabled else n_temp
        pivot= special.comb(int(n_temp-1), int(k-1)) if not torch_enabled\
                else torch.round(torch_binom(n_temp - 1, k - 1))
        if index < pivot:
            result.append(item)
            k-= 1
            if k <= 0: break
        else:
            index-= pivot
            
    return result

def array_pos(index, cols):
    
    row = index//cols
    col = index - cols*row
    
    return row, col