import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional

def subs_weighted_BCE(y_pred: Tensor, 
                      y_true: Tensor, 
                      weights: Optional[Tensor] = None, 
                      reduction: str = 'mean') -> Tensor:
    '''
    Computes the weighted BCE Loss with a different 0/1 class weight per substructure. 

    y_pred: (batch_size, num_substructures), the predicted probabilities for each substructure
    y_true: (batch_size, num_substructures), the true substructures present in each molecule
    weights: (batch_size, num_substructures, 2), the weights for each substructure. Each row i represents the 
        0 and 1 weight for substructure i. If weights are not given, then the loss is unweighted.
    reduction: The method for reducing the loss. Defaults to 'mean', but can be 'mean' or 'sum'.
    '''
    criterion = nn.BCELoss(reduction = 'none')
    unweighted_loss = criterion(y_pred, y_true)
    #Compute the weight for each substructure now
    if weights is not None:
        w = y_true * weights[:,:, 1] + (1 - y_true) * weights[:,:, 0]
        weighted_loss = w * unweighted_loss
    else:
        weighted_loss = 1 * unweighted_loss
    #Averaged across substructures
    rowwise_meaned = torch.mean(weighted_loss, dim = 0)
    assert(rowwise_meaned.shape == (957,))
    #Sum for total loss
    tot_loss = torch.sum(rowwise_meaned)
    return tot_loss

CrossEntropyLoss = nn.CrossEntropyLoss
BCELoss = nn.BCELoss