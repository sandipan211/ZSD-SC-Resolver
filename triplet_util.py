
from itertools import combinations
from itertools import permutations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


import numpy as np
import pandas as pd

import sklearn
import sklearn.covariance
import scipy
import torch.optim as optim
from torch.autograd import Variable

def pairwise_mahalanobis(S1, S2, Cov_1=None):
    """
        S1: C1 x K matrix (torch.FloatTensor)
          -> C1 K-dimensional semantic prototypes
        S2: C2 x K matrix (torch.FloatTensor)
          -> C2 K-dimensional semantic prototypes
        Sigma_1: K x K matrix (torch.FloatTensor)
          -> inverse of the covariance matrix Sigma; used to compute Mahalanobis distances
          by default Sigma is the identity matrix (and so distances are euclidean distances)
        
        returns an C1 x C2 matrix corresponding to the Mahalanobis distance between each element of S1 and S2
        (Equation 5)
    """
    if S1.dim() != 2 or S2.dim() != 2 or S1.shape[1] != S2.shape[1]:
        raise RuntimeError("Bad input dimension")
    C1, K = S1.shape
    C2, K = S2.shape
    if Cov_1 is None:
        Cov_1 = torch.eye(K)
    if Cov_1.shape != (K, K):
        raise RuntimeError("Bad input dimension")
    
    S1S2t = S1.matmul(Cov_1).matmul(S2.t())
    S1S1 = S1.matmul(Cov_1).mul(S1).sum(dim=1, keepdim=True).expand(-1, C2)
    S2S2 = S2.matmul(Cov_1).mul(S2).sum(dim=1, keepdim=True).t().expand(C1, -1)
    return torch.sqrt(torch.abs(S1S1 + S2S2 - 2. * S1S2t) + 1e-32)  # to avoid numerical instabilities


def distance_matrix(S, mahalanobis=True, mean=1., std=0.5):
    """
        S: C x K matrix (numpy array)
          -> K-dimensional prototypes of C classes
        mahalanobis: indicates whether to use Mahalanobis distance (uses euclidean distance if False)
        mean & std: target mean and standard deviation
        
        returns a C x C matrix corresponding to the Mahalanobis distance between each pair of elements of S
        rescaled to have approximately target mean and standard deviation while keeping values positive
        (Equation 6)
    """
    Cov_1 = None
    if mahalanobis:
        #Cov = np.cov(S.T)
        #Cov, _ = sklearn.covariance.ledoit_wolf(S) # robust estimation of covariance matrix
        lw = sklearn.covariance.LedoitWolf(assume_centered=False).fit(S)
        #Cov_1 = torch.FloatTensor(np.linalg.inv(Cov))
        Cov_1 = torch.FloatTensor(lw.precision_)
    S = torch.FloatTensor(S)
    
    distances = pairwise_mahalanobis(S, S, Cov_1)
    
    # Rescaling to have approximately target mean and standard deviation while keeping values positive
    max_zero_distance = distances.diag().max()
    positive_distances = np.array([x for x in distances.view(-1) if x > max_zero_distance])
    emp_std = float(positive_distances.std())
    emp_mean = float(positive_distances.mean())
    distances = F.relu(std * (distances - emp_mean) / emp_std + mean)
    max_zero_distance = distances.diag().max()
    positive_distances = np.array([x for x in distances.view(-1) if x > max_zero_distance])
    emp_std = float(positive_distances.std())
    emp_mean = float(positive_distances.mean())
    distances = F.relu(std * (distances - emp_mean) / emp_std + mean)
    return distances


class AllTripletSelector():
    """
    Returns all possible triplets
  
    """

    def __init__(self):
        super(AllTripletSelector, self).__init__()

    def get_triplets(self, embeddings, labels):
        labels = labels.cpu().data.numpy()
        triplets = []
        for label in set(labels):
            label_mask = (labels == label)
            label_indices = np.where(label_mask)[0]
            if len(label_indices) < 2:
                continue
            negative_indices = np.where(np.logical_not(label_mask))[0]
            anchor_positives = list(permutations(label_indices, 2))  # All anchor-positive pairs,so using permutation

            # Add all negatives for all positive pairs
            temp_triplets = [[anchor_positive[0], anchor_positive[1], neg_ind] for anchor_positive in anchor_positives
                             for neg_ind in negative_indices]
            triplets += temp_triplets

        return torch.LongTensor(np.array(triplets))

class TripletLoss():
    """
    Triplets loss
    Takes a batch of embeddings and corresponding labels.
    Triplets are generated using all_triplet_selector object that take embeddings and targets and return indices of
    triplets
    """

    # for var margin
    def __init__(self, D_tilde,labels, triplet_selector):
        super(TripletLoss, self).__init__()
        
        self.D_tilde=D_tilde.cuda()
        self.triplet_selector = triplet_selector
        self.labels=labels

    
    def forward(self, embeddings, target):

        triplets = self.triplet_selector.get_triplets(embeddings, target)

        if embeddings.is_cuda:
            triplets = triplets.cuda()
        
        ap_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 1]]).pow(2).sum(1) 
        
        an_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 2]]).pow(2).sum(1)  
        
        #for variable margin
        margin=self.D_tilde[self.labels[triplets[:,0]],self.labels[triplets[:,2]]]
       
        losses = F.relu(ap_distances - an_distances + margin)
        
        
        return losses.mean()
