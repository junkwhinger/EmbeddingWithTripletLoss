"""
Reference
 - https://github.com/adambielski/siamese-triplet/blob/master/utils.py
 - https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065/2
"""

import logging
logging.basicConfig(level=logging.DEBUG)


import torch

def pdist(x):
    """
    compute distance matrix
    """
    x_norm = x.pow(2).sum(1).view(-1, 1)
    x_t = torch.t(x)
    x_t_norm = x_norm.view(1, -1)

    distance_matrix =  - 2.0 * torch.mm(x, x_t) + x_norm + x_t_norm
    return distance_matrix


class TripletSelector:
    """
    return indices of anchors, positive and negative samples
    """

    def __init__(self):
        pass

    def get_triplets(self, embeddings, labels):
        raise NotImplementedError


class NegativeTripletSelector(TripletSelector):
    """
    returns indices of anchors, positives and negative samples
    using margin and func_list to select negative samples
    """

    def __init__(self, margin, func_list):
        super(NegativeTripletSelector, self).__init__()
        self.margin = margin

        if not func_list:
            raise ValueError("at least one function is required")
        self.func_list = func_list

    def get_triplets(self, embeddings, labels):

        # calculate distances between embedded vectors
        distance_matrix = pdist(embeddings)
        # logging.debug(distance_matrix)

        # triplets..
        triplets = []

        # get unique labels in this batch
        unique_labels = labels.unique()
        for anchor_label in unique_labels:

            # anchor indices
            anchor_mask = labels == anchor_label
            anchor_indices = (anchor_mask == 1).nonzero()[:, 0]

            # skip the following procedure if we have one or zero positive case
            if anchor_indices.size(0) < 2:
                continue

            # negative indices
            negative_indices = (anchor_mask == 0).nonzero()[:, 0]

            # generate anchor-positive pairs using combination
            anchor_positive_pairs = torch.combinations(anchor_indices)

            # for each pair
            for ap_pair in anchor_positive_pairs:
                anchor_idx = ap_pair[0]
                positive_idx = ap_pair[1]

                # distance between anchor-positive
                ap_distance = distance_matrix[anchor_idx, positive_idx]

                # distanceS between anchor-negativeS
                an_distances = distance_matrix[anchor_idx][negative_indices]

                # loss_valueS = ap - anS + margin
                loss_values = ap_distance - an_distances + self.margin

                # for each fn in func_list (ordered by high priority)
                negative_idx = None
                for fn in self.func_list:
                    negative_idx = fn(loss_values, self.margin)
                    if negative_idx is not None:
                        break

                # if no negative_idx is found, just pick a random one
                if negative_idx is None:
                    logging.debug("No negative idx found. Picking random..")
                    choice = torch.randint(0, negative_indices.size(0), (1,))
                    negative_idx = negative_indices[choice].view(-1).squeeze(0)

                triplet = torch.stack([anchor_idx, positive_idx, negative_idx], 0)
                triplets.append(triplet)

        triplet_tensor = torch.stack(triplets)
        logging.debug(triplet_tensor.size())
        return triplet_tensor



def hard_fn(loss_values, margin):
    # margin > margin
    mask = loss_values > margin

    # if no match, return None
    if mask.sum() == 0:
        return None

    # else, randomly pick one index and return it
    else:
        indices = (mask == 1).nonzero().view(-1)
        choice = torch.randint(0, indices.size(0), (1,))
        chosen = indices[choice].view(-1).squeeze(0)

        logging.debug("found hard")
        return chosen


def semihard_fn(loss_values, margin):
    # margin < loss < 0
    mask = (loss_values < margin) & (loss_values > 0)

    # if no match, return None
    if mask.sum() == 0:
        return None

    # else, randomly pick one index and return it
    else:
        indices = (mask == 1).nonzero().view(-1)
        choice = torch.randint(0, indices.size(0), (1,))
        chosen = indices[choice].view(-1).squeeze(0)

        logging.debug("found semi-hard")
        return chosen


def easy_fn(loss_values, margin):
    # loss < 0
    mask = loss_values < 0

    # if no match, return None
    if mask.sum() == 0:
        return None

    # else, randomly pick one index and return it
    else:
        indices = (mask == 1).nonzero().view(-1)
        choice = torch.randint(0, indices.size(0), (1,))

        chosen = indices[choice].view(-1).squeeze(0)

        logging.debug("found easy")
        return chosen

#
# # test sample
# embeddings = torch.randn(16, 2)
# embeddings[0, :] = torch.FloatTensor([0.0, 0.0])
# embeddings[1, :] = torch.FloatTensor([1.0, 1.0])
# embeddings[2, :] = torch.FloatTensor([0.5, 0.5])
#
# # logging.debug(embeddings)
# labels = torch.LongTensor([0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3])
#
# # triplet miner test
# margin = 0.2
# func_list = [semihard_fn, easy_fn, hard_fn]
# miner = NegativeTripletSelector(margin, func_list)
# logging.debug(miner.get_triplets(embeddings, labels))
