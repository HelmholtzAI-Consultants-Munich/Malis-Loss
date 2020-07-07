import numpy as np
import pyximport
pyximport.install()
from . import pairs_cython


def get_pairs(labels, edge_weights, neighborhood=None,
              keep_objs_per_edge=20, stochastic_malis_param=0, count_method=0,
              ignore_background=True):
    """
    This function simply combines the build_tree and compute just the weights of the MALIS loss.
    
    Roughly speaking the malis weights quantify the impact of an edge in
    the predicted affinity graph on the resulting segmentation.
    
    pos_pairs: numpy array of shape (C,) + labels.shape
               matching pairs
    neg_pairs: numpy array of shape (C,) + labels.shape
               nonmatching pairs
    """
    labels = labels.astype(np.uint32)
    if neighborhood is None:
        neighborhood = np.array([[-1, 0, 0],
                                 [0, -1, 0],
                                 [0, 0, -1]], dtype=np.int32)
        
    aff=pairs_cython.seg_to_affgraph(labels,neighborhood)
    labels = pairs_cython.affgraph_to_seg(aff,neighborhood)
    labels = labels.astype(np.uint32)
    edge_tree = pairs_cython.build_tree(labels, edge_weights, neighborhood,
                                        stochastic_malis_param=stochastic_malis_param)
    return pairs_cython.compute_pairs_with_tree(labels, edge_weights, neighborhood, edge_tree,
                                                keep_objs_per_edge=keep_objs_per_edge,
                                                count_method=count_method,
                                                ignore_background=ignore_background)
