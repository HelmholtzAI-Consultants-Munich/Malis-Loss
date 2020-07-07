# distutils: language = c++

import numpy as np
cimport numpy as np
cimport cython
from cython.view cimport array as cvarray
import sys
from libcpp.unordered_map cimport unordered_map
from libcpp.stack cimport stack
from libc.stdlib cimport malloc, free, realloc
from libc.math cimport log, sqrt
from cython.operator cimport dereference
try:  # scipy >= 0.19
    from scipy.special import comb
except ImportError:
    from scipy.misc import comb
import scipy.sparse
from libc.stdint cimport uint64_t

cdef extern from "malis_cpp.h":
    void connected_components_cpp(const uint64_t nVert,
                   const uint64_t nEdge, const uint64_t* node1, const uint64_t* node2, const int* edgeWeight,
                   uint64_t* seg);
    void marker_watershed_cpp(const uint64_t nVert, const uint64_t* marker,
                   const uint64_t nEdge, const uint64_t* node1, const uint64_t* node2, const float* edgeWeight,
                   uint64_t* seg);


@cython.boundscheck(False)
@cython.wraparound(False)
cdef int chase(unsigned int [:] id_table,unsigned int idx) nogil:
    """ Terminates once it finds an index into id_table where the corresponding
        element in id_table is an index to itself """
    if id_table[idx] != idx:
        id_table[idx] = chase(id_table, id_table[idx])
    return id_table[idx]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void my_unravel_index(int flat_idx, int [::1] shape, int[4] return_idxes) nogil:
    """
    unravels the 4-dimensional index of flat_idx which points into a flattened
    4-dimensional array
    """
    cdef int len_shape = 4 # hardcoded, could be changed later
    cdef int dim, i
    cdef int current_stride

    for dim in range(len_shape):
        # compute stride of dimension dim by multiplying the dimensions
        # from the current dim to the final dimension
        current_stride = 1
        for i in range(dim + 1, len_shape):
            current_stride *= shape[i]

        # compute index
        return_idxes[dim] = int(flat_idx / current_stride)
        flat_idx = flat_idx % current_stride

def connected_components(uint64_t nVert,
                         np.ndarray[uint64_t,ndim=1] node1,
                         np.ndarray[uint64_t,ndim=1] node2,
                         np.ndarray[int,ndim=1] edgeWeight,
                         int sizeThreshold=1):
    cdef uint64_t nEdge = node1.shape[0]
    node1 = np.ascontiguousarray(node1)
    node2 = np.ascontiguousarray(node2)
    edgeWeight = np.ascontiguousarray(edgeWeight)
    cdef np.ndarray[uint64_t,ndim=1] seg = np.zeros(nVert,dtype=np.uint64)
    connected_components_cpp(nVert,
                             nEdge, &node1[0], &node2[0], &edgeWeight[0],
                             &seg[0]);
    (seg,segSizes) = prune_and_renum(seg,sizeThreshold)
    return (seg, segSizes)


def marker_watershed(np.ndarray[uint64_t,ndim=1] marker,
                     np.ndarray[uint64_t,ndim=1] node1,
                     np.ndarray[uint64_t,ndim=1] node2,
                     np.ndarray[float,ndim=1] edgeWeight,
                     int sizeThreshold=1):
    cdef uint64_t nVert = marker.shape[0]
    cdef uint64_t nEdge = node1.shape[0]
    marker = np.ascontiguousarray(marker)
    node1 = np.ascontiguousarray(node1)
    node2 = np.ascontiguousarray(node2)
    edgeWeight = np.ascontiguousarray(edgeWeight)
    cdef np.ndarray[uint64_t,ndim=1] seg = np.zeros(nVert,dtype=np.uint64)
    marker_watershed_cpp(nVert, &marker[0],
                         nEdge, &node1[0], &node2[0], &edgeWeight[0],
                         &seg[0]);
    (seg,segSizes) = prune_and_renum(seg,sizeThreshold)
    return (seg, segSizes)


def prune_and_renum(np.ndarray[uint64_t,ndim=1] seg,
                    int sizeThreshold=1):
    # renumber the components in descending order by size
    segId,segSizes = np.unique(seg, return_counts=True)
    descOrder = np.argsort(segSizes)[::-1]
    renum = np.zeros(int(segId.max()+1),dtype=np.uint64)
    segId = segId[descOrder]
    segSizes = segSizes[descOrder]
    renum[segId] = np.arange(1,len(segId)+1)

    if sizeThreshold>0:
        renum[segId[segSizes<=sizeThreshold]] = 0
        segSizes = segSizes[segSizes>sizeThreshold]

    seg = renum[seg]
    return (seg, segSizes)


def bmap_to_affgraph(bmap,nhood,return_min_idx=False):
    """
    Construct an affinity graph from a boundary map
    
    The spatial shape of the affinity graph is the same as of seg_gt.
    This means that some edges are are undefined and therefore treated as disconnected.
    If the offsets in nhood are positive, the edges with largest spatial index are undefined.    
    
    Parameters
    ----------
    
    bmap: 3d np.ndarray, int
        Volume of boundaries
        0: object interior, 1: boundaries / ECS 
    nhood: 2d np.ndarray, int
        Neighbourhood pattern specifying the edges in the affinity graph
        Shape: (#edges, ndim)
        nhood[i] contains the displacement coordinates of edge i
        The number and order of edges is arbitrary
        
    Returns
    -------
    
    aff: 4d np.ndarray int32
        Affinity graph of shape (#edges, x, y, z)
        1: connected, 0: disconnected
        
    """
    shape = bmap.shape
    nEdge = nhood.shape[0]
    aff = np.zeros((nEdge,)+shape,dtype=np.int32)
    minidx = np.zeros((nEdge,)+shape,dtype=np.int32)

    for e in range(nEdge):
        aff[e, \
            max(0,-nhood[e,0]):min(shape[0],shape[0]-nhood[e,0]), \
            max(0,-nhood[e,1]):min(shape[1],shape[1]-nhood[e,1]), \
            max(0,-nhood[e,2]):min(shape[2],shape[2]-nhood[e,2])] = np.minimum( \
                        bmap[max(0,-nhood[e,0]):min(shape[0],shape[0]-nhood[e,0]), \
                            max(0,-nhood[e,1]):min(shape[1],shape[1]-nhood[e,1]), \
                            max(0,-nhood[e,2]):min(shape[2],shape[2]-nhood[e,2])], \
                        bmap[max(0,nhood[e,0]):min(shape[0],shape[0]+nhood[e,0]), \
                            max(0,nhood[e,1]):min(shape[1],shape[1]+nhood[e,1]), \
                            max(0,nhood[e,2]):min(shape[2],shape[2]+nhood[e,2])] )
        minidx[e, \
            max(0,-nhood[e,0]):min(shape[0],shape[0]-nhood[e,0]), \
            max(0,-nhood[e,1]):min(shape[1],shape[1]-nhood[e,1]), \
            max(0,-nhood[e,2]):min(shape[2],shape[2]-nhood[e,2])] = \
                        bmap[max(0,-nhood[e,0]):min(shape[0],shape[0]-nhood[e,0]), \
                            max(0,-nhood[e,1]):min(shape[1],shape[1]-nhood[e,1]), \
                            max(0,-nhood[e,2]):min(shape[2],shape[2]-nhood[e,2])] > \
                        bmap[max(0,nhood[e,0]):min(shape[0],shape[0]+nhood[e,0]), \
                            max(0,nhood[e,1]):min(shape[1],shape[1]+nhood[e,1]), \
                            max(0,nhood[e,2]):min(shape[2],shape[2]+nhood[e,2])]

    return aff

def seg_to_affgraph(seg,nhood):
    """
    Construct an affinity graph from a segmentation (IDs) 
    
    Segments with ID 0 are regarded as disconnected
    The spatial shape of the affinity graph is the same as of seg_gt.
    This means that some edges are are undefined and therefore treated as disconnected.
    If the offsets in nhood are positive, the edges with largest spatial index are undefined.
    
    Input:    
    seg_gt: 3d np.ndarray
        Volume of segmentation IDs
    nhood: 2d np.ndarray
        Neighbourhood pattern specifying the edges in the affinity graph
        Shape: (#edges, ndim)
        nhood[i] contains the displacement coordinates of edge i
        The number and order of edges is arbitrary
        
    Returns:
    
    aff: 4d np.ndarray
        Affinity graph of shape (#edges, x, y, z)
        1: connected, 0: disconnected        
    """
    shape = seg.shape
    nEdge = nhood.shape[0]
    aff = np.zeros((nEdge,)+shape,dtype=np.int32)

    for e in range(nEdge):
        aff[e, \
            max(0,-nhood[e,0]):min(shape[0],shape[0]-nhood[e,0]), \
            max(0,-nhood[e,1]):min(shape[1],shape[1]-nhood[e,1]), \
            max(0,-nhood[e,2]):min(shape[2],shape[2]-nhood[e,2])] = \
                        (seg[max(0,-nhood[e,0]):min(shape[0],shape[0]-nhood[e,0]), \
                            max(0,-nhood[e,1]):min(shape[1],shape[1]-nhood[e,1]), \
                            max(0,-nhood[e,2]):min(shape[2],shape[2]-nhood[e,2])] == \
                         seg[max(0,nhood[e,0]):min(shape[0],shape[0]+nhood[e,0]), \
                            max(0,nhood[e,1]):min(shape[1],shape[1]+nhood[e,1]), \
                            max(0,nhood[e,2]):min(shape[2],shape[2]+nhood[e,2])] ) \
                        * ( seg[max(0,-nhood[e,0]):min(shape[0],shape[0]-nhood[e,0]), \
                            max(0,-nhood[e,1]):min(shape[1],shape[1]-nhood[e,1]), \
                            max(0,-nhood[e,2]):min(shape[2],shape[2]-nhood[e,2])] > 0 ) \
                        * ( seg[max(0,nhood[e,0]):min(shape[0],shape[0]+nhood[e,0]), \
                            max(0,nhood[e,1]):min(shape[1],shape[1]+nhood[e,1]), \
                            max(0,nhood[e,2]):min(shape[2],shape[2]+nhood[e,2])] > 0 )

    return aff

def nodelist_like(shape,nhood):
    """
    Constructs the two node lists to represent edges of
    an affinity graph for a given volume shape and neighbourhood pattern.
    
    Parameters
    ----------
    
    shape: tuple/list
     shape of corresponding volume (z, y, x)
    nhood: 2d np.ndarray, int
        Neighbourhood pattern specifying the edges in the affinity graph
        Shape: (#edges, ndim)
        nhood[i] contains the displacement coordinates of edge i
        The number and order of edges is arbitrary
        
    Returns
    -------
    
    node1: 4d np.ndarray, int32
        Start nodes, reshaped as array
    
    node2: 4d np.ndarray, int32
        End nodes, reshaped as array
    """
    nEdge = nhood.shape[0]
    nodes = np.arange(np.prod(shape),dtype=np.uint64).reshape(shape)
    node1 = np.tile(nodes,(nEdge,1,1,1))
    node2 = np.full(node1.shape,-1,dtype=np.uint64)

    for e in range(nEdge):
        node2[e, \
            max(0,-nhood[e,0]):min(shape[0],shape[0]-nhood[e,0]), \
            max(0,-nhood[e,1]):min(shape[1],shape[1]-nhood[e,1]), \
            max(0,-nhood[e,2]):min(shape[2],shape[2]-nhood[e,2])] = \
                nodes[max(0,nhood[e,0]):min(shape[0],shape[0]+nhood[e,0]), \
                     max(0,nhood[e,1]):min(shape[1],shape[1]+nhood[e,1]), \
                     max(0,nhood[e,2]):min(shape[2],shape[2]+nhood[e,2])]

    return (node1, node2)


def affgraph_to_edgelist(aff,nhood):
    node1,node2 = nodelist_like(aff.shape[1:],nhood)
    return (node1.ravel(),node2.ravel(),aff.ravel())

def affgraph_to_seg(aff,nhood):
    """
    Input:
    
    affinity_gt: 4d np.ndarray
        Affinity graph of shape (#edges, x, y, z)
        1: connected, 0: disconnected 
    nhood: 2d np.ndarray
        Neighbourhood pattern specifying the edges in the affinity graph
        Shape: (#edges, ndim)
        nhood[i] contains the displacement coordinates of edge i
        The number and order of edges is arbitrary
        
    Output:
    
    seg_gt: 3d np.ndarray, int 
        Volume of segmentation IDs
    
    """
    (node1,node2,edge) = affgraph_to_edgelist(aff,nhood)
    (seg,segSizes) = connected_components(int(np.prod(aff.shape[1:])),node1,node2,edge)
    seg = seg.reshape(aff.shape[1:])
    return seg

def mk_cont_table(seg1,seg2):
    cont_table = scipy.sparse.coo_matrix((np.ones(seg1.shape),(seg1,seg2))).toarray()
    return cont_table

def compute_V_rand_N2(segTrue,segEst):
    """
    Computes Rand index of ``seg_pred`` w.r.t ``seg_true``.
    The input arrays both contain label IDs and
    may be of arbitrary, but equal, shape.
    
    Pixels which are have ID in the true segmentation are not counted!
    
    Parameters:
    
    seg_true: np.ndarray
      True segmentation, IDs
    seg_pred: np.ndarray
      Predicted segmentation
    
    Returns
    -------
    
    ri: ???
    """
    segTrue = segTrue.ravel()
    segEst = segEst.ravel()
    idx = segTrue != 0
    segTrue = segTrue[idx]
    segEst = segEst[idx]

    cont_table = scipy.sparse.coo_matrix((np.ones(segTrue.shape),(segTrue,segEst))).toarray()
    P = cont_table/cont_table.sum()
    t = P.sum(axis=0)
    s = P.sum(axis=1)

    V_rand_split = (P**2).sum() / (t**2).sum()
    V_rand_merge = (P**2).sum() / (s**2).sum()
    V_rand = 2*(P**2).sum() / ((t**2).sum()+(s**2).sum())

    return (V_rand,V_rand_split,V_rand_merge)

def rand_index(segTrue,segEst):
    segTrue = segTrue.ravel()
    segEst = segEst.ravel()
    idx = segTrue != 0
    segTrue = segTrue[idx]
    segEst = segEst[idx]

    tp_plus_fp = comb(np.bincount(segTrue), 2).sum()
    tp_plus_fn = comb(np.bincount(segEst), 2).sum()
    A = np.c_[(segTrue, segEst)]
    tp = sum(comb(np.bincount(A[A[:, 0] == i, 1]), 2).sum()
             for i in set(segTrue))
    fp = tp_plus_fp - tp
    fn = tp_plus_fn - tp
    tn = comb(len(A), 2) - tp - fp - fn
    ri = (tp + tn) / (tp + fp + fn + tn)
    prec = tp/(tp+fp)
    rec = tp/(tp+fn)
    fscore = 2*prec*rec/(prec+rec)
    return (ri,fscore,prec,rec)

def mknhood2d(radius=1):
    # Makes nhood structures for some most used dense graphs.

    ceilrad = np.ceil(radius)
    x = np.arange(-ceilrad,ceilrad+1,1)
    y = np.arange(-ceilrad,ceilrad+1,1)
    [i,j] = np.meshgrid(y,x)

    idxkeep = (i**2+j**2)<=radius**2
    i=i[idxkeep].ravel(); j=j[idxkeep].ravel();
    zeroIdx = np.ceil(len(i)/2).astype(np.int32);

    nhood = np.vstack((i[:zeroIdx],j[:zeroIdx])).T.astype(np.int32)
    return np.ascontiguousarray(np.flipud(nhood))

def mknhood3d(radius=1):
    """
    Makes nhood structures for some most used dense graphs.
    The neighborhood reference for the dense graph representation we use
    nhood(1,:) is a 3 vector that describe the node that conn(:,:,:,1) connects to
    so to use it: conn(23,12,42,3) is the edge between node [23 12 42] and [23 12 42]+nhood(3,:)
    See? It's simple! nhood is just the offset vector that the edge corresponds to.
    """

    ceilrad = np.ceil(radius)
    x = np.arange(-ceilrad,ceilrad+1,1)
    y = np.arange(-ceilrad,ceilrad+1,1)
    z = np.arange(-ceilrad,ceilrad+1,1)
    [i,j,k] = np.meshgrid(z,y,x)

    idxkeep = (i**2+j**2+k**2)<=radius**2
    i=i[idxkeep].ravel(); j=j[idxkeep].ravel(); k=k[idxkeep].ravel();
    zeroIdx = np.ceil(len(i)/2).astype(np.int32);

    nhood = np.vstack((k[:zeroIdx],i[:zeroIdx],j[:zeroIdx])).T.astype(np.int32)
    return np.ascontiguousarray(np.flipud(nhood))

def mknhood3d_aniso(radiusxy=1,radiusxy_zminus1=1.8):
    # Makes nhood structures for some most used dense graphs.

    nhoodxyz = mknhood3d(radiusxy)
    nhoodxy_zminus1 = mknhood2d(radiusxy_zminus1)
    
    nhood = np.zeros((nhoodxyz.shape[0]+2*nhoodxy_zminus1.shape[0],3),dtype=np.int32)
    nhood[:3,:3] = nhoodxyz
    nhood[3:,0] = -1
    nhood[3:,1:] = np.vstack((nhoodxy_zminus1,-nhoodxy_zminus1))

    return np.ascontiguousarray(nhood)


        
        
        
def scramble_sort(array, stochastic_malis_param):
    """
    This function scrambles a pre-sorted array. The
    stochastic malis_param controls how much it gets scrambled,
    higher means more scrambled

    params:
        stochastic_malis_parameter: int
            roughly determines by how many places rows are shuffled
    """
    if stochastic_malis_param == 0:
        return array
    
    for array_idx in range(len(array)):
        # only move every 2nd item
        if np.random.uniform() < 0.5:
            continue
        move_by = np.random.randint(0, stochastic_malis_param)
        move_to_idx = array_idx - move_by
        if  move_to_idx > 0:
            aux = array[array_idx]
            array[array_idx] = array[move_to_idx]
            array[move_to_idx] = aux
    return array


@cython.boundscheck(False)
@cython.wraparound(False)
def build_tree(labels, edge_weights, neighborhood,
               stochastic_malis_param=0):
    '''find tree of edges linking regions.

    params:
        labels: (D, W, H) integer
            labels.
        edge_weights: (K, D, W, H) float
            Kth entry corresponds to Kth offset in neighborhood
        neighborhood: (K, 3)
            offsets from pixel to linked pixel
        stochastic_malis_parameter: int
            roughly determines by how many places rows are shuffled

    returns:
        edge tree (D * W * H, 3) (int32)
            each row corresponds to one edge

            first column
                index into the flattened edge array,
                indicates which edge the current row corresponds to 
            second, third column
                are indices into edge_tree itself (not flattened edge array!)
                indicate the rows in edge_tree that are the parents of the two sub regions
                -1 means the sub-region is just the voxel that the edge connects
    '''
    cdef int [:, :] neighborhood_view = neighborhood
    cdef int D, W, H
    cdef unsigned int[:, :, :] merged_labels
    cdef unsigned int [:] merged_labels_raveled
    cdef int [:] region_parents
    cdef int [:, :] edge_tree

    # get shape information
    D, W, H = labels.shape

    # get flattened edge_weights
    ew_flat = edge_weights.ravel()

    # this acts as both the merged label matrix, as well as the pixel-to-pixel
    # linking graph.
    merged_labels_array = np.arange(labels.size, dtype=np.uint32).reshape((D, W, H))
    merged_labels = merged_labels_array
    merged_labels_raveled_array = merged_labels_array.view()
    merged_labels_raveled_array.shape = (-1,)
    merged_labels_raveled = merged_labels_raveled_array

    # edge that last merged a region, or -1 if this pixel hasn't been merged
    # into a region, yet.
    region_parents = - np.ones(np.asarray(merged_labels_raveled).shape, dtype=np.int32)

    # edge tree
    edge_tree = - np.ones((D * W * H, 3), dtype=np.int32)

    # sort array and get corresponding indices
    cdef unsigned int [:] ordered_indices = ew_flat.argsort()[::-1].astype(np.uint32)

    # scramble the sorting a bit
    ordered_indices = scramble_sort(ordered_indices, stochastic_malis_param)

    cdef int order_index = 0 # index into edge tree
    cdef int edge_idx
    cdef int d_1, w_1, h_1, k, d_2, w_2, h_2
    cdef unsigned int orig_label_1, orig_label_2, region_label_1, region_label_2, new_label
    cdef int [:] offset
    cdef int [::1] ew_shape = np.array(edge_weights.shape).astype(np.int32)
    cdef int return_idxes[4]
    cdef int n_loop_iterations = len(ordered_indices)
    cdef int i

    with nogil:
        for i in range(n_loop_iterations):
            edge_idx = ordered_indices[i]
            # the size of ordered_indices is k times bigger than the amount of
            # voxels, but the amount of merger edges is much smaller

            # get first voxel connected by the current edge
            my_unravel_index(edge_idx, ew_shape, return_idxes)
            k = return_idxes[0]
            d_1 = return_idxes[1]
            w_1 = return_idxes[2]
            h_1 = return_idxes[3]

            # get idxes of second voxel connected by this edge
            offset = neighborhood_view[k,:]
            d_2 = d_1 + offset[0]
            w_2 = w_1 + offset[1]
            h_2 = h_1 + offset[2]


            # ignore out-of-volume links
            if ((not 0 <= d_2 < D) or
                (not 0 <= w_2 < W) or
                (not 0 <= h_2 < H)):
                continue

            orig_label_1 = merged_labels[d_1, w_1, h_1]
            orig_label_2 = merged_labels[d_2, w_2, h_2]

            region_label_1 = chase(merged_labels_raveled, orig_label_1)
            region_label_2 = chase(merged_labels_raveled, orig_label_2)

            if region_label_1 == region_label_2:
                # already linked in tree, do not create a new edge.
                continue

            # make the entry in the edge tree
            edge_tree[order_index, 0] = edge_idx
            edge_tree[order_index, 1] = region_parents[region_label_1]
            edge_tree[order_index, 2] = region_parents[region_label_2]

            # choose one of the two labels to be the new label
            new_label = min(region_label_1, region_label_2)

            # one of these two is already the new label, but for simplicity sake
            # we assign the new label to both
            merged_labels_raveled[region_label_1] = new_label
            merged_labels_raveled[region_label_2] = new_label

            # store parent edge of region by location in tree
            region_parents[region_label_1] = order_index
            region_parents[region_label_2] = order_index

            # increase index of next to be assigned element in edge_tree
            # this must not be incremented earlier, because lots of edges
            # that link to voxels that already belong to the same region
            # will ultimately be ignored and hence order_indedx shouldn't
            # be increased
            order_index += 1
    return np.asarray(edge_tree)


# the following struct will hold information about recursion while traversing the tree
cdef struct stackelement:
    int edge_tree_idx
    int child_1_status
    int child_2_status
    unordered_map[unsigned int, unsigned long]* region_counts_1


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void compute_pairs_iterative(
                unsigned int[:, :, :] labels,
                int[::1] ew_shape,
                int[:, :] neighborhood, 
                int[:, :] edge_tree, 
                int edge_tree_idx, 
                unsigned long[:, :, :, :] pos_pairs, 
                unsigned long[:, :, :, :] neg_pairs,
                int keep_objs_per_edge,
                int count_method,
                int ignore_background=True) nogil:

    cdef int linear_edge_index, child_1, child_2
    cdef int d_1, w_1, h_1, k, d_2, w_2, h_2
    cdef int return_idxes[4]
    cdef int[:] offset
    cdef unordered_map[unsigned int, unsigned long] *return_dict
    cdef unordered_map[unsigned int, unsigned long] *region_counts_2
    cdef stack[stackelement*] mystack
    cdef stackelement *stackentry, *next_stackentry
    cdef int key_smallest_count # used to determine smalles object counts
    cdef unsigned long smallest_count
    cdef unsigned long paircount

    # create the first entry on the stack
    next_stackentry = <stackelement*> malloc(sizeof(stackelement))
    next_stackentry.edge_tree_idx = edge_tree_idx
    next_stackentry.child_1_status = 0
    next_stackentry.child_2_status = 0
    mystack.push(next_stackentry)

    while not mystack.empty():

        stackentry = mystack.top()
        linear_edge_index = edge_tree[stackentry.edge_tree_idx, 0]
        my_unravel_index(linear_edge_index, ew_shape, return_idxes)
        k = return_idxes[0]
        d_1 = return_idxes[1]
        w_1 = return_idxes[2]
        h_1 = return_idxes[3]

        ########################################################################
        # Child 1
        if stackentry.child_1_status == 0:
            child_1 = edge_tree[stackentry.edge_tree_idx, 1]
            if child_1 == -1:
                # add this region count to the current stackentry
                stackentry.region_counts_1 = new unordered_map[unsigned int, unsigned long]()
                dereference(stackentry.region_counts_1)[labels[d_1, w_1, h_1]] = 1
                stackentry.child_1_status = 2
            else:
                stackentry.child_1_status = 1
                # create the next entry on the stack
                next_stackentry = <stackelement*> malloc(sizeof(stackelement))
                next_stackentry.edge_tree_idx = child_1
                next_stackentry.child_1_status = 0
                next_stackentry.child_2_status = 0
                mystack.push(next_stackentry)

                continue
        elif stackentry.child_1_status == 1:
            stackentry.region_counts_1 = return_dict
            stackentry.child_1_status = 2

        ########################################################################
        # Child 2
        if stackentry.child_2_status == 0:
            child_2 = edge_tree[stackentry.edge_tree_idx, 2]
            if child_2 == -1:
                offset = neighborhood[k, :]
                d_2 = d_1 + offset[0]
                w_2 = w_1 + offset[1]
                h_2 = h_1 + offset[2]

                # create new region_counts_2
                region_counts_2 = new unordered_map[unsigned int, unsigned long]()
                # add this region count to the current stackentry
                dereference(region_counts_2)[labels[d_2, w_2, h_2]] = 1
            else:
                stackentry.child_2_status = 1
                next_stackentry = <stackelement*> malloc(sizeof(stackelement))
                next_stackentry.edge_tree_idx = child_2
                next_stackentry.child_1_status = 0
                next_stackentry.child_2_status = 0
                mystack.push(next_stackentry)
                continue
        elif stackentry.child_2_status == 1:
            region_counts_2 = return_dict


        #########################################################################
        # compute pair-counts and save them in pos_pairs and neg_pairs
        # mark this edge as done so recursion doesn't hit it again
        edge_tree[stackentry.edge_tree_idx, 0] = -1

        for item1 in dereference(stackentry.region_counts_1):
            for item2 in dereference(region_counts_2):
                if count_method == 0:
                    paircount = item1.second * item2.second
                elif count_method == 1:
                    paircount = <int> (sqrt(item1.second) * sqrt(item2.second))
                elif count_method == 2:
                    # making sure that the paircount is at least 1:
                    paircount = <int> ((log(item1.second + 1) * log(item2.second + 1)) + 1)
                if item1.first == item2.first and \
                    not item1.first == 0 and \
                    not item2.first == 0:
                    # the labels were the same -> positive count
                    pos_pairs[k, d_1, w_1, h_1] += paircount
                else:
                    if ignore_background and (item1.first == 0 or item2.first == 0):
                        pass
                    else:
                        # the labels were different or they were 0 (background)
                        neg_pairs[k, d_1, w_1, h_1] += paircount

        #########################################################################
        # Prepare return dict that will be used in the next iteration of the loop
        # create new return dict
        return_dict = new unordered_map[unsigned int, unsigned long]()

        # add counts to return_dict
        for item1 in dereference(stackentry.region_counts_1):
            dereference(return_dict)[item1.first] = item1.second

        for item2 in dereference(region_counts_2):
            if dereference(return_dict).count(item2.first) == 1:
                dereference(return_dict)[item2.first] = dereference(return_dict)[item2.first] + item2.second
            else:
                # this object was not present in those objects added from
                # region_counts_1
                dereference(return_dict)[item2.first] = item2.second

        # delete smallest object counts in return dict until it has keep_objs_per_edge counts left
        while return_dict.size() > keep_objs_per_edge:
            # find smallest object count
            #   first, set the key of the smallest count and the smallest count itself to those
            #   values of the first item in return_dict
            #   then loop over the rest of the keys and exchange the key and smallest count with
            #   any that has a lower count
            key_smallest_count = dereference(return_dict.begin()).first
            smallest_count = dereference(return_dict.begin()).second
            for item in dereference(return_dict):
                if item.second < smallest_count:
                    key_smallest_count = item.first
                    smallest_count = item.second
            # delete key with smallest count
            return_dict.erase(key_smallest_count)

        # free region counts and the stackentry. Remember we are using return_dict in the next
        # iteration of the loop so we need it
        free(stackentry)
        del stackentry.region_counts_1
        del region_counts_2
        mystack.pop()
    del return_dict


def compute_pairs_with_tree(labels, edge_weights, neighborhood, edge_tree, keep_objs_per_edge=10,
                            count_method=0, ignore_background=True):
    cdef unsigned int [:, :, :] labels_view = labels
    cdef int [:, :] neighborhood_view = neighborhood
    cdef int [:, :] edge_tree_view = edge_tree
    cdef int [::1] ew_shape = np.array(edge_weights.shape).astype(np.int32)
    cdef unsigned long [:, :, :, :] pos_pairs = np.zeros((neighborhood.shape[0],) + labels.shape, dtype=np.uint64)
    cdef unsigned long [:, :, :, :] neg_pairs = np.zeros((neighborhood.shape[0],) + labels.shape, dtype=np.uint64)
    cdef int keep_objs = keep_objs_per_edge
    cdef int count_method_int = count_method
    cdef int ignore_background_ctype = ignore_background

    # process tree from root (later in array) to leaves (earlier)
    cdef int idx
    for idx in range(edge_tree.shape[0] - 1, 0, -1):
        if edge_tree[idx, 0] == -1:
            continue
        with nogil:
            compute_pairs_iterative(labels_view, ew_shape, neighborhood_view,
                edge_tree_view, idx, pos_pairs, neg_pairs, keep_objs, count_method_int,
                ignore_background=ignore_background_ctype)
    return np.array(pos_pairs), np.array(neg_pairs)


