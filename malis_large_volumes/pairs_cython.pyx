import numpy as np
cimport numpy as np
cimport cython
from cython.view cimport array as cvarray
from .malis_python import merge as merge_python
from .malis_python import chase as chase_python
import sys
from libcpp.unordered_map cimport unordered_map
from libcpp.stack cimport stack
from libc.stdlib cimport malloc, free, realloc
from libc.math cimport log
from cython.operator cimport dereference


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


def scramble_sort(array, stochastic_malis_param):
    """
    This function scrambles a pre-sorted array. The
    stochastic malis_param controls how much it gets scrambled,
    higher means more scrambled
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
        labels = (D, W, H) integer label volume.  0s ignored
        edge_weights = (K, D, W, H) floating point values.
                  Kth entry corresponds to Kth offset in neighborhood.
        neighborhood = (K, 3) offsets from pixel to linked pixel.

        returns: edge tree (3, D * W * H) (int32)
            array of: (linear edge index, child 1 index in edge tree, child 2 index in edge tree)
            If a child index is -1, it's a pixel, not an edge, and can be
                inferred from the linear edge index.
            Tree is terminated by linear_edge_index == -1

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

            # merge regions
            new_label = min(region_label_1, region_label_2)
            if new_label != region_label_1:
                merged_labels_raveled[region_label_1] = new_label
            else:
                merged_labels_raveled[region_label_2] = new_label

            # store parent edge of region by location in tree
            region_parents[new_label] = order_index

            # increase index of next to be assigned element in edge_tree
            # this can't be incremented earlier, because lots of edges
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
cdef void compute_pairs_iterative(  \
                unsigned int[:, :, :] labels,
                int[::1] ew_shape,
                int[:, :] neighborhood, 
                int[:, :] edge_tree, 
                int edge_tree_idx, 
                unsigned long[:, :, :, :] pos_pairs, 
                unsigned long[:, :, :, :] neg_pairs,
                int keep_objs_per_edge,
                int count_method) nogil:

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
                    paircount = item1.second * <int>log(item2.second+1) + \
                                <int>log(item1.second+1) * item2.second
                if item1.first == item2.first and \
                    not item1.first == 0 and \
                    not item2.first == 0:
                    # the labels were the same -> positive count
                    pos_pairs[k, d_1, w_1, h_1] += paircount
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
                            count_method=0):
    cdef unsigned int [:, :, :] labels_view = labels
    cdef int [:, :] neighborhood_view = neighborhood
    cdef int [:, :] edge_tree_view = edge_tree
    cdef int [::1] ew_shape = np.array(edge_weights.shape).astype(np.int32)
    cdef unsigned long [:, :, :, :] pos_pairs = np.zeros((neighborhood.shape[0],) + labels.shape, dtype=np.uint64)
    cdef unsigned long [:, :, :, :] neg_pairs = np.zeros((neighborhood.shape[0],) + labels.shape, dtype=np.uint64)
    cdef int keep_objs = keep_objs_per_edge
    cdef int count_method_int = count_method

    # process tree from root (later in array) to leaves (earlier)
    cdef int idx
    for idx in range(edge_tree.shape[0] - 1, 0, -1):
        if edge_tree[idx, 0] == -1:
            continue
        with nogil:
            compute_pairs_iterative(labels_view, ew_shape, neighborhood_view,
                edge_tree_view, idx, pos_pairs, neg_pairs, keep_objs, count_method_int)
    return np.array(pos_pairs), np.array(neg_pairs)
