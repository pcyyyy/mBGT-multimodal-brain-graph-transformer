import numpy as np

def floyd_warshall(adjacency_matrix):
    nrows, ncols = adjacency_matrix.shape
    assert nrows == ncols
    n = nrows

    adj_mat_copy = np.array(adjacency_matrix, dtype=np.int64, order='C')
    assert adj_mat_copy.flags['C_CONTIGUOUS']
    M = adj_mat_copy
    path = -1 * np.ones((n, n), dtype=np.int64)

    for i in range(n):
        for j in range(n):
            if i == j:
                M[i][j] = 0
            elif M[i][j] == 0:
                M[i][j] = 510

    for k in range(n):
        for i in range(n):
            for j in range(n):
                M_ik = M[i][k]
                cost_ikkj = M_ik + M[k][j]
                M_ij = M[i][j]
                if M_ij > cost_ikkj:
                    M[i][j] = cost_ikkj
                    path[i][j] = k

    for i in range(n):
        for j in range(n):
            if M[i][j] >= 510:
                path[i][j] = 510
                M[i][j] = 510

    return M, path

def get_all_edges(path, i, j):
    k = path[i][j]
    if k == -1:
        return []
    else:
        return get_all_edges(path, i, k) + [k] + get_all_edges(path, k, j)

def gen_edge_input(max_dist, path, edge_feat):
    nrows, ncols = path.shape
    assert nrows == ncols
    n = nrows
    max_dist_copy = max_dist

    path_copy = np.array(path, dtype=np.int64, order='C')
    edge_feat_copy = np.array(edge_feat, dtype=np.int64, order='C')
    assert path_copy.flags['C_CONTIGUOUS']
    assert edge_feat_copy.flags['C_CONTIGUOUS']

    edge_fea_all = -1 * np.ones((n, n, max_dist_copy, edge_feat.shape[-1]), dtype=np.int64)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if path_copy[i][j] == 510:
                continue
            path = [i] + get_all_edges(path_copy, i, j) + [j]
            num_path = len(path) - 1
            for k in range(num_path):
                edge_fea_all[i, j, k, :] = edge_feat_copy[path[k], path[k+1], :]

    return edge_fea_all
