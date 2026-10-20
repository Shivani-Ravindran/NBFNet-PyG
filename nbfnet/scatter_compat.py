from torch_geometric.utils import scatter  # same signature as torch_scatter.scatter


def scatter_add(src, index, dim=0, out=None, dim_size=None):
    return scatter(src, index, dim=dim, dim_size=dim_size, reduce="sum")