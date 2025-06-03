import os

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.io import mmwrite


def generate_test_data(
    n_regions: int = 200, n_topics: int = 50, n_cells: int = 1000, sparsity: float = 0.95, output_dir: str = "test_data"
) -> tuple[pd.DataFrame, sp.csr_matrix]:
    """
    Generate synthetic data for topic matching testing.

    Args:
        n_regions: Number of genomic regions
        n_topics: Number of topics
        n_cells: Number of cells
        sparsity: Fraction of zeros in cell-region matrix
        output_dir: Directory to save output files

    Returns:
        Tuple containing:
        - region_topic_df: DataFrame with region-topic distributions
        - region_cell_matrix: Sparse matrix with region-cell counts (regions x cells)
    """
    # Generate region-topic distribution (reference)
    region_topic = np.random.dirichlet(alpha=[0.5] * n_topics, size=n_regions)
    region_topic_df = pd.DataFrame(
        region_topic, index=[f"region_{i}" for i in range(n_regions)], columns=[f"topic_{i}" for i in range(n_topics)]
    )

    # Generate sparse region-cell matrix (regions x cells)
    n_entries = int(n_cells * n_regions * (1 - sparsity))

    # First ensure each cell has at least one entry
    col_indices = np.arange(n_cells)
    row_indices = np.random.randint(0, n_regions, n_cells)
    data = np.random.poisson(lam=2, size=n_cells)  # Count data for guaranteed entries

    # Then add remaining random entries
    remaining_entries = n_entries - n_cells
    if remaining_entries > 0:
        extra_cols = np.random.randint(0, n_cells, remaining_entries)
        extra_rows = np.random.randint(0, n_regions, remaining_entries)
        extra_data = np.random.poisson(lam=2, size=remaining_entries)

        row_indices = np.concatenate([row_indices, extra_rows])
        col_indices = np.concatenate([col_indices, extra_cols])
        data = np.concatenate([data, extra_data])

    # Create sparse matrix in CSR format (regions x cells)
    region_cell_matrix = sp.csr_matrix((data, (row_indices, col_indices)), shape=(n_regions, n_cells))

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Save files
    region_topic_df.to_csv(os.path.join(output_dir, "region_topic.csv"))
    mmwrite(os.path.join(output_dir, "region_cell_matrix.mtx"), region_cell_matrix)

    return region_topic_df, region_cell_matrix


if __name__ == "__main__":
    ref_dist, query_matrix = generate_test_data()
    print(f"Reference shape: {ref_dist.shape}")
    print(f"Query matrix shape: {query_matrix.shape} (regions x cells)")
    print(f"Query matrix sparsity: {1 - (query_matrix.nnz / (query_matrix.shape[0] * query_matrix.shape[1])):.2%}")
    print("Cells with no entries:", (query_matrix.sum(axis=0) == 0).sum())
