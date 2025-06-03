from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import scipy.sparse as sp
from joblib import Parallel, delayed

from atac_mapper.topic_matching.utils import infer_topic_distribution

if TYPE_CHECKING:
    pass


class TopicMatch:
    """A class to map a query dataset to the cistopics of a reference dataset."""

    def __init__(self, region_topic: pd.DataFrame):
        # usually stored in cistopic_obj.selected_model.region_topic
        # output of cistopic analysis
        self.region_topic = region_topic.T.values

    def infer_topics(
        self,
        query: sp.csr_matrix | sp.csc_matrix | sp.spmatrix,
        njobs: int = -1,
        n_iterations: int = 100,
        tol: float = 1e-4,
    ) -> np.ndarray:
        """
        Map a query dataset to the reference dataset

        Args:
            query: The query dataset in sparse matrix format (regions x cells matrix, must have the same
                number of regions as the reference dataset).
            njobs: Number of parallel jobs to run. -1 means using all available cores.
            n_iterations: Number of iterations for convergence.
            tol: Tolerance for convergence.

        Returns
        -------
            topic_distributions: A numpy array of shape (n_cells, n_topics) containing the inferred
            topic distributions for each cell in the query dataset.
        """
        n_topics, n_regions = self.region_topic.shape
        n_regions_query, n_cells = query.shape

        if isinstance(query, (sp.csr_matrix, sp.csc_matrix, sp.spmatrix)):
            self.query = query
        else:
            raise ValueError("Query must be a sparse matrix (csr, csc, or spmatrix).")

        # Add dimension check
        if n_regions_query != n_regions:
            raise ValueError(
                f"Number of regions in query ({n_regions_query}) must match "
                f"number of regions in reference ({n_regions})"
            )

        # Function to infer topic distribution for a single cell
        def infer_for_cell(i):
            # Get the cell profile (transpose by using column slice)
            tf_vector = self.query[:, i].toarray().flatten()  # Get column i and convert to dense
            return infer_topic_distribution(
                tf_vector, self.region_topic, n_topics, n_regions, n_iterations=n_iterations, tol=tol
            )

        # Run in parallel
        results = Parallel(n_jobs=njobs)(delayed(infer_for_cell)(i) for i in range(n_cells))

        # Convert results to an array: shape (n_cells, n_topics)
        topic_distributions = np.vstack(results)

        return topic_distributions
