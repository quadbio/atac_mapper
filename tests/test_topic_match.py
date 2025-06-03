import numpy as np
import pandas as pd
import pytest
from scipy import sparse as sp

from atac_mapper.topic_matching import TopicMatch

from .generate_test_data import generate_test_data


@pytest.fixture
def test_data():
    """Generate test data for topic matching"""
    return generate_test_data(
        n_regions=20,  # Much smaller for faster tests
        n_topics=5,  # Much smaller for faster tests
        n_cells=10,  # Much smaller for faster tests
        sparsity=0.95,
    )


def test_topic_match_initialization(test_data):
    """Test that TopicMatch class initializes correctly"""
    region_topic_df, _ = test_data
    topic_matcher = TopicMatch(region_topic_df)
    assert topic_matcher.region_topic.shape == (5, 20)  # (n_topics, n_regions)


def test_infer_topics(test_data):
    """Test topic inference on synthetic data"""
    region_topic_df, query_matrix = test_data
    topic_matcher = TopicMatch(region_topic_df)

    print(f"\nShape of region_topic: {topic_matcher.region_topic.shape}")
    print(f"Shape of query_matrix: {query_matrix.shape}")

    # Run inference with single thread for debugging
    topic_distributions = topic_matcher.infer_topics(
        query=query_matrix,
        njobs=1,  # Single thread for debugging
        n_iterations=20,  # Fewer iterations for testing
        tol=1e-4,
    )

    # Check output shape and properties
    assert topic_distributions.shape == (10, 5)  # (n_cells, n_topics)
    assert np.allclose(topic_distributions.sum(axis=1), 1, atol=1e-5)  # Sum to 1
    assert (topic_distributions >= 0).all()  # Non-negative

    # Save results
    results_df = pd.DataFrame(
        topic_distributions,
        index=[f"cell_{i}" for i in range(topic_distributions.shape[0])],
        columns=[f"topic_{i}" for i in range(topic_distributions.shape[1])],
    )
    results_df.to_csv("test_data/inferred_topics.csv")

    # Save some statistics about the results
    with open("test_data/inference_stats.txt", "w") as f:
        f.write("Topic Distribution Statistics\n")
        f.write("===========================\n\n")
        f.write(f"Number of cells: {topic_distributions.shape[0]}\n")
        f.write(f"Number of topics: {topic_distributions.shape[1]}\n")
        f.write(f"Average topic probability: {topic_distributions.mean():.4f}\n")
        f.write(f"Max topic probability: {topic_distributions.max():.4f}\n")
        f.write(f"Min topic probability: {topic_distributions.min():.4f}\n")
        f.write("\nPer-topic mean probabilities:\n")
        for i, mean_prob in enumerate(topic_distributions.mean(axis=0)):
            f.write(f"Topic {i}: {mean_prob:.4f}\n")


def test_dimension_mismatch(test_data):
    """Test that appropriate error is raised for dimension mismatch"""
    region_topic_df, _ = test_data
    topic_matcher = TopicMatch(region_topic_df)

    # Create query with wrong dimensions
    wrong_query = sp.csr_matrix((30, 10))  # Wrong number of regions

    with pytest.raises(ValueError) as excinfo:
        topic_matcher.infer_topics(wrong_query)
    assert "Number of regions in query" in str(excinfo.value)


def test_input_validation(test_data):
    """Test input validation for query matrix"""
    region_topic_df, _ = test_data
    topic_matcher = TopicMatch(region_topic_df)

    # Try with dense array instead of sparse
    wrong_type_query = np.random.rand(20, 10)  # (regions x cells)

    with pytest.raises(ValueError) as excinfo:
        topic_matcher.infer_topics(wrong_type_query)
    assert "must be a sparse matrix" in str(excinfo.value)
