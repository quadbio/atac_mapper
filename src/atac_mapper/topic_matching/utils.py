import numpy as np


def infer_topic_distribution(tf_vector, topic_word, n_topics, n_words, n_iterations=100, tol=1e-4):
    """Infer topic distribution for a single document.

    This function is reimplementation of reference topic inference from lda package
    https://lda.readthedocs.io/en/latest/

    Args:
        tf_vector: Term frequency vector for a single document
        topic_word: Topic-word matrix from reference data
        n_topics: Number of topics
        n_words: Number of words (features)
        n_iterations: Maximum number of iterations
        tol: Convergence tolerance

    Returns
    -------
        topic_dist: Inferred topic distribution for the document
    """
    # Print input shapes for debugging
    # print(f"Input shapes - tf_vector: {tf_vector.shape}, topic_word: {topic_word.shape}")

    # If the cell has no counts, return uniform distribution
    if np.sum(tf_vector) == 0:
        return np.ones(n_topics) / n_topics

    # Initialize topic distribution (uniform distribution as a starting point)
    topic_dist = np.ones(n_topics) / n_topics

    # EM algorithm
    for iter_num in range(n_iterations):
        old_topic_dist = topic_dist.copy()

        # E-step: calculate word topic assignments
        word_topic = topic_dist[:, np.newaxis] * topic_word
        word_topic /= word_topic.sum(axis=0, keepdims=True) + 1e-10

        # M-step: update topic distribution
        topic_dist = tf_vector @ word_topic.T
        topic_dist /= topic_dist.sum() + 1e-10

        # Check convergence
        delta = np.abs(topic_dist - old_topic_dist).max()
        if delta < tol:
            print(f"Converged after {iter_num + 1} iterations (delta={delta:.2e})")
            break

    return topic_dist
