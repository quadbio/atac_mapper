import os

os.environ["SCIPY_ARRAY_API"] = "1"
import anndata as ad
import numpy as np
import pytest
from scarches.models.scpoli import scPoli

from atac_mapper.reference_mapping.mapping_atac import AtlasMapper


@pytest.fixture
def reference_dataset():
    # Create a simple reference dataset
    n_cells, n_features = 100, 50
    X = np.random.rand(n_cells, n_features)
    adata = ad.AnnData(X)
    adata.obs["batch"] = ["batch1"] * n_cells
    adata.obs["cell_type"] = ["type1"] * (n_cells // 2) + ["type2"] * (n_cells // 2)
    return adata


@pytest.fixture
def query_dataset(reference_dataset):
    # Create a matching query dataset
    n_cells = 50
    X = np.random.rand(n_cells, reference_dataset.n_vars)
    adata = ad.AnnData(X, var=reference_dataset.var)
    # Use the same batch name as reference to match scPoli condition requirements
    adata.obs["batch"] = ["batch1"] * n_cells
    # Add cell_type information to match reference
    adata.obs["cell_type"] = ["type1"] * (n_cells // 2) + ["type2"] * (n_cells // 2)
    return adata


@pytest.fixture
def reference_model(reference_dataset):
    # Initialize and train a scPoli model

    import scvi

    scvi.settings.seed = 0
    reference_dataset.obs["batch"] = reference_dataset.obs["batch"].astype("category")
    reference_dataset.obs["cell_type"] = reference_dataset.obs["cell_type"].astype("category")

    # Create a properly formatted scPoli model
    model = scPoli(
        reference_dataset,
        condition_keys=["batch"],
        hidden_layer_sizes=[128],
        latent_dim=20,
        embedding_dims=5,
        recon_loss="mse",
    )
    model.train(max_epochs=1)  # Short training for testing

    # Mock the cell_type_keys_ attribute to avoid None values
    model.cell_type_keys_ = ["cell_type"]

    return model


def test_init_atlas_mapper(reference_model):
    """Test AtlasMapper initialization"""
    mapper = AtlasMapper(reference_model)
    assert mapper.ref_model == reference_model
    assert mapper.ref_adata is reference_model.adata
    assert mapper.query_model is None


def test_validate_features(reference_model, query_dataset):
    """Test feature validation"""
    mapper = AtlasMapper(reference_model)

    # Should pass with matching features
    mapper._validate_features(query_dataset)

    # Should fail with mismatched features
    mismatched_query = query_dataset[:, :10].copy()
    with pytest.raises(ValueError, match="Number of features differs"):
        mapper._validate_features(mismatched_query)


def test_map_query(reference_model, query_dataset):
    """Test query mapping functionality"""
    mapper = AtlasMapper(reference_model)

    # Test with different retrain options
    for retrain in ["partial", "full", "none"]:
        mapped_data = mapper.map_query(
            query_dataset,
            retrain=retrain,
            max_epochs=1,  # Short training for testing
        )

        # Check that the mapping produced expected outputs
        assert "X_scpoli" in mapped_data.obsm
        assert mapped_data.obsm["X_scpoli"].shape[0] == query_dataset.n_obs


def test_save_load(reference_model, tmp_path):
    """Test saving and loading functionality"""
    mapper = AtlasMapper(reference_model)

    # Save the mapper
    save_path = str(tmp_path / "test_mapper")
    mapper.save(save_path)
    assert os.path.exists(os.path.join(save_path, "mapper.pkl"))

    # Load the mapper
    loaded_mapper = AtlasMapper.load(save_path)
    assert isinstance(loaded_mapper, AtlasMapper)
    assert loaded_mapper.ref_adata.shape == mapper.ref_adata.shape


def test_layer_handling(reference_model, query_dataset):
    """Test handling of different data layers"""
    # Add a test layer to both datasets
    layer_name = "test_layer"
    reference_model.adata.layers[layer_name] = reference_model.adata.X * 2
    query_dataset.layers[layer_name] = query_dataset.X * 2

    # Test with layer specification
    mapper = AtlasMapper(reference_model, layer=layer_name)
    mapped_data = mapper.map_query(query_dataset, query_layer=layer_name, max_epochs=1)

    assert mapped_data is not None
    assert "X_scpoli" in mapped_data.obsm


def test_invalid_inputs(reference_model, query_dataset):
    """Test error handling for invalid inputs"""
    mapper = AtlasMapper(reference_model)

    # Test invalid retrain option
    with pytest.raises(ValueError):
        mapper.map_query(query_dataset, retrain="invalid")

    # Test invalid layer name
    with pytest.raises(ValueError):
        mapper.map_query(query_dataset, query_layer="nonexistent_layer")
