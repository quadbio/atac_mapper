from __future__ import annotations

import os
from typing import Literal

import anndata as ad
import cloudpickle
import numpy as np
import scanpy as sc
import scvi
from scarches.models.scpoli import scPoli


# adapted from HNOCA-tools
# https://github.com/devsystemslab/HNOCA-tools/blob/main/src/hnoca/mapping/mapper.py
class AtlasMapper:
    """A class to map a query dataset to a reference dataset using scPoli"""

    def __init__(
        self,
        ref_model: scPoli,
        layer: str | None = None,
    ):
        """
        Initialize the AtlasMapper object

        Args:
            ref_model: The reference model to map the query dataset to.
            layer: The layer in the AnnData object to use for features. If None, use .X
        """
        import scarches

        self.scvi = scvi
        self.scarches = scarches

        self.layer = layer

        self.model_type = self._check_model_type(ref_model)
        self.ref_model = ref_model
        self.ref_adata = ref_model.adata
        self.query_model = None

    def _validate_features(self, query_adata: ad.AnnData, query_layer: str | None = None):
        """
        Validate that the features in query and reference match

        Args:
            query_adata: Query dataset to validate
            query_layer: Optional layer name for query data. If None, uses the layer
                        specified during initialization. Can be different from the reference layer.

        Raises
        ------
            ValueError: If features don't match between reference and query, or if
                      specified layers don't exist
        """
        ref_layer = self.layer

        # Check if layers exist if specified
        if ref_layer is not None:
            if ref_layer not in self.ref_adata.layers:
                raise ValueError(
                    f"Layer '{ref_layer}' not found in reference data. Available layers: {list(self.ref_adata.layers.keys())}"
                )

        if query_layer is not None:
            if query_layer not in query_adata.layers:
                raise ValueError(
                    f"Layer '{query_layer}' not found in query data. Available layers: {list(query_adata.layers.keys())}"
                )

        # Get features from appropriate source
        ref_features = self.ref_adata.var_names.tolist()
        query_features = query_adata.var_names.tolist()

        if not len(ref_features) == len(query_features):
            raise ValueError(
                f"Number of features differs: reference has {len(ref_features)}, query has {len(query_features)}"
            )

        if not all(ref_feat == query_feat for ref_feat, query_feat in zip(ref_features, query_features)):
            raise ValueError("Features in reference and query datasets don't match")

        # Validate that matrices have the same number of features using appropriate layers
        ref_data = self.ref_adata.layers[ref_layer] if ref_layer else self.ref_adata.X
        query_data = query_adata.layers[query_layer] if query_layer else query_adata.X

        if ref_data.shape[1] != query_data.shape[1]:
            raise ValueError(
                f"Feature dimensions don't match: reference has {ref_data.shape[1]}, query has {query_data.shape[1]}"
            )

    def map_query(
        self,
        query_adata: ad.AnnData,
        retrain: Literal["partial", "full", "none"] = "partial",
        labeled_indices: np.ndarray | None = None,
        query_layer: str | None = None,
        embed_key: str | None = None,
        **kwargs,
    ) -> ad.AnnData:
        """
        Map a query dataset to the reference

        Args:
            query_adata: Query dataset to map
            retrain: How to train the query model:
                - 'partial': freeze encoder weights and update other parameters
                - 'full': retrain all parameters
                - 'none': use reference model as is
            batch_key: Key for batch information in query_adata.obs
            query_layer: Layer in query_adata to use. If None, uses the layer specified
                        during AtlasMapper initialization. This allows using different
                        layers for reference and query data.
            embed_key: name of obsm key to use for embedding. If None, uses 'X_scpoli' by default.
            **kwargs: Additional arguments passed to model.train()

        Returns
        -------
            AnnData object with query data mapped to reference
        """
        # Check features match and validate layers
        self._validate_features(query_adata, query_layer)

        # Determine which layer to use - query_layer takes precedence over self.layer
        layer = query_layer if query_layer is not None else self.layer

        # If using a layer, create a copy of AnnData with that layer as X
        if layer is not None:
            if layer not in query_adata.layers:
                raise ValueError(
                    f"Layer '{layer}' not found in query data. Available layers: {list(query_adata.layers.keys())}"
                )
            query_adata = query_adata.copy()
            query_adata.X = query_adata.layers[layer]

        if retrain == "none":
            self.query_model = self.ref_model
            self.query_adata = query_adata
        else:
            self._train_scpoli(query_adata, retrain, labeled_indices, **kwargs)
            self.query_adata = self.query_model.adata

        # Get the latent representation after training
        embed_key = embed_key if embed_key is not None else "X_scpoli"
        query_adata.obsm[embed_key] = self.query_model.get_latent(query_adata, mean=True)

        return query_adata

    def _train_scpoli(self, query_adata: sc.AnnData, retrain: str = "partial", labeled_indices=None, **kwargs):
        """Train a new scpoli model on the query data"""
        labeled_indices = [] if labeled_indices is None else labeled_indices

        # Set cell type to unknown if not present
        missing_cell_types = np.setdiff1d(np.array(self.ref_model.cell_type_keys_), query_adata.obs.columns)
        if len(missing_cell_types) > 0:
            query_adata.obs[missing_cell_types] = "Unknown"

        vae_q = self.scarches.models.scPoli.load_query_data(
            query_adata,
            reference_model=self.ref_model,
            unknown_ct_names=["Unknown"],
            labeled_indices=labeled_indices,
        )

        vae_q.train(**kwargs)

        self.query_model = vae_q

    def get_latent_representation(self) -> np.ndarray:
        """
        Get latent representation of query data after mapping

        Returns
        -------
            Array of latent representations
        """
        if self.query_model is None:
            raise ValueError("No query data has been mapped yet")

        return self.query_model.get_latent()

    def _check_model_type(self, model) -> type:
        """
        Check that the model is a valid scPoli model

        Args:
            model: Model to check

        Returns
        -------
            The model class type
        """
        try:
            from scarches.models.scpoli import scPoli

            if not isinstance(model, scPoli):
                raise ValueError("Reference model must be a scPoli model")
        except ImportError as err:
            raise ImportError(
                "scarches is not installed or cannot be imported. Please install it with 'pip install scarches'"
            ) from err
        return type(model)

    def save(self, output_dir: str):
        """
        Save the mapper object to disk

        Args:
            output_dir: str
                The directory to save the mapper object
        """
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "mapper.pkl"), "wb") as f:
            cloudpickle.dump(self, f)

    @classmethod
    def load(cls, input_dir: str):
        """
        Load the mapper object from disk

        Args:
            input_dir: str
                The directory to load the mapper object
        """
        with open(os.path.join(input_dir, "mapper.pkl"), "rb") as f:
            mapper = cloudpickle.load(f)
        return mapper
