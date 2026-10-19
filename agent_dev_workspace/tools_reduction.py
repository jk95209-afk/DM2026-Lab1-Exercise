from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.sparse as sp
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from langchain_core.tools import tool

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

def make_tools(session):
    @tool
    def reduce_dimensions_tool(
        method: str = "pca",
        n_components: int = 2,
        random_state: Optional[int] = None,
        perplexity: float = 30.0,
        n_neighbors: int = 15
    ) -> Dict[str, Any]:
        """Reduce dimensions of the Document-Term Matrix using PCA, t-SNE, or UMAP, and plot a 2D scatter plot.

        Args:
            method: Dimensionality reduction technique ('pca', 'tsne', or 'umap').
            n_components: Number of target dimensions (default 2).
            random_state: Random seed for reproducible output.
            perplexity: Perplexity parameter for t-SNE (default 30.0).
            n_neighbors: Number of neighbors parameter for UMAP (default 15).

        Returns:
            Dict containing reduced coordinates, explained variance ratio (for PCA), method, and result_id.
        """
        if session.feature_matrix is None:
            return {"status": "error", "message": "No feature_matrix found in session. Please run build_dtm_tool first."}

        method = method.lower()
        if method not in ["pca", "tsne", "umap"]:
            return {"status": "error", "message": f"Invalid method '{method}'. Must be 'pca', 'tsne', or 'umap'."}

        if n_components < 2:
            return {"status": "error", "message": "n_components must be at least 2 for 2D plotting."}

        if method == "umap" and not UMAP_AVAILABLE:
            return {"status": "error", "message": "umap-learn package is not installed."}

        matrix = session.feature_matrix
        n_samples, n_features = matrix.shape

        if n_components > min(n_samples, n_features):
            return {"status": "error", "message": f"n_components ({n_components}) cannot exceed min(n_samples, n_features) ({min(n_samples, n_features)})."}

        if method == "tsne" and perplexity >= n_samples:
            return {"status": "error", "message": f"perplexity ({perplexity}) must be less than n_samples ({n_samples})."}

        if method == "umap" and n_neighbors > n_samples:
            return {"status": "error", "message": f"n_neighbors ({n_neighbors}) cannot be greater than n_samples ({n_samples})."}

        explained_variance_ratio = None

        if method == "pca":
            # Convert matrix to dense for PCA to handle mean-centering correctly
            if sp.issparse(matrix):
                matrix_data = matrix.toarray()
            else:
                matrix_data = np.asarray(matrix)
            model = PCA(n_components=n_components, random_state=random_state)
            coords = model.fit_transform(matrix_data)
            explained_variance_ratio = [float(round(val, 4)) for val in model.explained_variance_ratio_]
        elif method == "tsne":
            # TSNE supports sparse matrix with init="random"
            model = TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state, init="random")
            coords = model.fit_transform(matrix)
        elif method == "umap":
            # UMAP supports sparse matrix directly
            model = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, random_state=random_state)
            coords = model.fit_transform(matrix)

        # Retrieve labels giving precedence to category_name, then session.labels
        labels = None
        if session.dataframe is not None:
            for col in ["category_name", "label", "category"]:
                if col in session.dataframe.columns:
                    labels = session.dataframe[col].values
                    break
        if labels is None and session.labels is not None:
            labels = session.labels

        labels_arr = np.asarray(labels) if labels is not None else None

        # Build 2D scatter plot using first 2 components
        fig, ax = plt.subplots(figsize=(8, 6))
        plot_coords = coords[:, :2]

        if labels_arr is not None and len(labels_arr) == len(coords):
            unique_labels = np.unique(labels_arr)
            for lab in unique_labels:
                idx = (labels_arr == lab)
                ax.scatter(plot_coords[idx, 0], plot_coords[idx, 1], label=str(lab), alpha=0.8)
            ax.legend(title="Category")
        else:
            ax.scatter(plot_coords[:, 0], plot_coords[:, 1], alpha=0.8)

        ax.set_title(f"2D Dimension Reduction ({method.upper()})")
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        fig.tight_layout()

        # Set figure on session.pending_figure (contract requirement)
        session.pending_figure = fig

        coords_list = coords.tolist()
        session.artifacts["reduced_coordinates"] = coords_list

        result_id = session.next_result_id("reduce_dimensions")

        summary = {
            "result_id": result_id,
            "status": "success",
            "method": method,
            "n_components": n_components,
            "n_documents": len(coords),
            "coordinates": coords_list
        }
        if explained_variance_ratio is not None:
            summary["explained_variance_ratio"] = explained_variance_ratio

        session.store_result(
            "reduce_dimensions_tool",
            {
                "method": method,
                "n_components": n_components,
                "random_state": random_state,
                "perplexity": perplexity,
                "n_neighbors": n_neighbors
            },
            summary
        )

        return summary

    @tool
    def binarize_labels_tool(
        target_class: str
    ) -> Dict[str, Any]:
        """Convert multi-class category labels to a one-vs-rest binary indicator (1 for target_class, 0 otherwise).

        Args:
            target_class: Category label to set as target (class 1).

        Returns:
            Dict containing binarization summary (result_id, target_class, n_documents, n_target, n_rest, binary_counts).
        """
        # Primary source of labels: session.labels
        labels = session.labels

        # Fallback to session.dataframe if session.labels is None
        if labels is None and session.dataframe is not None:
            for col in ["category_name", "label", "category"]:
                if col in session.dataframe.columns:
                    labels = session.dataframe[col].values
                    break

        if labels is None:
            return {"status": "error", "message": "No document labels found in session."}

        labels_arr = np.asarray(labels).astype(str)

        if str(target_class) not in labels_arr:
            return {"status": "error", "message": f"Target class '{target_class}' not found in document labels."}

        # Create one-vs-rest binary indicator vector
        binary_vec = (labels_arr == str(target_class)).astype(int)

        n_docs = len(binary_vec)
        n_target = int(np.sum(binary_vec == 1))
        n_rest = int(np.sum(binary_vec == 0))

        # Reject a binary vector containing only one class (all 1s or all 0s)
        if n_target == 0 or n_rest == 0:
            return {"status": "error", "message": "Binary labels vector contains only one class (all 1s or all 0s)."}

        # Validate that label count matches dataframe row count before assigning binary_label
        if session.dataframe is not None:
            if len(binary_vec) != len(session.dataframe):
                return {"status": "error", "message": f"Label count ({len(binary_vec)}) does not match dataframe row count ({len(session.dataframe)})."}
            session.dataframe["binary_label"] = binary_vec

        # Store binary vector in session.artifacts
        session.artifacts["binary_label"] = binary_vec.tolist()
        session.artifacts[f"binary_label_{target_class}"] = binary_vec.tolist()

        result_id = session.next_result_id("binarize_labels")

        summary = {
            "result_id": result_id,
            "status": "success",
            "target_class": target_class,
            "n_documents": n_docs,
            "n_target": n_target,
            "n_rest": n_rest,
            "binary_counts": {
                "1": n_target,
                "0": n_rest
            }
        }

        session.store_result(
            "binarize_labels_tool",
            {"target_class": target_class},
            summary
        )

        return summary

    return [reduce_dimensions_tool, binarize_labels_tool]
