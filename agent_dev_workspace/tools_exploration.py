from typing import Dict, Any, Optional
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.tools import tool

def make_tools(session):
    @tool
    def cosine_similarity_tool(
        doc_id1: int,
        doc_id2: int
    ) -> Dict[str, Any]:
        """Compute cosine similarity between two documents using their document-term matrix vectors.

        Args:
            doc_id1: Index of the first document (0-based).
            doc_id2: Index of the second document (0-based).

        Returns:
            Dict containing doc_id1, doc_id2, similarity score, and result_id.
        """
        if session.feature_matrix is None:
            return {"status": "error", "message": "No feature_matrix found in session. Please run build_dtm_tool first."}

        matrix = session.feature_matrix
        n_docs = matrix.shape[0]

        if doc_id1 < 0 or doc_id1 >= n_docs:
            return {"status": "error", "message": f"doc_id1 ({doc_id1}) is out of bounds for dataset with {n_docs} documents."}

        if doc_id2 < 0 or doc_id2 >= n_docs:
            return {"status": "error", "message": f"doc_id2 ({doc_id2}) is out of bounds for dataset with {n_docs} documents."}

        v1 = matrix[doc_id1]
        v2 = matrix[doc_id2]

        norm1 = sp.linalg.norm(v1) if sp.issparse(v1) else np.linalg.norm(v1)
        norm2 = sp.linalg.norm(v2) if sp.issparse(v2) else np.linalg.norm(v2)

        if norm1 == 0 or norm2 == 0:
            sim = 0.0
        else:
            sim_matrix = cosine_similarity(v1, v2)
            sim = float(sim_matrix[0, 0])
            if np.isnan(sim):
                sim = 0.0

        result_id = session.next_result_id("cosine_similarity")

        summary = {
            "result_id": result_id,
            "status": "success",
            "doc_id1": doc_id1,
            "doc_id2": doc_id2,
            "similarity": float(round(sim, 4))
        }

        session.store_result(
            "cosine_similarity_tool",
            {"doc_id1": doc_id1, "doc_id2": doc_id2},
            summary
        )

        return summary

    @tool
    def feature_correlation_matrix_tool(
        top_k: int = 20
    ) -> Dict[str, Any]:
        """Compute Pearson correlation matrix and build a heatmap over top-K terms (at most 20) by variance.

        Args:
            top_k: Maximum number of top variance terms to include (must be between 1 and 20, default 20).

        Returns:
            Dict containing top terms, correlation matrix, and result_id.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"status": "error", "message": "No feature_matrix found in session. Please run build_dtm_tool first."}

        X = session.feature_matrix
        feature_names = np.array(session.feature_names)
        n_samples, n_features = X.shape

        if n_features == 0 or n_samples < 2:
            return {"status": "error", "message": "Dataset must contain at least 2 documents and non-empty vocabulary."}

        # Cap top_k at 20 as required by contract
        if top_k <= 0 or top_k > 20:
            top_k = 20
        top_k = min(top_k, n_features, 20)

        # Step 1: Compute per-column population variance using sparse-compatible operations (mean_x2 - mean_x^2)
        if sp.issparse(X):
            mean_x = np.asarray(X.mean(axis=0)).ravel()
            mean_x2 = np.asarray(X.power(2).mean(axis=0)).ravel()
            variances = mean_x2 - mean_x**2
        else:
            variances = np.var(X, axis=0)

        # Step 2: Select top_k indices sorted descending by variance
        top_indices = np.argsort(variances)[::-1][:top_k]
        top_terms = feature_names[top_indices].tolist()

        # Densify ONLY the selected top_k columns (memory-conscious approach)
        if sp.issparse(X):
            top_matrix = X[:, top_indices].toarray()
        else:
            top_matrix = np.asarray(X)[:, top_indices]

        # Handle 1-feature edge case with variance check
        if top_k == 1:
            var_single = np.var(top_matrix[:, 0]) if top_matrix.shape[1] > 0 else 0.0
            corr_matrix = np.array([[1.0 if var_single > 0 else 0.0]])
        else:
            # Step 3: Compute Pearson correlation matrix across top terms
            corr_matrix = np.corrcoef(top_matrix, rowvar=False)
            corr_matrix = np.nan_to_num(corr_matrix, nan=0.0, posinf=0.0, neginf=0.0)

        # Step 4: Build heatmap plot and set on session.pending_figure
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            corr_matrix,
            xticklabels=top_terms,
            yticklabels=top_terms,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            ax=ax
        )
        ax.set_title(f"Feature Correlation Matrix (Top {len(top_terms)} by Variance)")
        fig.tight_layout()

        session.pending_figure = fig

        corr_matrix_list = [[float(round(val, 4)) for val in row] for row in corr_matrix]

        result_id = session.next_result_id("feature_correlation_matrix")

        summary = {
            "result_id": result_id,
            "status": "success",
            "n_terms": len(top_terms),
            "terms": top_terms,
            "correlation_matrix": corr_matrix_list
        }

        session.store_result(
            "feature_correlation_matrix_tool",
            {"top_k": top_k},
            summary
        )

        return summary

    return [cosine_similarity_tool, feature_correlation_matrix_tool]
