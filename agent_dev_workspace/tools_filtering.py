from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import spearmanr
from langchain_core.tools import tool

def make_tools(session):
    @tool
    def variance_filter_tool(
        threshold: float = 0.0
    ) -> Dict[str, Any]:
        """Filter terms in the Document-Term Matrix (DTM) based on variance across documents.

        Args:
            threshold: Minimum variance required to keep a term (default 0.0).

        Returns:
            Dict containing filtering summary (result_id, n_total, n_kept, n_removed, kept_terms, removed_terms).
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"status": "error", "message": "No feature_matrix found in session. Please run build_dtm_tool first."}

        # Convert sparse feature matrix to dense array for term statistics
        matrix = session.feature_matrix.toarray() if hasattr(session.feature_matrix, "toarray") else np.asarray(session.feature_matrix)
        feature_names = session.feature_names

        # Compute population variance for each column (term)
        variances = np.var(matrix, axis=0)

        # Build full_report DataFrame with term as 1st column and variance as 2nd column
        report_df = pd.DataFrame({
            "term": feature_names,
            "variance": variances
        })

        # Sort descending by variance for visualization compatibility
        report_df = report_df.sort_values(by="variance", ascending=False).reset_index(drop=True)

        # Apply variance threshold filter
        kept_mask = report_df["variance"] >= threshold
        kept_df = report_df[kept_mask]
        removed_df = report_df[~kept_mask]

        kept_terms = kept_df["term"].tolist()
        removed_terms = removed_df["term"].tolist()

        result_id = session.next_result_id("variance_filter")

        summary = {
            "result_id": result_id,
            "status": "success",
            "threshold": threshold,
            "n_total": len(feature_names),
            "n_kept": len(kept_terms),
            "n_removed": len(removed_terms),
            "kept_terms": kept_terms,
            "removed_terms": removed_terms
        }

        # Store result with full_report DataFrame as required by spec
        session.store_result(
            "variance_filter_tool",
            {"threshold": threshold},
            summary,
            full_report=report_df
        )

        return summary

    @tool
    def pearson_filter_tool(
        target_class: str,
        threshold: float = 0.0
    ) -> Dict[str, Any]:
        """Filter terms based on Pearson correlation between term counts and a target class indicator.

        Args:
            target_class: Target category label to calculate correlation against (one-vs-rest).
            threshold: Minimum correlation value threshold (default 0.0).

        Returns:
            Dict containing filtering summary (result_id, target_class, threshold, n_total, n_kept, n_removed, kept_terms, removed_terms).
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"status": "error", "message": "No feature_matrix found in session. Please run build_dtm_tool first."}

        if session.labels is None:
            return {"status": "error", "message": "No document labels found in session."}

        labels = np.asarray(session.labels)

        if target_class not in labels:
            return {"status": "error", "message": f"Target class '{target_class}' not found in document labels."}

        # Construct one-vs-rest binary target vector y
        y = (labels == target_class).astype(float)
        n_samples = len(y)

        # Check if target vector has zero variance
        std_y = np.std(y)
        if std_y == 0 or n_samples == 0:
            return {"status": "error", "message": f"Target vector for class '{target_class}' has zero variance."}

        # Sparse-compatible vectorized Pearson correlation computation
        X = session.feature_matrix
        if not sp.issparse(X):
            X = sp.csr_matrix(X)
        else:
            X = X.tocsr()

        # Means and variances across columns (axis=0)
        mean_x = np.asarray(X.mean(axis=0)).ravel()
        mean_x2 = np.asarray(X.power(2).mean(axis=0)).ravel()
        var_x = mean_x2 - mean_x**2
        # Use clipped variance to prevent tiny negative floats from precision issues
        std_x = np.sqrt(np.maximum(var_x, 0.0))

        # Dot product X.T @ y for numerator sum( (x - mean_x)*(y - mean_y) ) = sum(x*y) - N * mean_x * mean_y
        mean_y = np.mean(y)
        dot_xy = X.T.dot(y)
        cov_xy = (dot_xy / n_samples) - (mean_x * mean_y)

        # Vectorized correlation coefficient calculation
        # Handle zero variance columns (std_x == 0) safely
        with np.errstate(divide='ignore', invalid='ignore'):
            pearson_rs = cov_xy / (std_x * std_y)
            pearson_rs = np.nan_to_num(pearson_rs, nan=0.0, posinf=0.0, neginf=0.0)

        feature_names = session.feature_names

        # Build full_report DataFrame preserving full floating-point precision
        report_df = pd.DataFrame({
            "term": feature_names,
            "pearson_r": pearson_rs
        })

        # Sort descending by signed pearson_r as required by contract
        report_df = report_df.sort_values(by="pearson_r", ascending=False).reset_index(drop=True)

        # Filter by magnitude abs(pearson_r) >= threshold using full floating-point precision
        kept_mask = report_df["pearson_r"].abs() >= threshold
        kept_df = report_df[kept_mask]
        removed_df = report_df[~kept_mask]

        kept_terms = kept_df["term"].tolist()
        removed_terms = removed_df["term"].tolist()

        result_id = session.next_result_id("pearson_filter")

        summary = {
            "result_id": result_id,
            "status": "success",
            "target_class": target_class,
            "threshold": threshold,
            "n_total": len(feature_names),
            "n_kept": len(kept_terms),
            "n_removed": len(removed_terms),
            "kept_terms": kept_terms,
            "removed_terms": removed_terms
        }

        # Store result with full_report DataFrame as required by spec
        session.store_result(
            "pearson_filter_tool",
            {"target_class": target_class, "threshold": threshold},
            summary,
            full_report=report_df
        )

        return summary

    @tool
    def spearman_filter_tool(
        target_class: str,
        threshold: float = 0.0
    ) -> Dict[str, Any]:
        """Filter terms based on Spearman rank correlation between term counts and a target class indicator.

        Args:
            target_class: Target category label to calculate correlation against (one-vs-rest).
            threshold: Minimum correlation value threshold (default 0.0).

        Returns:
            Dict containing filtering summary (result_id, target_class, threshold, n_total, n_kept, n_removed, kept_terms, removed_terms).
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"status": "error", "message": "No feature_matrix found in session. Please run build_dtm_tool first."}

        if session.labels is None:
            return {"status": "error", "message": "No document labels found in session."}

        labels = np.asarray(session.labels)

        if target_class not in labels:
            return {"status": "error", "message": f"Target class '{target_class}' not found in document labels."}

        # Construct one-vs-rest binary target vector y
        y = (labels == target_class).astype(float)

        if np.std(y) == 0:
            return {"status": "error", "message": f"Target vector for class '{target_class}' has zero variance."}

        X = session.feature_matrix
        if not sp.issparse(X):
            X = sp.csc_matrix(X)
        else:
            X = X.tocsc()

        n_samples, n_features = X.shape
        spearman_rs = np.zeros(n_features, dtype=float)

        # Process columns individually to avoid dense memory allocation across entire matrix
        for j in range(n_features):
            col_data = X.getcol(j).toarray().ravel()
            if np.std(col_data) == 0:
                spearman_rs[j] = 0.0
            else:
                # Calculate Spearman rank correlation using scipy.stats.spearmanr
                res = spearmanr(col_data, y)
                rho = float(res.statistic) if hasattr(res, 'statistic') else float(res[0])
                if np.isnan(rho):
                    rho = 0.0
                spearman_rs[j] = rho

        feature_names = session.feature_names

        # Build full_report DataFrame with term as 1st column and spearman_r as 2nd column
        report_df = pd.DataFrame({
            "term": feature_names,
            "spearman_r": spearman_rs
        })

        # Sort descending by signed spearman_r as required by contract
        report_df = report_df.sort_values(by="spearman_r", ascending=False).reset_index(drop=True)

        # Filter by magnitude abs(spearman_r) >= threshold preserving full floating-point precision
        kept_mask = report_df["spearman_r"].abs() >= threshold
        kept_df = report_df[kept_mask]
        removed_df = report_df[~kept_mask]

        kept_terms = kept_df["term"].tolist()
        removed_terms = removed_df["term"].tolist()

        result_id = session.next_result_id("spearman_filter")

        summary = {
            "result_id": result_id,
            "status": "success",
            "target_class": target_class,
            "threshold": threshold,
            "n_total": len(feature_names),
            "n_kept": len(kept_terms),
            "n_removed": len(removed_terms),
            "kept_terms": kept_terms,
            "removed_terms": removed_terms
        }

        # Store result with full_report DataFrame as required by spec
        session.store_result(
            "spearman_filter_tool",
            {"target_class": target_class, "threshold": threshold},
            summary,
            full_report=report_df
        )

        return summary

    return [variance_filter_tool, pearson_filter_tool, spearman_filter_tool]
