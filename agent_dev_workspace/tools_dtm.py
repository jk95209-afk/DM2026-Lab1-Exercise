from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from pydantic import StrictInt
from langchain_core.tools import tool

def make_tools(session):
    @tool
    def build_dtm_tool(
        max_features: Optional[int] = None,
        min_df: int = 1,
        max_df: float = 1.0,
        ngram_range: List[int] = [1, 1],
        stop_words: Optional[str] = None
    ) -> Dict[str, Any]:
        """Build a Document-Term Matrix (DTM) from the current session dataframe using CountVectorizer.

        Args:
            max_features: Maximum number of terms to keep in the vocabulary.
            min_df: Minimum document frequency threshold (count or fraction).
            max_df: Maximum document frequency threshold (count or fraction).
            ngram_range: Lower and upper boundary of range of n-values for n-grams, e.g., [1, 1].
            stop_words: Language for stop words removal (e.g. 'english') or None.

        Returns:
            Dict containing summary statistics including matrix shape, sparsity, and result_id.
        """
        if session.dataframe is None or "text" not in session.dataframe.columns:
            return {"status": "error", "message": "No valid dataset loaded in session with a 'text' column."}

        # Ensure text column missing values are replaced with empty strings
        texts = session.dataframe["text"].fillna("").astype(str).tolist()

        # Convert list parameter to tuple for CountVectorizer
        ngram_tuple = tuple(ngram_range) if ngram_range and len(ngram_range) == 2 else (1, 1)

        vectorizer = CountVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            ngram_range=ngram_tuple,
            stop_words=stop_words
        )

        dtm = vectorizer.fit_transform(texts)
        feature_names = list(vectorizer.get_feature_names_out())

        # Update session state
        session.feature_matrix = dtm
        session.feature_names = feature_names
        session.artifacts["count_vectorizer"] = vectorizer

        if "category_name" in session.dataframe.columns:
            session.set_labels(session.dataframe["category_name"].values)
        elif "category" in session.dataframe.columns:
            session.set_labels(session.dataframe["category"].values)

        # Calculate sparsity according to spec
        n_rows, n_cols = dtm.shape
        total_elements = n_rows * n_cols
        non_zero = int(dtm.nnz)
        sparsity_pct = 100.0 * (1.0 - non_zero / total_elements) if total_elements > 0 else 0.0

        result_id = session.next_result_id("build_dtm")

        summary = {
            "result_id": result_id,
            "status": "success",
            "n_documents": n_rows,
            "n_terms": n_cols,
            "non_zero": non_zero,
            "total_elements": total_elements,
            "sparsity_pct": float(round(sparsity_pct, 4)),
            "feature_names_sample": feature_names[:10]
        }

        session.store_result("build_dtm_tool", {"max_features": max_features, "min_df": min_df, "max_df": max_df, "ngram_range": ngram_range, "stop_words": stop_words}, summary)

        return summary

    @tool
    def term_frequency_tool(
        top_k: Optional[StrictInt] = None
    ) -> Dict[str, Any]:
        """Aggregate term frequencies across all documents in the Document-Term Matrix (DTM).

        Args:
            top_k: Optional maximum number of top frequent terms to return in summary (must be positive integer).

        Returns:
            Dict containing total_terms, total_word_count, term_frequencies, and result_id.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"status": "error", "message": "No feature_matrix found in session. Please run build_dtm_tool first."}

        if top_k is not None and top_k <= 0:
            return {"status": "error", "message": "top_k must be a positive integer."}

        X = session.feature_matrix
        feature_names = session.feature_names

        # Compute column sums (total term frequency across corpus) using sparse-compatible operations
        frequencies = np.asarray(X.sum(axis=0)).ravel()

        report_df = pd.DataFrame({
            "term": feature_names,
            "frequency": frequencies
        })

        # Sort descending by frequency, then ascending by term for deterministic tie breaking
        report_df = report_df.sort_values(by=["frequency", "term"], ascending=[False, True]).reset_index(drop=True)

        if top_k is not None:
            summary_df = report_df.head(top_k)
        else:
            summary_df = report_df

        freq_dict = dict(zip(summary_df["term"], [int(x) for x in summary_df["frequency"]]))

        result_id = session.next_result_id("term_frequency")

        summary = {
            "result_id": result_id,
            "status": "success",
            "total_terms": len(feature_names),
            "total_word_count": int(np.sum(frequencies)),
            "term_frequencies": freq_dict
        }

        session.store_result(
            "term_frequency_tool",
            {"top_k": top_k},
            summary,
            full_report=report_df
        )

        return summary

    @tool
    def dtm_heatmap_tool(
        n_terms: StrictInt = 20,
        n_documents: StrictInt = 20
    ) -> Dict[str, Any]:
        """Build and plot a raw heatmap slice of the Document-Term Matrix (DTM).

        Args:
            n_terms: Number of terms (columns) to include in vocabulary order (default 20).
            n_documents: Number of documents (rows) to include in document order (default 20).

        Returns:
            Dict containing n_documents, n_terms, terms list, matrix_slice, and result_id.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"status": "error", "message": "No feature_matrix found in session. Please run build_dtm_tool first."}

        if n_terms <= 0 or n_documents <= 0:
            return {"status": "error", "message": "n_terms and n_documents must be positive integers."}

        X = session.feature_matrix
        feature_names = session.feature_names
        total_docs, total_terms = X.shape

        if total_docs == 0 or total_terms == 0:
            return {"status": "error", "message": "Document-Term Matrix is empty (zero documents or zero terms)."}

        actual_docs = min(int(n_documents), total_docs)
        actual_terms = min(int(n_terms), total_terms)

        sliced_sparse = X[:actual_docs, :actual_terms]
        sliced_dense = sliced_sparse.toarray() if sp.issparse(sliced_sparse) else np.asarray(sliced_sparse)

        sliced_term_names = feature_names[:actual_terms]
        doc_labels = [f"Doc {i}" for i in range(actual_docs)]

        # Dynamic annotation formatting depending on matrix data type
        fmt = "d" if np.issubdtype(sliced_dense.dtype, np.integer) else "g"

        # Build heatmap figure
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            sliced_dense,
            xticklabels=sliced_term_names,
            yticklabels=doc_labels,
            annot=True,
            fmt=fmt,
            cmap="YlGnBu",
            ax=ax
        )
        ax.set_title(f"DTM Heatmap Slice ({actual_docs} Docs x {actual_terms} Terms)")
        fig.tight_layout()

        session.pending_figure = fig

        matrix_slice_list = [[int(val) if np.issubdtype(sliced_dense.dtype, np.integer) else float(val) for val in row] for row in sliced_dense]

        result_id = session.next_result_id("dtm_heatmap")

        summary = {
            "result_id": result_id,
            "status": "success",
            "n_documents": actual_docs,
            "n_terms": actual_terms,
            "terms": sliced_term_names,
            "matrix_slice": matrix_slice_list
        }

        session.store_result(
            "dtm_heatmap_tool",
            {"n_terms": n_terms, "n_documents": n_documents},
            summary
        )

        return summary

    return [build_dtm_tool, term_frequency_tool, dtm_heatmap_tool]
