from typing import Dict, Any, List, Optional
import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pydantic import StrictInt
from langchain_core.tools import tool

def make_tools(session):
    @tool
    def list_files_tool(
        subdirectory: str = "."
    ) -> Dict[str, Any]:
        """List files and subdirectories in a relative directory within the workspace or repository.

        Args:
            subdirectory: Relative path to list (default '.').

        Returns:
            Dict containing subdirectory path, directories list, files list, and result_id.
        """
        # Reject absolute paths
        if os.path.isabs(subdirectory):
            return {"status": "error", "message": "Only relative paths within the workspace are allowed."}

        # Resolve repo root directory (current working directory)
        repo_root = Path.cwd().resolve()
        target_path = (repo_root / subdirectory).resolve()

        # Security check: ensure target_path is within repo_root
        try:
            target_path.relative_to(repo_root)
        except ValueError:
            return {"status": "error", "message": "Access denied: Path is outside the repository workspace."}

        if not target_path.exists():
            return {"status": "error", "message": f"Directory '{subdirectory}' does not exist."}

        if not target_path.is_dir():
            return {"status": "error", "message": f"Path '{subdirectory}' is a file, not a directory."}

        directories = []
        files = []

        for entry in os.scandir(target_path):
            if entry.is_dir():
                directories.append(entry.name)
            elif entry.is_file():
                files.append(entry.name)

        directories.sort()
        files.sort()

        result_id = session.next_result_id("list_files")

        summary = {
            "result_id": result_id,
            "status": "success",
            "subdirectory": subdirectory,
            "directories": directories,
            "files": files
        }

        session.store_result(
            "list_files_tool",
            {"subdirectory": subdirectory},
            summary
        )

        return summary

    @tool
    def check_missing_tool(
        column: str = "text"
    ) -> Dict[str, Any]:
        """Check for missing or empty text values in a specified dataframe column.

        Args:
            column: Name of the column to inspect (default 'text').

        Returns:
            Dict containing column name, total_rows, n_missing, missing_indices, and result_id.
        """
        if session.dataframe is None:
            return {"status": "error", "message": "No dataframe loaded in session."}

        df = session.dataframe

        if column not in df.columns:
            return {"status": "error", "message": f"Column '{column}' not found in dataframe."}

        col_series = df[column]

        # Identify missing or empty text values (NaN, None, empty string, or whitespace-only)
        is_missing = col_series.isna() | (col_series.astype(str).str.strip() == "")

        # Use positional 0-based row indices regardless of DataFrame index labels
        missing_indices = np.flatnonzero(is_missing.to_numpy()).tolist()
        missing_indices = [int(i) for i in missing_indices]

        result_id = session.next_result_id("check_missing")

        summary = {
            "result_id": result_id,
            "status": "success",
            "column": column,
            "total_rows": len(df),
            "n_missing": len(missing_indices),
            "missing_indices": missing_indices
        }

        session.store_result(
            "check_missing_tool",
            {"column": column},
            summary
        )

        return summary

    @tool
    def check_duplicates_tool(
        column: str = "text",
        drop: bool = False
    ) -> Dict[str, Any]:
        """Check for duplicate documents in a dataframe column, and optionally drop all copies of duplicate rows.

        Args:
            column: Name of the column to check for duplicates (default 'text').
            drop: Whether to drop duplicate documents in-place using keep=False (default False).

        Returns:
            Dict containing column name, total_rows_before, n_duplicates, duplicate_indices, dropped, total_rows_after, and result_id.
        """
        if session.dataframe is None:
            return {"status": "error", "message": "No dataframe loaded in session."}

        df = session.dataframe

        if column not in df.columns:
            return {"status": "error", "message": f"Column '{column}' not found in dataframe."}

        n_before = len(df)

        # Detect duplicate rows beyond first occurrence for count and index reporting
        dup_mask_first = df.duplicated(subset=[column], keep="first")
        n_duplicates = int(dup_mask_first.sum())
        duplicate_indices = [int(i) for i in np.flatnonzero(dup_mask_first.to_numpy())]

        if drop and n_duplicates > 0:
            # Check row count compatibility for existing session.labels
            if session.labels is not None and len(session.labels) != n_before:
                return {"status": "error", "message": f"Existing session.labels length ({len(session.labels)}) does not match dataframe row count ({n_before})."}

            # Check row count compatibility for existing feature_matrix
            if session.feature_matrix is not None and session.feature_matrix.shape[0] != n_before:
                return {"status": "error", "message": f"Existing feature_matrix row count ({session.feature_matrix.shape[0]}) does not match dataframe row count ({n_before})."}

            # Compute positional keep_mask to drop EVERY copy of duplicated rows (keep=False)
            dup_mask_all = df.duplicated(subset=[column], keep=False)
            keep_mask = ~dup_mask_all.to_numpy()

            # Update session.dataframe
            session.dataframe = df[keep_mask].reset_index(drop=True)

            # Preserve row alignment for session.labels and session.categories
            if session.labels is not None:
                filtered_labels = np.asarray(session.labels)[keep_mask]
                session.set_labels(filtered_labels)

            # Preserve row alignment for session.feature_matrix if present
            if session.feature_matrix is not None:
                session.feature_matrix = session.feature_matrix[keep_mask]

        n_after = len(session.dataframe)

        result_id = session.next_result_id("check_duplicates")

        summary = {
            "result_id": result_id,
            "status": "success",
            "column": column,
            "total_rows_before": n_before,
            "n_duplicates": n_duplicates,
            "duplicate_indices": duplicate_indices,
            "dropped": drop,
            "total_rows_after": n_after
        }

        session.store_result(
            "check_duplicates_tool",
            {"column": column, "drop": drop},
            summary
        )

        return summary

    @tool
    def tokenize_tool(
        column: str = "text"
    ) -> Dict[str, Any]:
        """Tokenize each document's text in the dataframe into unigrams using NLTK word_tokenize.

        Args:
            column: Name of the column containing text to tokenize (default 'text').

        Returns:
            Dict containing n_documents, column, sample_tokens, and result_id.
        """
        if session.dataframe is None:
            return {"status": "error", "message": "No dataframe loaded in session."}

        df = session.dataframe

        if column not in df.columns:
            return {"status": "error", "message": f"Column '{column}' not found in dataframe."}

        # Verify NLTK availability
        try:
            import nltk
            from nltk.tokenize import word_tokenize
        except ImportError:
            return {"status": "error", "message": "NLTK library is not installed in the environment."}

        # Tokenize rows
        unigrams_list = []
        try:
            for val in df[column]:
                if pd.isna(val) or str(val).strip() == "":
                    unigrams_list.append([])
                else:
                    text_str = str(val).lower()
                    tokens = word_tokenize(text_str)
                    unigrams_list.append(tokens)
        except LookupError as err:
            return {"status": "error", "message": f"Required NLTK tokenizer resource missing: {err}"}
        except Exception as err:
            return {"status": "error", "message": f"Tokenization failed: {err}"}

        # Store unigrams in session.dataframe and session.artifacts
        session.dataframe["unigrams"] = unigrams_list
        session.artifacts["unigrams"] = unigrams_list

        result_id = session.next_result_id("tokenize")

        summary = {
            "result_id": result_id,
            "status": "success",
            "column": column,
            "n_documents": len(unigrams_list),
            "sample_tokens": unigrams_list[:3]
        }

        session.store_result(
            "tokenize_tool",
            {"column": column},
            summary
        )

        return summary

    @tool
    def inspect_data_tool(
        n_rows: StrictInt = 5,
        columns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Inspect the first n_rows and column metadata of the working dataframe.

        Args:
            n_rows: Number of head rows to inspect (default 5, must be positive).
            columns: Optional list of specific column names to inspect.

        Returns:
            Dict containing total_rows, total_columns, n_rows_returned, columns, dtypes, sample_data, and result_id.
        """
        if session.dataframe is None:
            return {"status": "error", "message": "No dataframe loaded in session."}

        if n_rows <= 0:
            return {"status": "error", "message": "n_rows must be a positive integer."}

        df = session.dataframe

        # Validate requested columns
        if columns is not None:
            missing_cols = [col for col in columns if col not in df.columns]
            if missing_cols:
                return {"status": "error", "message": f"Requested column(s) not found in dataframe: {missing_cols}"}
            inspect_df = df[columns].head(n_rows)
        else:
            inspect_df = df.head(n_rows)

        # Robust, pandas-native JSON-serializable records conversion
        clean_records = json.loads(inspect_df.to_json(orient="records", date_format="iso"))

        dtypes_dict = {col: str(dtype) for col, dtype in inspect_df.dtypes.items()}

        result_id = session.next_result_id("inspect_data")

        summary = {
            "result_id": result_id,
            "status": "success",
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "n_rows_returned": len(inspect_df),
            "columns": list(inspect_df.columns),
            "dtypes": dtypes_dict,
            "sample_data": clean_records
        }

        session.store_result(
            "inspect_data_tool",
            {"n_rows": n_rows, "columns": columns},
            summary
        )

        return summary

    @tool
    def sample_data_tool(
        n: StrictInt = 5,
        random_state: Optional[StrictInt] = None
    ) -> Dict[str, Any]:
        """Subsample the working dataframe using pandas DataFrame.sample on positional indices.

        Args:
            n: Number of items to sample (default 5, must be positive integer).
            random_state: Optional seed for reproducible random sampling.

        Returns:
            Dict containing n_requested, n_returned, total_rows, sampled_indices, sample_data, and result_id.
        """
        if session.dataframe is None:
            return {"status": "error", "message": "No dataframe loaded in session."}

        if n <= 0:
            return {"status": "error", "message": "n must be a positive integer."}

        df = session.dataframe
        n_total = len(df)

        if n_total == 0:
            sampled_indices = []
            sample_df = df.iloc[:0]
        else:
            actual_n = min(int(n), n_total)
            # Positional sampling using 0-based integer positions (robust against duplicate index labels)
            sampled_pos = pd.Series(np.arange(n_total)).sample(n=actual_n, random_state=random_state)
            sampled_indices = [int(i) for i in sampled_pos.tolist()]
            sample_df = df.iloc[sampled_indices]

        clean_records = json.loads(sample_df.to_json(orient="records", date_format="iso"))

        result_id = session.next_result_id("sample_data")

        summary = {
            "result_id": result_id,
            "status": "success",
            "n_requested": n,
            "n_returned": len(sample_df),
            "total_rows": n_total,
            "random_state": random_state,
            "sampled_indices": sampled_indices,
            "sample_data": clean_records
        }

        session.store_result(
            "sample_data_tool",
            {"n": n, "random_state": random_state},
            summary
        )

        return summary

    @tool
    def describe_data_tool(
        text_column: str = "text",
        category_column: Optional[str] = None
    ) -> Dict[str, Any]:
        """Compute text length statistics and build a box plot grouped by category.

        Args:
            text_column: Column containing document text (default 'text').
            category_column: Optional category column for grouped stats and box plot.

        Returns:
            Dict containing overall_stats, by_category stats, and result_id.
        """
        if session.dataframe is None:
            return {"status": "error", "message": "No dataframe loaded in session."}

        df = session.dataframe

        if text_column not in df.columns:
            return {"status": "error", "message": f"Text column '{text_column}' not found in dataframe."}

        # Explicit category_column validation
        if category_column is not None and category_column not in df.columns:
            return {"status": "error", "message": f"Requested category column '{category_column}' not found in dataframe."}

        # Resolve category column if category_column is None
        cat_col = category_column
        if cat_col is None:
            for col in ["category_name", "label", "category"]:
                if col in df.columns:
                    cat_col = col
                    break

        # Clear session.pending_figure for empty dataframe
        if len(df) == 0:
            session.pending_figure = None
            result_id = session.next_result_id("describe_data")
            empty_stats = {"count": 0.0, "mean": None, "std": None, "min": None, "25%": None, "50%": None, "75%": None, "max": None}
            summary = {
                "result_id": result_id,
                "status": "success",
                "text_column": text_column,
                "category_column": cat_col,
                "total_documents": 0,
                "overall_stats": empty_stats,
                "by_category": {},
                "note": "Empty dataframe provided; no box plot generated."
            }
            session.store_result("describe_data_tool", {"text_column": text_column, "category_column": category_column}, summary)
            return summary

        # Calculate text length (character count) per document
        text_series = df[text_column].fillna("").astype(str)
        text_lengths = text_series.apply(len)

        # Overall .describe() statistics over text_length
        overall_desc = text_lengths.describe()
        overall_stats = {}
        for k, v in overall_desc.to_dict().items():
            overall_stats[k] = float(round(v, 4)) if not pd.isna(v) else None

        by_category = {}
        if cat_col is not None and cat_col in df.columns:
            # Preserve missing category values and exclude them from per-category stats and grouped box plot
            valid_mask = df[cat_col].notna()
            
            if valid_mask.any():
                # Use unique internal column names to prevent name collision if category_column == 'text_length'
                plot_df = pd.DataFrame({
                    "__text_length__": text_lengths[valid_mask],
                    "__category__": df.loc[valid_mask, cat_col].astype(str)
                })

                # Grouped .describe() statistics per category
                grouped_desc = plot_df.groupby("__category__")["__text_length__"].describe()
                for cat_name, row in grouped_desc.iterrows():
                    cat_dict = {}
                    for k, v in row.to_dict().items():
                        cat_dict[k] = float(round(v, 4)) if not pd.isna(v) else None
                    by_category[str(cat_name)] = cat_dict

                # Build Seaborn box plot using internal column names, setting clean display labels
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.boxplot(x="__category__", y="__text_length__", data=plot_df, ax=ax)
                ax.set_title("Document Text Length by Category")
                ax.set_xlabel(str(cat_col))
                ax.set_ylabel("Text Length (characters)")
                fig.tight_layout()

                session.pending_figure = fig
            else:
                # Single box plot if all categories are NA
                fig, ax = plt.subplots(figsize=(6, 6))
                sns.boxplot(y=text_lengths, ax=ax)
                ax.set_title("Document Text Length Distribution")
                ax.set_ylabel("Text Length (characters)")
                fig.tight_layout()

                session.pending_figure = fig
        else:
            # Single box plot if no category column exists
            fig, ax = plt.subplots(figsize=(6, 6))
            sns.boxplot(y=text_lengths, ax=ax)
            ax.set_title("Document Text Length Distribution")
            ax.set_ylabel("Text Length (characters)")
            fig.tight_layout()

            session.pending_figure = fig

        result_id = session.next_result_id("describe_data")

        summary = {
            "result_id": result_id,
            "status": "success",
            "text_column": text_column,
            "category_column": cat_col,
            "total_documents": len(df),
            "overall_stats": overall_stats,
            "by_category": by_category
        }

        session.store_result(
            "describe_data_tool",
            {"text_column": text_column, "category_column": category_column},
            summary
        )

        return summary

    return [list_files_tool, check_missing_tool, check_duplicates_tool, tokenize_tool, inspect_data_tool, sample_data_tool, describe_data_tool]
