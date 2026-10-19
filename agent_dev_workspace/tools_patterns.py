import tempfile
import os
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from pydantic import StrictInt
from langchain_core.tools import tool

# PAMI Imports
try:
    from PAMI.frequentPattern.basic import FPGrowth
    from PAMI.frequentPattern.topk import FAE
    from PAMI.frequentPattern.maximal import MaxFPGrowth
    PAMI_AVAILABLE = True
except ImportError:
    PAMI_AVAILABLE = False

def make_tools(session):
    @tool
    def mine_patterns_tool(
        category_name: str,
        filtering_method: str = "tfidf",
        algorithm: str = "fpgrowth",
        min_sup: Optional[StrictInt] = None,
        k: Optional[StrictInt] = None
    ) -> Dict[str, Any]:
        """Mine frequent patterns for a specific category using PAMI algorithms.

        Args:
            category_name: Target category label to filter and mine.
            filtering_method: Per-category vocabulary filter ('variance', 'tfidf', or 'term_frequency').
            algorithm: PAMI mining algorithm ('fpgrowth', 'topk', or 'maxfpgrowth').
            min_sup: Minimum support count threshold (required for 'fpgrowth' and 'maxfpgrowth').
            k: Number of top patterns to return (required for 'topk').

        Returns:
            Dict containing mined patterns, support counts, and result_id.
        """
        if not PAMI_AVAILABLE:
            return {"status": "error", "message": "PAMI library is not installed in the environment."}

        if session.dataframe is None:
            return {"status": "error", "message": "No dataframe loaded in session."}

        df = session.dataframe

        # Detect category column in session.dataframe
        cat_col = None
        for col in ["category_name", "label", "category"]:
            if col in df.columns:
                cat_col = col
                break

        if cat_col is None:
            return {"status": "error", "message": "No category column found in dataframe."}

        # Filter dataframe to target category
        category_df = df[df[cat_col].astype(str) == str(category_name)]
        if category_df.empty:
            return {"status": "error", "message": f"Category '{category_name}' not found or has no documents."}

        filtering_method = filtering_method.lower()
        algorithm = algorithm.lower()

        if filtering_method not in ["variance", "tfidf", "term_frequency"]:
            return {"status": "error", "message": f"Invalid filtering_method '{filtering_method}'."}

        if algorithm not in ["fpgrowth", "topk", "maxfpgrowth"]:
            return {"status": "error", "message": f"Invalid algorithm '{algorithm}'."}

        # Positive integer validation for min_sup and k parameters
        if algorithm in ["fpgrowth", "maxfpgrowth"]:
            if min_sup is None or min_sup <= 0:
                return {"status": "error", "message": f"Algorithm '{algorithm}' requires 'min_sup' to be a positive integer."}

        if algorithm == "topk":
            if k is None or k <= 0:
                return {"status": "error", "message": "Algorithm 'topk' requires 'k' to be a positive integer."}

        texts = category_df["text"].fillna("").astype(str).tolist()

        # Step 1: Fit category-local CountVectorizer with English stopwords
        try:
            vectorizer = CountVectorizer(stop_words="english")
            dtm_sparse = vectorizer.fit_transform(texts)
            feature_names = np.array(vectorizer.get_feature_names_out())
        except ValueError:
            # Vocabulary empty (e.g., all words dropped or empty documents)
            feature_names = np.array([])
            dtm_sparse = sp.csr_matrix((len(texts), 0))

        artifact_key = f"patterns_{category_name}_{filtering_method}_{algorithm}"

        if len(feature_names) == 0:
            result_id = session.next_result_id("mine_patterns")
            summary = {
                "result_id": result_id,
                "status": "success",
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "n_patterns": 0,
                "patterns": [],
                "note": "no terms survived stopword removal"
            }
            session.artifacts[artifact_key] = []
            session.store_result("mine_patterns_tool", {"category_name": category_name, "filtering_method": filtering_method, "algorithm": algorithm, "min_sup": min_sup, "k": k}, summary)
            return summary

        # Step 2: Calculate feature scores using sparse matrix operations
        if filtering_method == "variance":
            # Column variance: mean(x^2) - (mean(x))^2
            mean_x = np.asarray(dtm_sparse.mean(axis=0)).ravel()
            mean_x2 = np.asarray(dtm_sparse.power(2).mean(axis=0)).ravel()
            scores = mean_x2 - mean_x**2
            p5 = np.percentile(scores, 5)
            p95 = np.percentile(scores, 95)
            mask = (scores > p5) & (scores < p95)

        elif filtering_method == "term_frequency":
            scores = np.asarray(dtm_sparse.sum(axis=0)).ravel()
            p5 = np.percentile(scores, 5)
            p95 = np.percentile(scores, 95)
            mask = (scores > p5) & (scores < p95)

        elif filtering_method == "tfidf":
            tfidf_transformer = TfidfTransformer()
            tfidf_sparse = tfidf_transformer.fit_transform(dtm_sparse)
            scores = np.asarray(tfidf_sparse.mean(axis=0)).ravel()
            p20 = np.percentile(scores, 20)
            mask = scores >= p20

        surviving_features = feature_names[mask]

        if len(surviving_features) == 0:
            result_id = session.next_result_id("mine_patterns")
            summary = {
                "result_id": result_id,
                "status": "success",
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "n_patterns": 0,
                "patterns": [],
                "note": "no terms survived filtering"
            }
            session.artifacts[artifact_key] = []
            session.store_result("mine_patterns_tool", {"category_name": category_name, "filtering_method": filtering_method, "algorithm": algorithm, "min_sup": min_sup, "k": k}, summary)
            return summary

        # Step 3: Construct binary DataFrame for surviving terms
        surviving_sparse = dtm_sparse[:, mask]
        binary_matrix = (surviving_sparse >= 1).astype(int).toarray()
        df_binary = pd.DataFrame(binary_matrix, columns=surviving_features)

        # Filter out rows with zero items
        df_binary = df_binary[df_binary.sum(axis=1) > 0]

        if df_binary.empty:
            result_id = session.next_result_id("mine_patterns")
            summary = {
                "result_id": result_id,
                "status": "success",
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "n_patterns": 0,
                "patterns": [],
                "note": "no non-empty transactions for surviving terms"
            }
            session.artifacts[artifact_key] = []
            session.store_result("mine_patterns_tool", {"category_name": category_name, "filtering_method": filtering_method, "algorithm": algorithm, "min_sup": min_sup, "k": k}, summary)
            return summary

        # Step 4: Convert to PAMI Transactional DB using explicit UTF-8 tab-separated writer
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "pami_db.txt")
            matrix = df_binary.values
            cols = df_binary.columns.to_numpy()
            with open(db_path, "w", encoding="utf-8") as f:
                for row in matrix:
                    active_indices = np.flatnonzero(row)
                    if active_indices.size > 0:
                        f.write("\t".join(cols[active_indices]) + "\n")

            if algorithm == "fpgrowth":
                miner = FPGrowth.FPGrowth(db_path, minSup=min_sup)
            elif algorithm == "topk":
                miner = FAE.FAE(db_path, k=k)
            elif algorithm == "maxfpgrowth":
                miner = MaxFPGrowth.MaxFPGrowth(db_path, minSup=min_sup)

            miner.startMine()
            patterns_df = miner.getPatternsAsDataFrame()

        patterns = []
        if patterns_df is not None and not patterns_df.empty:
            pat_col = "Patterns" if "Patterns" in patterns_df.columns else patterns_df.columns[0]
            sup_col = "Support" if "Support" in patterns_df.columns else patterns_df.columns[1]

            for _, row in patterns_df.iterrows():
                pattern_str = str(row[pat_col])
                items = pattern_str.strip().split()
                support_val = int(row[sup_col])
                patterns.append({
                    "pattern": items,
                    "support": support_val
                })

        result_id = session.next_result_id("mine_patterns")

        summary = {
            "result_id": result_id,
            "status": "success",
            "category_name": category_name,
            "filtering_method": filtering_method,
            "algorithm": algorithm,
            "n_patterns": len(patterns),
            "patterns": patterns
        }

        session.artifacts[artifact_key] = patterns
        session.store_result("mine_patterns_tool", {"category_name": category_name, "filtering_method": filtering_method, "algorithm": algorithm, "min_sup": min_sup, "k": k}, summary)

        return summary

    return [mine_patterns_tool]
