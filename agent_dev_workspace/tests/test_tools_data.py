import pytest
import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib.figure
from pydantic import ValidationError
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_data import make_tools
from tools_dtm import make_tools as make_dtm_tools

def test_list_files_tool_known_answer_11():
    session = SessionState()
    tools = make_tools(session)
    list_files_tool = next(t for t in tools if t.name == "list_files_tool")

    # Known Answer #11: subdirectory="newdataset"
    res = list_files_tool.invoke({"subdirectory": "newdataset"})

    assert res["status"] == "success"
    assert res["subdirectory"] == "newdataset"
    assert res["directories"] == []
    assert res["files"] == ["Reddit-stock-sentiment.csv"]

def test_list_files_tool_edge_cases_and_sorting(tmp_path, monkeypatch):
    session = SessionState()
    tools = make_tools(session)
    list_files_tool = next(t for t in tools if t.name == "list_files_tool")

    # 1. Nonexistent directory
    res_nonexistent = list_files_tool.invoke({"subdirectory": "nonexistent_dir_12345"})
    assert res_nonexistent["status"] == "error"
    assert "does not exist" in res_nonexistent["message"]

    # 2. File instead of directory
    res_file = list_files_tool.invoke({"subdirectory": "agent_dev/sample_fixture.csv"})
    assert res_file["status"] == "error"
    assert "file, not a directory" in res_file["message"]

    # 3. Absolute path rejection
    abs_path = str(Path.cwd().resolve())
    res_abs = list_files_tool.invoke({"subdirectory": abs_path})
    assert res_abs["status"] == "error"
    assert "Only relative paths" in res_abs["message"]

    # 4. Path traversal rejection
    res_traversal = list_files_tool.invoke({"subdirectory": "../../.."})
    assert res_traversal["status"] == "error"
    assert "Access denied" in res_traversal["message"]

    # 5. Empty directory and alphabetical sorting test using monkeypatch
    mock_workspace = tmp_path / "mock_workspace"
    mock_workspace.mkdir()
    
    # Create empty subfolder
    empty_dir = mock_workspace / "empty_folder"
    empty_dir.mkdir()

    # Create folder with unordered files and subfolders
    sort_dir = mock_workspace / "sort_folder"
    sort_dir.mkdir()
    (sort_dir / "zebra.txt").touch()
    (sort_dir / "apple.txt").touch()
    (sort_dir / "dir_b").mkdir()
    (sort_dir / "dir_a").mkdir()

    # Monkeypatch cwd to point to mock_workspace
    monkeypatch.chdir(mock_workspace)

    # Empty directory check
    res_empty = list_files_tool.invoke({"subdirectory": "empty_folder"})
    assert res_empty["status"] == "success"
    assert res_empty["directories"] == []
    assert res_empty["files"] == []

    # Alphabetical sorting check
    res_sort = list_files_tool.invoke({"subdirectory": "sort_folder"})
    assert res_sort["status"] == "success"
    assert res_sort["directories"] == ["dir_a", "dir_b"]
    assert res_sort["files"] == ["apple.txt", "zebra.txt"]

def test_check_missing_tool_known_answer_2_and_edge_cases():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    tools = make_tools(session)
    check_missing_tool = next(t for t in tools if t.name == "check_missing_tool")

    # 1. Known Answer #2: 8 rows, exactly 1 missing row at positional index 7
    res = check_missing_tool.invoke({"column": "text"})
    assert res["status"] == "success"
    assert res["column"] == "text"
    assert res["total_rows"] == 8
    assert res["n_missing"] == 1
    assert res["missing_indices"] == [7]

    # 2. Test clean column with no missing values (e.g., category_name column)
    res_clean = check_missing_tool.invoke({"column": "category_name"})
    assert res_clean["status"] == "success"
    assert res_clean["n_missing"] == 0
    assert res_clean["missing_indices"] == []

    # 3. Test missing column error
    res_col_err = check_missing_tool.invoke({"column": "nonexistent_col"})
    assert res_col_err["status"] == "error"
    assert "not found" in res_col_err["message"]

    # 4. Test no dataframe loaded error
    session_no_df = SessionState()
    tools_no_df = make_tools(session_no_df)
    check_missing_no_df = next(t for t in tools_no_df if t.name == "check_missing_tool")
    res_no_df = check_missing_no_df.invoke({"column": "text"})
    assert res_no_df["status"] == "error"
    assert "No dataframe loaded" in res_no_df["message"]

    # 5. Test DataFrame with non-default index labels, NaN, None, empty string, and whitespace-only string
    session_custom = SessionState()
    session_custom.dataframe = pd.DataFrame(
        {"text": ["valid", None, "   ", "", np.nan, "\t\n", "valid2"]},
        index=[100, 200, 300, 400, 500, 600, 700]  # Non-default index labels
    )
    tools_custom = make_tools(session_custom)
    check_missing_custom = next(t for t in tools_custom if t.name == "check_missing_tool")
    res_custom = check_missing_custom.invoke({"column": "text"})

    assert res_custom["status"] == "success"
    assert res_custom["total_rows"] == 7
    assert res_custom["n_missing"] == 5
    # Must return positional 0-based indices [1, 2, 3, 4, 5], not DataFrame index labels [200, 300, 400, 500, 600]
    assert res_custom["missing_indices"] == [1, 2, 3, 4, 5]

def test_check_duplicates_tool_known_answer_3_and_edge_cases():
    # 1. Known Answer #3 check without drop (drop=False)
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    tools = make_tools(session)
    check_duplicates_tool = next(t for t in tools if t.name == "check_duplicates_tool")

    res = check_duplicates_tool.invoke({"column": "text", "drop": False})
    assert res["status"] == "success"
    assert res["total_rows_before"] == 8
    assert res["n_duplicates"] == 1
    assert res["duplicate_indices"] == [6]
    assert res["total_rows_after"] == 8
    assert len(session.dataframe) == 8

    # 2. Known Answer #3 check with drop (drop=True, keep=False)
    session_drop = SessionState()
    premade_tools_drop, _ = load_workspace_tools("premade_tools", session_drop)
    load_dataset_tool_drop = next(t for t in premade_tools_drop if t.name == "load_dataset_tool")
    load_dataset_tool_drop.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    dtm_tools_drop = make_dtm_tools(session_drop)
    build_dtm_drop = next(t for t in dtm_tools_drop if t.name == "build_dtm_tool")
    build_dtm_drop.invoke({})

    original_labels = session_drop.labels.copy() if session_drop.labels is not None else None
    assert session_drop.feature_matrix.shape[0] == 8

    tools_drop = make_tools(session_drop)
    check_duplicates_drop = next(t for t in tools_drop if t.name == "check_duplicates_tool")

    res_drop = check_duplicates_drop.invoke({"column": "text", "drop": True})
    assert res_drop["status"] == "success"
    assert res_drop["total_rows_before"] == 8
    assert res_drop["n_duplicates"] == 1
    assert res_drop["duplicate_indices"] == [6]
    assert res_drop["total_rows_after"] == 6
    assert len(session_drop.dataframe) == 6

    # Verify label and feature_matrix alignment after dropping rows 5 and 6
    if original_labels is not None:
        expected_labels = original_labels[[0, 1, 2, 3, 4, 7]]
        assert np.array_equal(session_drop.labels, expected_labels)

    assert session_drop.feature_matrix.shape[0] == 6

    # 3. Missing column error test
    res_err = check_duplicates_tool.invoke({"column": "nonexistent_column"})
    assert res_err["status"] == "error"

    # 4. No dataframe error test
    session_empty = SessionState()
    tools_empty = make_tools(session_empty)
    check_duplicates_empty = next(t for t in tools_empty if t.name == "check_duplicates_tool")
    res_empty = check_duplicates_empty.invoke({"column": "text"})
    assert res_empty["status"] == "error"

    # 5. Mismatched labels row count error test
    session_mismatch_labels = SessionState()
    session_mismatch_labels.dataframe = session.dataframe.copy()
    session_mismatch_labels.labels = np.array(["catA", "catB"]) # length 2 vs dataframe 8
    tools_mismatch1 = make_tools(session_mismatch_labels)
    dup_mismatch1 = next(t for t in tools_mismatch1 if t.name == "check_duplicates_tool")
    res_mism1 = dup_mismatch1.invoke({"column": "text", "drop": True})
    assert res_mism1["status"] == "error"
    assert "labels length" in res_mism1["message"]

    # 6. Mismatched feature matrix row count error test
    session_mismatch_fm = SessionState()
    session_mismatch_fm.dataframe = session.dataframe.copy()
    session_mismatch_fm.feature_matrix = sp.csr_matrix(np.ones((2, 5))) # 2 rows vs dataframe 8
    tools_mismatch2 = make_tools(session_mismatch_fm)
    dup_mismatch2 = next(t for t in tools_mismatch2 if t.name == "check_duplicates_tool")
    res_mism2 = dup_mismatch2.invoke({"column": "text", "drop": True})
    assert res_mism2["status"] == "error"
    assert "feature_matrix row count" in res_mism2["message"]

def test_tokenize_tool_known_answer_10_and_edge_cases():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    tools = make_tools(session)
    tokenize_tool = next(t for t in tools if t.name == "tokenize_tool")

    # 1. Known Answer #10 check on sample_fixture.csv
    res = tokenize_tool.invoke({"column": "text"})
    assert res["status"] == "success"
    assert res["n_documents"] == 8

    expected_tokens_all = [
        ["always", "alpha", "alpha", "alpha", "beta"],
        ["always", "alpha", "alpha", "beta"],
        ["always", "alpha", "beta"],
        ["always", "gamma", "gamma", "gamma", "delta"],
        ["always", "gamma", "gamma", "delta"],
        ["always", "gamma", "delta"],
        ["always", "gamma", "delta"],
        []
    ]

    assert session.dataframe["unigrams"].tolist() == expected_tokens_all
    assert session.artifacts["unigrams"] == expected_tokens_all
    assert res["sample_tokens"] == expected_tokens_all[:3]

    # 2. Test missing column error
    res_err = tokenize_tool.invoke({"column": "nonexistent_column"})
    assert res_err["status"] == "error"

    # 3. Test no dataframe error
    session_empty = SessionState()
    tools_empty = make_tools(session_empty)
    tokenize_empty = next(t for t in tools_empty if t.name == "tokenize_tool")
    res_no_df = tokenize_empty.invoke({"column": "text"})
    assert res_no_df["status"] == "error"

    # 4. Test custom text containing uppercase, punctuation, whitespace, None, NaN
    session_custom = SessionState()
    session_custom.dataframe = pd.DataFrame({
        "text": ["Hello, WORLD!", "   ", None, np.nan, "Alpha-Beta!"]
    })
    tools_custom = make_tools(session_custom)
    tokenize_custom = next(t for t in tools_custom if t.name == "tokenize_tool")
    res_custom = tokenize_custom.invoke({"column": "text"})

    assert res_custom["status"] == "success"
    unigrams_custom = session_custom.dataframe["unigrams"].tolist()

    # Lowercase + punctuation tokenized by word_tokenize
    assert unigrams_custom[0] == ["hello", ",", "world", "!"]
    assert unigrams_custom[1] == []
    assert unigrams_custom[2] == []
    assert unigrams_custom[3] == []
    assert unigrams_custom[4] == ["alpha-beta", "!"] or unigrams_custom[4] == ["alpha", "-", "beta", "!"]

def test_inspect_data_tool_comprehensive():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    tools = make_tools(session)
    inspect_data_tool = next(t for t in tools if t.name == "inspect_data_tool")

    # 1. Default inspection (n_rows=5, all columns)
    res = inspect_data_tool.invoke({})
    assert res["status"] == "success"
    assert res["total_rows"] == 8
    assert res["n_rows_returned"] == 5
    assert len(res["sample_data"]) == 5
    assert "text" in res["dtypes"]

    # 2. Selected columns
    res_cols = inspect_data_tool.invoke({"n_rows": 2, "columns": ["text"]})
    assert res_cols["status"] == "success"
    assert res_cols["columns"] == ["text"]
    assert res_cols["n_rows_returned"] == 2
    assert list(res_cols["sample_data"][0].keys()) == ["text"]

    # 3. Requested column missing error
    res_col_err = inspect_data_tool.invoke({"columns": ["text", "nonexistent_col"]})
    assert res_col_err["status"] == "error"
    assert "nonexistent_col" in res_col_err["message"]

    # 4. n_rows=0 error
    res_zero = inspect_data_tool.invoke({"n_rows": 0})
    assert res_zero["status"] == "error"

    # 5. Invalid type raises ValidationError or returns error for non-positive int
    for bad_n in [-1, 1.5, "five", True, False]:
        if type(bad_n) is int and bad_n <= 0:
            res_bad = inspect_data_tool.invoke({"n_rows": bad_n})
            assert res_bad["status"] == "error"
        else:
            with pytest.raises(ValidationError):
                inspect_data_tool.invoke({"n_rows": bad_n})

    # 6. No dataframe error
    session_no_df = SessionState()
    tools_no_df = make_tools(session_no_df)
    inspect_no_df = next(t for t in tools_no_df if t.name == "inspect_data_tool")
    assert inspect_no_df.invoke({})["status"] == "error"

    # 7. n_rows exceeding total_rows returns all available rows
    res_exceed = inspect_data_tool.invoke({"n_rows": 50})
    assert res_exceed["status"] == "success"
    assert res_exceed["n_rows_returned"] == 8
    assert len(res_exceed["sample_data"]) == 8

    # 8. Empty DataFrame with defined columns
    session_empty_df = SessionState()
    session_empty_df.dataframe = pd.DataFrame(columns=["col1", "col2"])
    tools_empty_df = make_tools(session_empty_df)
    inspect_empty_df = next(t for t in tools_empty_df if t.name == "inspect_data_tool")
    res_empty_df = inspect_empty_df.invoke({"n_rows": 5})
    assert res_empty_df["status"] == "success"
    assert res_empty_df["total_rows"] == 0
    assert res_empty_df["n_rows_returned"] == 0
    assert res_empty_df["sample_data"] == []

    # 9. Test JSON serialization with unigrams list, NaN, NaT, NumPy scalars, Pandas Timestamp
    session_complex = SessionState()
    session_complex.dataframe = pd.DataFrame({
        "unigrams": [["word1", "word2"], [], None],
        "numbers": [np.int64(10), np.float64(2.5), np.nan],
        "timestamps": [pd.Timestamp("2025-01-01T12:00:00"), pd.NaT, pd.Timestamp("2025-01-02T15:30:00")]
    })
    tools_complex = make_tools(session_complex)
    inspect_complex = next(t for t in tools_complex if t.name == "inspect_data_tool")
    res_complex = inspect_complex.invoke({"n_rows": 3})

    assert res_complex["status"] == "success"
    assert len(res_complex["sample_data"]) == 3
    
    # Check JSON serializability
    json_str = json.dumps(res_complex["sample_data"])
    assert isinstance(json_str, str)
    
    data = res_complex["sample_data"]
    assert data[0]["unigrams"] == ["word1", "word2"]
    assert data[1]["unigrams"] == []
    assert data[2]["unigrams"] is None
    assert data[2]["numbers"] is None
    assert data[1]["timestamps"] is None
    assert "2025-01-01" in data[0]["timestamps"]
    assert "2025-01-02" in data[2]["timestamps"]

def test_sample_data_tool_known_answer_13_and_edge_cases():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    tools = make_tools(session)
    sample_data_tool = next(t for t in tools if t.name == "sample_data_tool")

    # 1. Known Answer #13 check: n=4, random_state=42 -> sampled_indices == [1, 5, 0, 7]
    res = sample_data_tool.invoke({"n": 4, "random_state": 42})
    assert res["status"] == "success"
    assert res["n_requested"] == 4
    assert res["n_returned"] == 4
    assert res["total_rows"] == 8
    assert res["random_state"] == 42
    assert res["sampled_indices"] == [1, 5, 0, 7]
    assert len(res["sample_data"]) == 4

    # Verify session results store using ResultRecord .summary attribute
    stored_result = session.results_store[res["result_id"]]
    assert stored_result["record"].summary["sampled_indices"] == [1, 5, 0, 7]

    # Verify read-only behavior: session.dataframe is unmodified
    assert len(session.dataframe) == 8

    # 2. Reproducibility test
    res2 = sample_data_tool.invoke({"n": 4, "random_state": 42})
    assert res2["sampled_indices"] == [1, 5, 0, 7]

    # 3. Duplicate DataFrame index labels test
    session_dup_idx = SessionState()
    session_dup_idx.dataframe = pd.DataFrame(
        {"text": [f"doc_{i}" for i in range(8)]},
        index=[10, 10, 20, 20, 30, 30, 40, 40]  # Duplicate index labels
    )
    tools_dup_idx = make_tools(session_dup_idx)
    sample_dup_idx = next(t for t in tools_dup_idx if t.name == "sample_data_tool")
    res_dup_idx = sample_dup_idx.invoke({"n": 4, "random_state": 42})
    assert res_dup_idx["status"] == "success"
    assert res_dup_idx["sampled_indices"] == [1, 5, 0, 7]

    # 4. n exceeding total rows returns all rows
    res_exceed = sample_data_tool.invoke({"n": 50})
    assert res_exceed["status"] == "success"
    assert res_exceed["n_returned"] == 8
    assert len(res_exceed["sampled_indices"]) == 8

    # 5. Empty DataFrame test
    session_empty_df = SessionState()
    session_empty_df.dataframe = pd.DataFrame(columns=["text"])
    tools_empty_df = make_tools(session_empty_df)
    sample_empty_df = next(t for t in tools_empty_df if t.name == "sample_data_tool")
    res_empty = sample_empty_df.invoke({"n": 5})
    assert res_empty["status"] == "success"
    assert res_empty["n_returned"] == 0
    assert res_empty["total_rows"] == 0
    assert res_empty["sampled_indices"] == []
    assert res_empty["sample_data"] == []

    # 6. Invalid non-positive n or random_state
    for bad_val in [0, -1]:
        res_bad_n = sample_data_tool.invoke({"n": bad_val})
        assert res_bad_n["status"] == "error"

    # 7. Invalid types raise ValidationError
    for bad_type in [1.5, "four", True, False]:
        with pytest.raises(ValidationError):
            sample_data_tool.invoke({"n": bad_type})
        with pytest.raises(ValidationError):
            sample_data_tool.invoke({"n": 4, "random_state": bad_type})

    # 8. No dataframe loaded error
    session_no_df = SessionState()
    tools_no_df = make_tools(session_no_df)
    sample_no_df = next(t for t in tools_no_df if t.name == "sample_data_tool")
    assert sample_no_df.invoke({"n": 4})["status"] == "error"

def test_describe_data_tool_known_answer_18_and_edge_cases():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    tools = make_tools(session)
    describe_data_tool = next(t for t in tools if t.name == "describe_data_tool")

    # 1. Known Answer #18 check
    res = describe_data_tool.invoke({})
    assert res["status"] == "success"
    assert res["total_documents"] == 8

    # Overall stats: count=8, mean=19.875, std=9.4330, min=0, 25%=17.75, 50%=20.5, 75%=25.25, max=30
    overall = res["overall_stats"]
    assert overall["count"] == 8.0
    assert np.isclose(overall["mean"], 19.875, atol=1e-3)
    assert np.isclose(overall["std"], 9.4330, atol=1e-3)
    assert overall["min"] == 0.0
    assert np.isclose(overall["25%"], 17.75, atol=1e-3)
    assert np.isclose(overall["50%"], 20.5, atol=1e-3)
    assert np.isclose(overall["75%"], 25.25, atol=1e-3)
    assert overall["max"] == 30.0

    # Per-category stats
    by_cat = res["by_category"]
    assert "catA" in by_cat and "catB" in by_cat

    # catA stats: count=4, mean=17.25, std=12.5000, min=0, 25%=12.75, 50%=20.0, 75%=24.5, max=29
    catA = by_cat["catA"]
    assert catA["count"] == 4.0
    assert np.isclose(catA["mean"], 17.25, atol=1e-3)
    assert np.isclose(catA["std"], 12.5000, atol=1e-3)
    assert catA["min"] == 0.0
    assert np.isclose(catA["25%"], 12.75, atol=1e-3)
    assert np.isclose(catA["50%"], 20.0, atol=1e-3)
    assert np.isclose(catA["75%"], 24.5, atol=1e-3)
    assert catA["max"] == 29.0

    # catB stats: count=4, mean=22.50, std=5.7446, min=18, 25%=18.00, 50%=21.0, 75%=25.5, max=30
    catB = by_cat["catB"]
    assert catB["count"] == 4.0
    assert np.isclose(catB["mean"], 22.50, atol=1e-3)
    assert np.isclose(catB["std"], 5.7446, atol=1e-3)
    assert catB["min"] == 18.0
    assert np.isclose(catB["25%"], 18.00, atol=1e-3)
    assert np.isclose(catB["50%"], 21.0, atol=1e-3)
    assert np.isclose(catB["75%"], 25.5, atol=1e-3)
    assert catB["max"] == 30.0

    # Check pending_figure contract
    assert session.pending_figure is not None
    assert isinstance(session.pending_figure, matplotlib.figure.Figure)

    # Check stored_result using ResultRecord .summary
    stored_result = session.results_store[res["result_id"]]
    assert stored_result["record"].summary["total_documents"] == 8

    # 2. Category column named 'text_length'
    session_textlen = SessionState()
    session_textlen.dataframe = pd.DataFrame({
        "text": ["hello", "world", "text mining"],
        "text_length": ["cat1", "cat1", "cat2"]
    })
    tools_textlen = make_tools(session_textlen)
    describe_textlen = next(t for t in tools_textlen if t.name == "describe_data_tool")
    res_textlen = describe_textlen.invoke({"category_column": "text_length"})
    assert res_textlen["status"] == "success"
    assert "cat1" in res_textlen["by_category"]

    # 3. Missing text values and missing category labels
    session_missing = SessionState()
    session_missing.dataframe = pd.DataFrame({
        "text": ["valid text", None, np.nan, "text2"],
        "cat": ["A", np.nan, "B", "B"]
    })
    tools_missing = make_tools(session_missing)
    describe_missing = next(t for t in tools_missing if t.name == "describe_data_tool")
    res_missing = describe_missing.invoke({"category_column": "cat"})
    assert res_missing["status"] == "success"
    assert res_missing["total_documents"] == 4
    # Missing category is excluded from by_category
    assert set(res_missing["by_category"].keys()) == {"A", "B"}

    # 4. Explicitly invalid category_column error
    res_cat_err = describe_data_tool.invoke({"category_column": "nonexistent_cat"})
    assert res_cat_err["status"] == "error"

    # 5. Missing text_column error
    res_text_err = describe_data_tool.invoke({"text_column": "nonexistent_text"})
    assert res_text_err["status"] == "error"

    # 6. Empty DataFrame clears pending_figure and returns count 0
    session_empty = SessionState()
    session_empty.pending_figure = matplotlib.figure.Figure() # Set existing figure
    session_empty.dataframe = pd.DataFrame(columns=["text", "cat"])
    tools_empty = make_tools(session_empty)
    describe_empty = next(t for t in tools_empty if t.name == "describe_data_tool")
    res_empty = describe_empty.invoke({})
    assert res_empty["status"] == "success"
    assert res_empty["total_documents"] == 0
    assert res_empty["overall_stats"]["count"] == 0.0
    assert session_empty.pending_figure is None

    # 7. No dataframe error
    session_no_df = SessionState()
    tools_no_df = make_tools(session_no_df)
    describe_no_df = next(t for t in tools_no_df if t.name == "describe_data_tool")
    assert describe_no_df.invoke({})["status"] == "error"
