import pytest
import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib.figure
from pydantic import ValidationError
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools

def test_build_dtm_tool_known_answer_4():
    # 1. Instantiate single session state
    session = SessionState()

    # 2. Load premade load_dataset_tool to load sample_fixture.csv
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # 3. Instantiate student tool under test with same session
    tools = make_tools(session)
    build_dtm_tool = next(t for t in tools if t.name == "build_dtm_tool")

    # 4. Invoke build_dtm_tool with default parameters
    res = build_dtm_tool.invoke({})

    # 5. Assertions matching known answer #4 in TEST_FIXTURE.md
    expected_vocabulary = ["alpha", "always", "beta", "delta", "gamma"]
    assert session.feature_names == expected_vocabulary
    assert res["n_terms"] == 5
    assert res["n_documents"] == 8

    assert res["non_zero"] == 21
    assert res["total_elements"] == 40
    assert np.isclose(res["sparsity_pct"], 47.5, atol=1e-3)

    assert "count_vectorizer" in session.artifacts

def test_term_frequency_tool_known_answer_14():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    tools = make_tools(session)
    build_dtm_tool = next(t for t in tools if t.name == "build_dtm_tool")
    term_frequency_tool = next(t for t in tools if t.name == "term_frequency_tool")

    build_dtm_tool.invoke({})

    # Verify feature_matrix is sparse
    assert sp.issparse(session.feature_matrix)

    res = term_frequency_tool.invoke({})

    assert res["status"] == "success"
    assert res["total_terms"] == 5
    assert res["total_word_count"] == 27  # 6 + 7 + 3 + 4 + 7

    # Known Answer #14 exact frequencies
    expected_freqs = {
        "alpha": 6,
        "always": 7,
        "beta": 3,
        "delta": 4,
        "gamma": 7
    }
    assert res["term_frequencies"] == expected_freqs

    # Verify full_report stored in session.results_store and accessible for visualization
    stored_result = session.results_store[res["result_id"]]
    full_report = stored_result["full_report"]
    assert isinstance(full_report, pd.DataFrame)
    assert list(full_report.columns[:2]) == ["term", "frequency"]

    # Verify deterministic sorting: frequency desc, term asc (ties: always, gamma both 7)
    expected_report_terms = ["always", "gamma", "alpha", "delta", "beta"]
    assert full_report["term"].tolist() == expected_report_terms
    assert full_report["frequency"].tolist() == [7, 7, 6, 4, 3]

    # Test top_k parameter
    res_top2 = term_frequency_tool.invoke({"top_k": 2})
    assert res_top2["status"] == "success"
    assert res_top2["term_frequencies"] == {"always": 7, "gamma": 7}

    # Test nonpositive integer top_k for returned error status
    for nonpos_k in [0, -1]:
        res_err = term_frequency_tool.invoke({"top_k": nonpos_k})
        assert res_err["status"] == "error"

    # Test invalid type top_k parameters for ValidationError
    for bad_type_k in [1.5, "two", True, False]:
        with pytest.raises(ValidationError):
            term_frequency_tool.invoke({"top_k": bad_type_k})

def test_dtm_heatmap_tool_known_answer_21(monkeypatch):
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    tools = make_tools(session)
    build_dtm_tool = next(t for t in tools if t.name == "build_dtm_tool")
    dtm_heatmap_tool = next(t for t in tools if t.name == "dtm_heatmap_tool")

    build_dtm_tool.invoke({})

    # 1. Default call returns the exact 8x5 matrix
    res = dtm_heatmap_tool.invoke({})
    assert res["status"] == "success"
    assert res["n_documents"] == 8
    assert res["n_terms"] == 5
    assert res["terms"] == ["alpha", "always", "beta", "delta", "gamma"]

    expected_full_matrix = [
        [3, 1, 1, 0, 0],
        [2, 1, 1, 0, 0],
        [1, 1, 1, 0, 0],
        [0, 1, 0, 1, 3],
        [0, 1, 0, 1, 2],
        [0, 1, 0, 1, 1],
        [0, 1, 0, 1, 1],
        [0, 0, 0, 0, 0]
    ]
    assert res["matrix_slice"] == expected_full_matrix

    # Check pending_figure contract
    assert session.pending_figure is not None
    assert isinstance(session.pending_figure, matplotlib.figure.Figure)

    # 2. n_terms=2, n_documents=3 returns [[3,1],[2,1],[1,1]]
    res_32 = dtm_heatmap_tool.invoke({"n_documents": 3, "n_terms": 2})
    assert res_32["status"] == "success"
    assert res_32["n_documents"] == 3
    assert res_32["n_terms"] == 2
    assert res_32["matrix_slice"] == [[3, 1], [2, 1], [1, 1]]

    # 3. Nonpositive dimensions return error status
    for bad_dim in [0, -1]:
        assert dtm_heatmap_tool.invoke({"n_terms": bad_dim})["status"] == "error"
        assert dtm_heatmap_tool.invoke({"n_documents": bad_dim})["status"] == "error"

    # 4. Invalid types raise ValidationError
    for bad_type in [1.5, "two", True, False]:
        with pytest.raises(ValidationError):
            dtm_heatmap_tool.invoke({"n_terms": bad_type})
        with pytest.raises(ValidationError):
            dtm_heatmap_tool.invoke({"n_documents": bad_type})

    # 5. Empty DTM returns error
    session_empty = SessionState()
    session_empty.feature_matrix = sp.csr_matrix((0, 0))
    session_empty.feature_names = []
    tools_empty = make_tools(session_empty)
    dtm_heatmap_empty = next(t for t in tools_empty if t.name == "dtm_heatmap_tool")
    assert dtm_heatmap_empty.invoke({})["status"] == "error"

    # 6. Sparse matrix test verifying only selected slice is densified
    session_large = SessionState()
    session_large.feature_matrix = sp.csr_matrix(np.ones((100, 100)))
    session_large.feature_names = [f"t_{i}" for i in range(100)]
    tools_large = make_tools(session_large)
    dtm_heatmap_large = next(t for t in tools_large if t.name == "dtm_heatmap_tool")

    original_toarray = sp.csr_matrix.toarray
    def mock_toarray(self, *args, **kwargs):
        assert self.shape[0] <= 5 and self.shape[1] <= 5, f"Dense conversion attempted on matrix larger than 5x5: {self.shape}"
        return original_toarray(self, *args, **kwargs)

    monkeypatch.setattr(sp.csr_matrix, "toarray", mock_toarray)
    res_large = dtm_heatmap_large.invoke({"n_documents": 5, "n_terms": 5})
    assert res_large["status"] == "success"
    assert res_large["n_documents"] == 5
    assert res_large["n_terms"] == 5
