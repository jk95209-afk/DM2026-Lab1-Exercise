import pytest
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_exploration import make_tools as make_exploration_tools

def test_cosine_similarity_tool_known_answer_16():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    exploration_tools = make_exploration_tools(session)
    cosine_similarity_tool = next(t for t in exploration_tools if t.name == "cosine_similarity_tool")

    # 1. Known Answer #16 Doc 0 vs 1 (both catA, near-identical text) = 0.9847
    res_01 = cosine_similarity_tool.invoke({"doc_id1": 0, "doc_id2": 1})
    assert res_01["status"] == "success"
    assert np.isclose(res_01["similarity"], 0.9847, atol=1e-3)

    # 2. Known Answer #16 Doc 0 vs 3 (catA vs catB, share only "always") = 0.0909
    res_03 = cosine_similarity_tool.invoke({"doc_id1": 0, "doc_id2": 3})
    assert res_03["status"] == "success"
    assert np.isclose(res_03["similarity"], 0.0909, atol=1e-3)

    # 3. Known Answer #16 Doc 3 vs 4 (both catB, near-identical text) = 0.9847
    res_34 = cosine_similarity_tool.invoke({"doc_id1": 3, "doc_id2": 4})
    assert res_34["status"] == "success"
    assert np.isclose(res_34["similarity"], 0.9847, atol=1e-3)

    # 4. Known Answer #16 Doc 0 vs 7 (catA vs empty text row) = 0.0
    res_07 = cosine_similarity_tool.invoke({"doc_id1": 0, "doc_id2": 7})
    assert res_07["status"] == "success"
    assert res_07["similarity"] == 0.0

    # 5. Out of bounds validation
    res_oob1 = cosine_similarity_tool.invoke({"doc_id1": -1, "doc_id2": 0})
    assert res_oob1["status"] == "error"

    res_oob2 = cosine_similarity_tool.invoke({"doc_id1": 0, "doc_id2": 99})
    assert res_oob2["status"] == "error"

    # 6. Symmetry: similarity(0, 1) == similarity(1, 0)
    res_10 = cosine_similarity_tool.invoke({"doc_id1": 1, "doc_id2": 0})
    assert res_10["status"] == "success"
    assert np.isclose(res_10["similarity"], res_01["similarity"], atol=1e-5)

    # 7. Self-similarity: similarity(0, 0) == 1.0 for a nonzero document vector
    res_00 = cosine_similarity_tool.invoke({"doc_id1": 0, "doc_id2": 0})
    assert res_00["status"] == "success"
    assert np.isclose(res_00["similarity"], 1.0, atol=1e-3)

def test_feature_correlation_matrix_tool_known_answer_19():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    # Verify that session.feature_matrix is sparse prior to tool invocation
    assert sp.issparse(session.feature_matrix)

    exploration_tools = make_exploration_tools(session)
    feature_corr_tool = next(t for t in exploration_tools if t.name == "feature_correlation_matrix_tool")

    res = feature_corr_tool.invoke({"top_k": 20})

    assert res["status"] == "success"
    assert res["n_terms"] == 5
    # Order descending by variance: alpha, gamma, delta, beta, always
    expected_terms = ["alpha", "gamma", "delta", "beta", "always"]
    assert res["terms"] == expected_terms

    # Check correlation matrix shape and values from Known Answer #19
    corr_matrix = np.array(res["correlation_matrix"])
    assert corr_matrix.shape == (5, 5)

    # Diagonal is 1.0
    assert np.allclose(np.diag(corr_matrix), 1.0)

    # Verify specific pairs numerically (atol=1e-3)
    # alpha (0) vs beta (3) = 0.8885
    assert np.isclose(corr_matrix[0, 3], 0.8885, atol=1e-3)
    # gamma (1) vs delta (2) = 0.8307
    assert np.isclose(corr_matrix[1, 2], 0.8307, atol=1e-3)
    # alpha (0) vs gamma (1) = -0.5718
    assert np.isclose(corr_matrix[0, 1], -0.5718, atol=1e-3)

    # Check pending_figure contract
    assert session.pending_figure is not None
    assert isinstance(session.pending_figure, plt.Figure)

def test_feature_correlation_matrix_sparse_and_edge_cases(monkeypatch):
    np.random.seed(42)
    session = SessionState()
    # Mock a large sparse matrix with 100 features and 10 rows
    session.feature_matrix = sp.csr_matrix(np.random.poisson(lam=1.0, size=(10, 100)))
    session.feature_names = [f"term_{i}" for i in range(100)]

    # Intercept toarray() to assert no matrix with > 20 columns is densified
    original_toarray = sp.csr_matrix.toarray
    def mock_toarray(self, *args, **kwargs):
        assert self.shape[1] <= 20, f"Dense conversion attempted on matrix with {self.shape[1]} columns (> 20)!"
        return original_toarray(self, *args, **kwargs)

    monkeypatch.setattr(sp.csr_matrix, "toarray", mock_toarray)

    exploration_tools = make_exploration_tools(session)
    feature_corr_tool = next(t for t in exploration_tools if t.name == "feature_correlation_matrix_tool")

    res = feature_corr_tool.invoke({"top_k": 50})
    assert res["status"] == "success"
    assert res["n_terms"] == 20
    assert len(res["terms"]) == 20
    assert len(res["correlation_matrix"]) == 20

    # Single feature edge case with non-zero variance -> expecting [[1.0]]
    session_single_nonzero = SessionState()
    session_single_nonzero.feature_matrix = sp.csr_matrix(np.array([[1], [2], [3], [4]]))
    session_single_nonzero.feature_names = ["var_term"]
    tools_single = make_exploration_tools(session_single_nonzero)
    corr_single = next(t for t in tools_single if t.name == "feature_correlation_matrix_tool")
    res_single = corr_single.invoke({"top_k": 1})
    assert res_single["status"] == "success"
    assert res_single["correlation_matrix"] == [[1.0]]

    # Single feature edge case with zero variance (constant column) -> expecting [[0.0]]
    session_single_zero = SessionState()
    session_single_zero.feature_matrix = sp.csr_matrix(np.ones((10, 1)))
    session_single_zero.feature_names = ["const_term"]
    tools_single2 = make_exploration_tools(session_single_zero)
    corr_single2 = next(t for t in tools_single2 if t.name == "feature_correlation_matrix_tool")
    res_single2 = corr_single2.invoke({"top_k": 1})
    assert res_single2["status"] == "success"
    assert res_single2["correlation_matrix"] == [[0.0]]
