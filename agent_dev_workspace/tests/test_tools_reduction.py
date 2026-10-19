import pytest
import numpy as np
import matplotlib.figure
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_reduction import make_tools as make_reduction_tools

def test_reduce_dimensions_pca_known_answer_7():
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

    reduction_tools = make_reduction_tools(session)
    reduce_dimensions_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")

    res = reduce_dimensions_tool.invoke({"method": "pca", "n_components": 2})

    assert res["status"] == "success"
    assert res["n_documents"] == 8
    coords_arr = np.array(res["coordinates"])
    assert coords_arr.shape == (8, 2)
    assert np.all(np.isfinite(coords_arr))

    # Known Answer #7 explained_variance_ratio check
    assert np.isclose(res["explained_variance_ratio"][0], 0.7532, atol=1e-3)
    assert np.isclose(res["explained_variance_ratio"][1], 0.1965, atol=1e-3)

    # Known Answer #7 exact coordinates check
    expected_pca_coords = np.array([
        [2.3669, 0.8871],
        [1.7039, 0.2642],
        [1.0408, -0.3587],
        [-2.0759, 1.0129],
        [-1.4554, 0.3405],
        [-0.8349, -0.3319],
        [-0.8349, -0.3319],
        [0.0894, -1.4822]
    ])
    assert np.allclose(coords_arr, expected_pca_coords, atol=1e-3) or \
           np.allclose(np.abs(coords_arr), np.abs(expected_pca_coords), atol=1e-3)

    # Check pending_figure contract
    assert session.pending_figure is not None
    assert isinstance(session.pending_figure, matplotlib.figure.Figure)

    # Check session artifacts
    assert "reduced_coordinates" in session.artifacts
    assert len(session.artifacts["reduced_coordinates"]) == 8

def test_reduce_dimensions_tsne_and_umap():
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

    reduction_tools = make_reduction_tools(session)
    reduce_dimensions_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")

    # t-SNE test with perplexity=2, random_state=42
    session.pending_figure = None
    res_tsne = reduce_dimensions_tool.invoke({"method": "tsne", "perplexity": 2, "random_state": 42})
    assert res_tsne["status"] == "success"
    tsne_coords = np.array(res_tsne["coordinates"])
    assert tsne_coords.shape == (8, 2)
    assert np.all(np.isfinite(tsne_coords))
    assert session.pending_figure is not None

    # UMAP test with n_neighbors=3, random_state=42
    session.pending_figure = None
    res_umap = reduce_dimensions_tool.invoke({"method": "umap", "n_neighbors": 3, "random_state": 42})
    assert res_umap["status"] == "success"
    umap_coords = np.array(res_umap["coordinates"])
    assert umap_coords.shape == (8, 2)
    assert np.all(np.isfinite(umap_coords))
    assert session.pending_figure is not None

def test_binarize_labels_tool_comprehensive():
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

    reduction_tools = make_reduction_tools(session)
    binarize_labels_tool = next(t for t in reduction_tools if t.name == "binarize_labels_tool")

    # Detect original label column and record original session labels
    label_col = "category_name" if "category_name" in session.dataframe.columns else "label"
    original_multiclass = session.dataframe[label_col].tolist()
    original_session_labels = session.labels.copy() if session.labels is not None else None

    # 1. Test correct encoding for target_class="catB"
    res = binarize_labels_tool.invoke({"target_class": "catB"})
    assert res["status"] == "success"
    assert res["n_documents"] == 8
    assert res["n_target"] == 4
    assert res["n_rest"] == 4
    assert res["binary_counts"] == {"1": 4, "0": 4}

    # Verify binary_label column added without modifying original multiclass column or session.labels
    expected_binary = [0, 0, 0, 1, 1, 1, 1, 0]
    assert session.dataframe["binary_label"].tolist() == expected_binary
    assert session.dataframe[label_col].tolist() == original_multiclass
    if original_session_labels is not None:
        assert np.array_equal(session.labels, original_session_labels)

    # Verify artifacts
    assert session.artifacts["binary_label"] == expected_binary
    assert session.artifacts["binary_label_catB"] == expected_binary

    # 2. Test repeated call with target_class="catA"
    res_a = binarize_labels_tool.invoke({"target_class": "catA"})
    assert res_a["status"] == "success"
    assert res_a["n_target"] == 4
    assert res_a["n_rest"] == 4
    expected_binary_a = [1, 1, 1, 0, 0, 0, 0, 1]
    assert session.dataframe["binary_label"].tolist() == expected_binary_a
    assert session.dataframe[label_col].tolist() == original_multiclass

    # 3. Test invalid target_class
    res_err = binarize_labels_tool.invoke({"target_class": "nonexistent"})
    assert res_err["status"] == "error"

    # 4. Test label count mismatch error
    session_mismatch = SessionState()
    session_mismatch.dataframe = session.dataframe.copy()
    session_mismatch.labels = np.array(["catA", "catB"]) # length 2 vs dataframe length 8
    tools_mismatch = make_reduction_tools(session_mismatch)
    binarize_mismatch = next(t for t in tools_mismatch if t.name == "binarize_labels_tool")
    res_mismatch = binarize_mismatch.invoke({"target_class": "catA"})
    assert res_mismatch["status"] == "error"

    # 5. Test single-class labels rejection error
    session_single = SessionState()
    session_single.dataframe = session.dataframe.copy()
    session_single.labels = np.array(["catA"] * 8)
    tools_single = make_reduction_tools(session_single)
    binarize_single = next(t for t in tools_single if t.name == "binarize_labels_tool")
    res_single = binarize_single.invoke({"target_class": "catA"})
    assert res_single["status"] == "error"
