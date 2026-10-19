import pytest
import numpy as np
import pandas as pd
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_filtering import make_tools as make_filtering_tools

def test_variance_filter_tool_known_answer_5():
    # 1. Instantiate single session state
    session = SessionState()

    # 2. Load dataset
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # 3. Build DTM
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    # 4. Filter by variance using threshold = 0.15
    filtering_tools = make_filtering_tools(session)
    variance_filter_tool = next(t for t in filtering_tools if t.name == "variance_filter_tool")
    res = variance_filter_tool.invoke({"threshold": 0.15})

    # 5. Assertions against known answer #5 in TEST_FIXTURE.md
    # Known variances: alpha=1.1875, always=0.109375, beta=0.234375, delta=0.25, gamma=1.109375
    # threshold=0.15 keeps alpha, beta, delta, gamma (4 terms) and removes always (1 term)
    assert res["n_total"] == 5
    assert res["n_kept"] == 4
    assert res["n_removed"] == 1
    assert "always" in res["removed_terms"]
    assert set(res["kept_terms"]) == {"alpha", "beta", "delta", "gamma"}

    # Verify full_report DataFrame stored in session results
    stored_result = session.results_store[res["result_id"]]
    full_report = stored_result["full_report"]
    assert isinstance(full_report, pd.DataFrame)
    assert list(full_report.columns[:2]) == ["term", "variance"]

    # Assert that full_report["variance"] is sorted in descending order as required by contract
    assert full_report["variance"].is_monotonic_decreasing

    # Verify specific term variances numerically
    var_dict = dict(zip(full_report["term"], full_report["variance"]))
    assert np.isclose(var_dict["alpha"], 1.1875, atol=1e-3)
    assert np.isclose(var_dict["always"], 0.109375, atol=1e-3)
    assert np.isclose(var_dict["beta"], 0.234375, atol=1e-3)
    assert np.isclose(var_dict["delta"], 0.25, atol=1e-3)
    assert np.isclose(var_dict["gamma"], 1.109375, atol=1e-3)

def test_pearson_filter_tool_known_answer_6():
    # 1. Instantiate fresh single session state
    session = SessionState()

    # 2. Load dataset
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # 3. Build DTM
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    # 4. Invoke pearson_filter_tool with target_class="catB" and threshold=0.7
    filtering_tools = make_filtering_tools(session)
    pearson_filter_tool = next(t for t in filtering_tools if t.name == "pearson_filter_tool")
    res = pearson_filter_tool.invoke({"target_class": "catB", "threshold": 0.7})

    # 5. Assertions against known answer #6 in TEST_FIXTURE.md
    # Known Pearson r for catB: alpha=-0.6882, always=0.3780, beta=-0.7746, delta=1.0000, gamma=0.8307
    # Threshold=0.7 on abs(pearson_r) keeps beta (|-0.7746|>=0.7), delta (|1.0|>=0.7), gamma (|0.8307|>=0.7) -> 3 terms
    # Removes alpha (|-0.6882|<0.7) and always (|0.3780|<0.7) -> 2 terms
    assert res["n_total"] == 5
    assert res["n_kept"] == 3
    assert res["n_removed"] == 2
    assert set(res["kept_terms"]) == {"beta", "delta", "gamma"}
    assert set(res["removed_terms"]) == {"alpha", "always"}

    # Verify full_report DataFrame stored in session results
    stored_result = session.results_store[res["result_id"]]
    full_report = stored_result["full_report"]
    assert isinstance(full_report, pd.DataFrame)
    assert list(full_report.columns[:2]) == ["term", "pearson_r"]

    # Verify full_report is sorted by signed pearson_r descending
    assert full_report["pearson_r"].is_monotonic_decreasing

    # Verify Pearson r values numerically for all five terms (atol=1e-3)
    r_dict = dict(zip(full_report["term"], full_report["pearson_r"]))
    assert np.isclose(r_dict["alpha"], -0.6882, atol=1e-3)
    assert np.isclose(r_dict["always"], 0.3780, atol=1e-3)
    assert np.isclose(r_dict["beta"], -0.7746, atol=1e-3)
    assert np.isclose(r_dict["delta"], 1.0000, atol=1e-3)
    assert np.isclose(r_dict["gamma"], 0.8307, atol=1e-3)

def test_spearman_filter_tool_known_answer_6():
    # 1. Instantiate fresh single session state
    session = SessionState()

    # 2. Load dataset
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # 3. Build DTM
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    # 4. Invoke spearman_filter_tool with target_class="catB" and threshold=0.7
    filtering_tools = make_filtering_tools(session)
    spearman_filter_tool = next(t for t in filtering_tools if t.name == "spearman_filter_tool")
    res = spearman_filter_tool.invoke({"target_class": "catB", "threshold": 0.7})

    # 5. Assertions against known answer #6 in TEST_FIXTURE.md
    # Known Spearman r for catB: alpha=-0.7500, always=0.3780, beta=-0.7746, delta=1.0000, gamma=0.9363
    # Threshold=0.7 on abs(spearman_r) keeps alpha (|-0.7500|>=0.7), beta (|-0.7746|>=0.7), delta (|1.0|>=0.7), gamma (|0.9363|>=0.7) -> 4 terms
    # Removes always (|0.3780|<0.7) -> 1 term
    assert res["n_total"] == 5
    assert res["n_kept"] == 4
    assert res["n_removed"] == 1
    assert set(res["kept_terms"]) == {"alpha", "beta", "delta", "gamma"}
    assert set(res["removed_terms"]) == {"always"}

    # Verify full_report DataFrame stored in session results
    stored_result = session.results_store[res["result_id"]]
    full_report = stored_result["full_report"]
    assert isinstance(full_report, pd.DataFrame)
    assert list(full_report.columns[:2]) == ["term", "spearman_r"]

    # Verify full_report is sorted by signed spearman_r descending
    assert full_report["spearman_r"].is_monotonic_decreasing

    # Verify Spearman r values numerically for all five terms (atol=1e-3)
    r_dict = dict(zip(full_report["term"], full_report["spearman_r"]))
    assert np.isclose(r_dict["alpha"], -0.7500, atol=1e-3)
    assert np.isclose(r_dict["always"], 0.3780, atol=1e-3)
    assert np.isclose(r_dict["beta"], -0.7746, atol=1e-3)
    assert np.isclose(r_dict["delta"], 1.0000, atol=1e-3)
    assert np.isclose(r_dict["gamma"], 0.9363, atol=1e-3)
