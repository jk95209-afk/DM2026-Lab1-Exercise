import pytest
import pandas as pd
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_patterns import make_tools

def test_mine_patterns_tool_known_answers_17_and_20():
    # 1. Instantiate single session state and load sample_fixture.csv
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # 2. Instantiate student tool
    tools = make_tools(session)
    mine_patterns_tool = next(t for t in tools if t.name == "mine_patterns_tool")

    # --- 1. Variance and term_frequency filtering: 0 surviving terms ---
    for cat in ["catA", "catB"]:
        for filt in ["variance", "term_frequency"]:
            res = mine_patterns_tool.invoke({
                "category_name": cat,
                "filtering_method": filt,
                "algorithm": "fpgrowth",
                "min_sup": 1
            })
            assert res["status"] == "success"
            assert res["n_patterns"] == 0
            assert res["patterns"] == []
            assert "no terms survived" in res.get("note", "")

    # --- 2. TF-IDF with FPGrowth, min_sup=1 ---
    # catA -> alpha with support 3
    res_a1 = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 1
    })
    assert res_a1["status"] == "success"
    assert res_a1["n_patterns"] == 1
    assert res_a1["patterns"][0]["pattern"] == ["alpha"]
    assert res_a1["patterns"][0]["support"] == 3

    # catB -> gamma with support 4
    res_b1 = mine_patterns_tool.invoke({
        "category_name": "catB",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 1
    })
    assert res_b1["status"] == "success"
    assert res_b1["n_patterns"] == 1
    assert res_b1["patterns"][0]["pattern"] == ["gamma"]
    assert res_b1["patterns"][0]["support"] == 4

    # --- 3. FPGrowth with min_sup=4 for catA -> 0 patterns ---
    res_a4 = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 4
    })
    assert res_a4["status"] == "success"
    assert res_a4["n_patterns"] == 0
    assert res_a4["patterns"] == []

    # --- 4. TopK with k=1, 2, 3 (Known Answer #20) ---
    for k_val in [1, 2, 3]:
        res_topk_a = mine_patterns_tool.invoke({
            "category_name": "catA",
            "filtering_method": "tfidf",
            "algorithm": "topk",
            "k": k_val
        })
        assert res_topk_a["status"] == "success"
        assert res_topk_a["n_patterns"] == 1
        assert res_topk_a["patterns"][0]["pattern"] == ["alpha"]
        assert res_topk_a["patterns"][0]["support"] == 3

        res_topk_b = mine_patterns_tool.invoke({
            "category_name": "catB",
            "filtering_method": "tfidf",
            "algorithm": "topk",
            "k": k_val
        })
        assert res_topk_b["status"] == "success"
        assert res_topk_b["n_patterns"] == 1
        assert res_topk_b["patterns"][0]["pattern"] == ["gamma"]
        assert res_topk_b["patterns"][0]["support"] == 4

    # --- 5. MaxFPGrowth with min_sup=1 and 4 -> 0 patterns for both categories ---
    for min_sup_val in [1, 4]:
        for cat in ["catA", "catB"]:
            res_max = mine_patterns_tool.invoke({
                "category_name": cat,
                "filtering_method": "tfidf",
                "algorithm": "maxfpgrowth",
                "min_sup": min_sup_val
            })
            assert res_max["status"] == "success"
            assert res_max["n_patterns"] == 0
            assert res_max["patterns"] == []

    # --- 6. Invalid min_sup and k values validation ---
    for bad_val in [0, -1, 1.5, "one", True, False]:
        # Validate min_sup
        try:
            result = mine_patterns_tool.invoke({
                "category_name": "catA",
                "filtering_method": "tfidf",
                "algorithm": "fpgrowth",
                "min_sup": bad_val
            })
        except Exception as exc:
            assert "ValidationError" in type(exc).__name__, (
                f"Unexpected exception: {type(exc).__name__}: {exc}"
            )
        else:
            assert isinstance(result, dict)
            assert result.get("status") == "error"

        # Validate k
        try:
            result = mine_patterns_tool.invoke({
                "category_name": "catA",
                "filtering_method": "tfidf",
                "algorithm": "topk",
                "k": bad_val
            })
        except Exception as exc:
            assert "ValidationError" in type(exc).__name__, (
                f"Unexpected exception: {type(exc).__name__}: {exc}"
            )
        else:
            assert isinstance(result, dict)
            assert result.get("status") == "error"


def test_mine_patterns_tool_unicode_regression():
    """Regression test for non-ASCII characters (café, résumé) in mine_patterns_tool."""
    session = SessionState()
    session.dataframe = pd.DataFrame({
        "text": [
            "café résumé mocha",
            "café résumé espresso",
            "café latte",
            "espresso latte"
        ],
        "category_name": ["cat_unicode"] * 4
    })

    tools = make_tools(session)
    mine_patterns_tool = next(t for t in tools if t.name == "mine_patterns_tool")

    res = mine_patterns_tool.invoke({
        "category_name": "cat_unicode",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 2
    })

    assert res["status"] == "success"
    assert res["n_patterns"] > 0

    # Collect patterns as item tuples
    pattern_map = {tuple(sorted(p["pattern"])): p["support"] for p in res["patterns"]}

    # Non-ASCII terms 'café' and 'résumé' must be preserved and mined correctly
    assert ("café",) in pattern_map
    assert pattern_map[("café",)] == 3

    assert ("résumé",) in pattern_map
    assert pattern_map[("résumé",)] == 2

    assert ("café", "résumé") in pattern_map
    assert pattern_map[("café", "résumé")] == 2
