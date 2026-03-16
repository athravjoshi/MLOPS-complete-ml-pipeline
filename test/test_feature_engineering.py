import importlib
import os
import sys
from unittest.mock import mock_open, patch

import pandas as pd
import pytest
import yaml


SRC_PATH = os.path.join(os.path.dirname(__file__), "../src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)


feature_engineering = importlib.import_module("feature_engineering")


def test_load_params_success(tmp_path):
    params_file = tmp_path / "params.yaml"
    params_file.write_text(
        "feature_engineering:\n  max_features: 25\n", encoding="utf-8"
    )

    params = feature_engineering.load_params(str(params_file))
    assert params["feature_engineering"]["max_features"] == 25


def test_load_params_file_not_found():
    with pytest.raises(FileNotFoundError):
        feature_engineering.load_params("does_not_exist.yaml")


def test_load_params_yaml_error():
    with patch("builtins.open", mock_open(read_data="bad: yaml")):
        with patch("feature_engineering.yaml.safe_load", side_effect=yaml.YAMLError):
            with pytest.raises(yaml.YAMLError):
                feature_engineering.load_params("params.yaml")


def test_load_params_generic_exception():
    with patch("builtins.open", side_effect=Exception("unexpected")):
        with pytest.raises(Exception):
            feature_engineering.load_params("params.yaml")


def test_load_data_success(tmp_path):
    csv_file = tmp_path / "data.csv"
    csv_file.write_text("text,target\nhello,\nworld,1\n", encoding="utf-8")

    df = feature_engineering.load_data(str(csv_file))
    assert isinstance(df, pd.DataFrame)
    assert df["text"].tolist() == ["hello", "world"]
    assert df["target"].tolist()[0] == ""
    assert float(df["target"].tolist()[1]) == 1.0


def test_load_data_parser_error():
    with patch.object(
        feature_engineering.pd,
        "read_csv",
        side_effect=pd.errors.ParserError("bad csv"),
    ):
        with pytest.raises(pd.errors.ParserError):
            feature_engineering.load_data("bad.csv")


def test_load_data_generic_exception():
    with patch.object(
        feature_engineering.pd, "read_csv", side_effect=Exception("unexpected")
    ):
        with pytest.raises(Exception):
            feature_engineering.load_data("data.csv")


def test_apply_tfidf_success():
    train_data = pd.DataFrame(
        {
            "text": ["buy now", "hello friend", "limited offer"],
            "target": [1, 0, 1],
        }
    )
    test_data = pd.DataFrame({"text": ["hello offer"], "target": [0]})

    train_df, test_df = feature_engineering.apply_tfidf(train_data, test_data, 10)

    assert "label" in train_df.columns
    assert "label" in test_df.columns
    assert len(train_df) == 3
    assert len(test_df) == 1


def test_apply_tfidf_error():
    train_data = pd.DataFrame({"body": ["text"], "target": [1]})
    test_data = pd.DataFrame({"text": ["text"], "target": [0]})
    with pytest.raises(Exception):
        feature_engineering.apply_tfidf(train_data, test_data, 10)


def test_save_data_success(tmp_path):
    output_file = tmp_path / "processed" / "train_tfidf.csv"
    df = pd.DataFrame({"f1": [0.1], "label": [1]})

    feature_engineering.save_data(df, str(output_file))
    assert output_file.exists()


def test_save_data_exception():
    df = pd.DataFrame({"f1": [0.1], "label": [1]})
    with patch.object(
        feature_engineering.os, "makedirs", side_effect=Exception("fail")
    ):
        with pytest.raises(Exception):
            feature_engineering.save_data(df, "out/file.csv")


def test_main_success():
    params = {"feature_engineering": {"max_features": 10}}
    train_data = pd.DataFrame({"text": ["hello"], "target": [0]})
    test_data = pd.DataFrame({"text": ["offer"], "target": [1]})
    train_df = pd.DataFrame({0: [0.5], "label": [0]})
    test_df = pd.DataFrame({0: [0.7], "label": [1]})

    with patch.object(feature_engineering, "load_params", return_value=params):
        with patch.object(
            feature_engineering, "load_data", side_effect=[train_data, test_data]
        ):
            with patch.object(
                feature_engineering, "apply_tfidf", return_value=(train_df, test_df)
            ):
                with patch.object(feature_engineering, "save_data") as mock_save_data:
                    feature_engineering.main()
                    assert mock_save_data.call_count == 2


def test_main_exception_prints():
    with patch.object(
        feature_engineering, "load_params", side_effect=Exception("fail")
    ):
        with patch("builtins.print") as mock_print:
            feature_engineering.main()
            assert mock_print.called
