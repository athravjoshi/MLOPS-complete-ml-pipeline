import importlib
import os
import sys
from unittest.mock import mock_open, patch

import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.ensemble import RandomForestClassifier


SRC_PATH = os.path.join(os.path.dirname(__file__), "../src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)


model_building = importlib.import_module("model_building")


def test_load_params_success(tmp_path):
    params_file = tmp_path / "params.yaml"
    params_file.write_text(
        "model_building:\n  n_estimators: 20\n  random_state: 42\n",
        encoding="utf-8",
    )

    params = model_building.load_params(str(params_file))
    assert params["model_building"]["n_estimators"] == 20


def test_load_params_file_not_found():
    with pytest.raises(FileNotFoundError):
        model_building.load_params("missing.yaml")


def test_load_params_yaml_error():
    with patch("builtins.open", mock_open(read_data="bad")):
        with patch("model_building.yaml.safe_load", side_effect=yaml.YAMLError):
            with pytest.raises(yaml.YAMLError):
                model_building.load_params("params.yaml")


def test_load_params_generic_exception():
    with patch("builtins.open", side_effect=Exception("unexpected")):
        with pytest.raises(Exception):
            model_building.load_params("params.yaml")


def test_load_data_success(tmp_path):
    csv_file = tmp_path / "train.csv"
    csv_file.write_text("f1,f2,label\n1,0,1\n0,1,0\n", encoding="utf-8")

    df = model_building.load_data(str(csv_file))
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (2, 3)


def test_load_data_parser_error():
    with patch.object(
        model_building.pd,
        "read_csv",
        side_effect=pd.errors.ParserError("bad csv"),
    ):
        with pytest.raises(pd.errors.ParserError):
            model_building.load_data("bad.csv")


def test_load_data_file_not_found():
    with patch.object(model_building.pd, "read_csv", side_effect=FileNotFoundError):
        with pytest.raises(FileNotFoundError):
            model_building.load_data("missing.csv")


def test_load_data_generic_exception():
    with patch.object(
        model_building.pd, "read_csv", side_effect=Exception("unexpected")
    ):
        with pytest.raises(Exception):
            model_building.load_data("data.csv")


def test_train_model_success():
    X_train = np.array([[0.1, 0.9], [0.9, 0.1], [0.2, 0.8], [0.8, 0.2]])
    y_train = np.array([0, 1, 0, 1])
    params = {"n_estimators": 10, "random_state": 7}

    clf = model_building.train_model(X_train, y_train, params)

    assert isinstance(clf, RandomForestClassifier)
    assert clf.n_estimators == 10


def test_train_model_shape_mismatch():
    X_train = np.array([[0.1, 0.2], [0.3, 0.4]])
    y_train = np.array([1])
    params = {"n_estimators": 10, "random_state": 7}

    with pytest.raises(ValueError):
        model_building.train_model(X_train, y_train, params)


def test_train_model_fit_failure():
    X_train = np.array([[0.1, 0.9], [0.9, 0.1]])
    y_train = np.array([0, 1])
    params = {"n_estimators": 10, "random_state": 7}

    with patch.object(
        model_building.RandomForestClassifier,
        "fit",
        side_effect=Exception("fit failed"),
    ):
        with pytest.raises(Exception):
            model_building.train_model(X_train, y_train, params)


def test_save_model_success(tmp_path):
    model_file = tmp_path / "models" / "model.pkl"
    model = {"artifact": "ok"}

    model_building.save_model(model, str(model_file))
    assert model_file.exists()


def test_save_model_exception():
    with patch("builtins.open", side_effect=Exception("write failed")):
        with pytest.raises(Exception):
            model_building.save_model({"x": 1}, "models/model.pkl")


def test_main_success():
    params = {"model_building": {"n_estimators": 5, "random_state": 1}}
    train_data = pd.DataFrame({"f1": [0.1, 0.2], "f2": [0.9, 0.8], "label": [0, 1]})

    with patch.object(model_building, "load_params", return_value=params):
        with patch.object(model_building, "load_data", return_value=train_data):
            with patch.object(
                model_building, "train_model", return_value="trained-model"
            ):
                with patch.object(model_building, "save_model") as mock_save_model:
                    model_building.main()
                    mock_save_model.assert_called_once_with(
                        "trained-model", "models/model.pkl"
                    )


def test_main_exception_prints():
    with patch.object(model_building, "load_params", side_effect=Exception("fail")):
        with patch("builtins.print") as mock_print:
            model_building.main()
            assert mock_print.called
