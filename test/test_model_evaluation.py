import importlib
import json
import os
import pickle
import sys
import types
from unittest.mock import mock_open, patch

import numpy as np
import pandas as pd
import pytest
import yaml


SRC_PATH = os.path.join(os.path.dirname(__file__), "../src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)


# Provide a minimal dvclive module so import works even if dvclive is not installed.
if "dvclive" not in sys.modules:
    dvclive_stub = types.ModuleType("dvclive")

    class _LiveStub:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def log_metric(self, *args, **kwargs):
            return None

        def log_params(self, *args, **kwargs):
            return None

    dvclive_stub.Live = _LiveStub
    sys.modules["dvclive"] = dvclive_stub


model_evaluation = importlib.import_module("model_evaluation")


class DummyClassifier:
    def predict(self, X):
        return np.array([0, 1])

    def predict_proba(self, X):
        return np.array([[0.9, 0.1], [0.1, 0.9]])


def test_load_params_success(tmp_path):
    params_file = tmp_path / "params.yaml"
    params_file.write_text("experiment:\n  name: test\n", encoding="utf-8")

    params = model_evaluation.load_params(str(params_file))
    assert params["experiment"]["name"] == "test"


def test_load_params_file_not_found():
    with pytest.raises(FileNotFoundError):
        model_evaluation.load_params("missing.yaml")


def test_load_params_yaml_error():
    with patch("builtins.open", mock_open(read_data="bad")):
        with patch("model_evaluation.yaml.safe_load", side_effect=yaml.YAMLError):
            with pytest.raises(yaml.YAMLError):
                model_evaluation.load_params("params.yaml")


def test_load_params_generic_exception():
    with patch("builtins.open", side_effect=Exception("unexpected")):
        with pytest.raises(Exception):
            model_evaluation.load_params("params.yaml")


def test_load_model_success(tmp_path):
    model_file = tmp_path / "model.pkl"
    with open(model_file, "wb") as f:
        pickle.dump({"model": "ok"}, f)

    model = model_evaluation.load_model(str(model_file))
    assert model["model"] == "ok"


def test_load_model_file_not_found():
    with pytest.raises(FileNotFoundError):
        model_evaluation.load_model("missing.pkl")


def test_load_model_generic_exception():
    with patch("builtins.open", side_effect=Exception("unexpected")):
        with pytest.raises(Exception):
            model_evaluation.load_model("model.pkl")


def test_load_data_success(tmp_path):
    csv_file = tmp_path / "test.csv"
    csv_file.write_text("f1,f2,label\n1,0,1\n0,1,0\n", encoding="utf-8")

    df = model_evaluation.load_data(str(csv_file))
    assert df.shape == (2, 3)


def test_load_data_parser_error():
    with patch.object(
        model_evaluation.pd,
        "read_csv",
        side_effect=pd.errors.ParserError("bad csv"),
    ):
        with pytest.raises(pd.errors.ParserError):
            model_evaluation.load_data("bad.csv")


def test_load_data_generic_exception():
    with patch.object(
        model_evaluation.pd, "read_csv", side_effect=Exception("unexpected")
    ):
        with pytest.raises(Exception):
            model_evaluation.load_data("data.csv")


def test_evaluate_model_success():
    clf = DummyClassifier()
    X_test = np.array([[0.1, 0.9], [0.8, 0.2]])
    y_test = np.array([0, 1])

    metrics = model_evaluation.evaluate_model(clf, X_test, y_test)

    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["auc"] == 1.0


def test_evaluate_model_exception():
    class BrokenClassifier:
        def predict(self, X):
            raise Exception("prediction failed")

    with pytest.raises(Exception):
        model_evaluation.evaluate_model(
            BrokenClassifier(), np.array([[1, 2]]), np.array([1])
        )


def test_save_metrics_success(tmp_path):
    metrics_file = tmp_path / "reports" / "metrics.json"
    metrics = {"accuracy": 0.95}

    model_evaluation.save_metrics(metrics, str(metrics_file))
    assert metrics_file.exists()

    saved = json.loads(metrics_file.read_text(encoding="utf-8"))
    assert saved["accuracy"] == 0.95


def test_save_metrics_exception():
    with patch.object(
        model_evaluation.os, "makedirs", side_effect=Exception("mkdir failed")
    ):
        with pytest.raises(Exception):
            model_evaluation.save_metrics({"accuracy": 1.0}, "reports/metrics.json")


class FakeLive:
    def __init__(self, *args, **kwargs):
        self.metrics = []
        self.params = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def log_metric(self, name, value):
        self.metrics.append((name, value))

    def log_params(self, params):
        self.params = params


def test_main_success():
    params = {"run": "ok"}
    clf = DummyClassifier()
    test_data = pd.DataFrame({"f1": [0.1, 0.8], "f2": [0.9, 0.2], "label": [0, 1]})
    metrics = {"accuracy": 1.0, "precision": 1.0, "recall": 1.0, "auc": 1.0}

    with patch.object(model_evaluation, "load_params", return_value=params):
        with patch.object(model_evaluation, "load_model", return_value=clf):
            with patch.object(model_evaluation, "load_data", return_value=test_data):
                with patch.object(
                    model_evaluation, "evaluate_model", return_value=metrics
                ):
                    with patch.object(model_evaluation, "Live", FakeLive):
                        with patch.object(
                            model_evaluation, "save_metrics"
                        ) as mock_save:
                            model_evaluation.main()
                            mock_save.assert_called_once_with(
                                metrics, "reports/metrics.json"
                            )


def test_main_exception_prints():
    with patch.object(model_evaluation, "load_params", side_effect=Exception("fail")):
        with patch("builtins.print") as mock_print:
            model_evaluation.main()
            assert mock_print.called
