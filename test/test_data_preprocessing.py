import importlib
import os
import sys
from unittest.mock import patch

import pandas as pd
import pytest


SRC_PATH = os.path.join(os.path.dirname(__file__), "../src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)


# Avoid network calls at import time from nltk.download(...)
with patch("nltk.download", return_value=True):
    data_preprocessing = importlib.import_module("data_preprocessing")


def test_transform_text_success():
    with patch.object(
        data_preprocessing.nltk,
        "word_tokenize",
        return_value=["hello", ",", "this", "is", "running", "123", "!"],
    ):
        with patch.object(
            data_preprocessing.stopwords, "words", return_value=["this", "is"]
        ):
            transformed = data_preprocessing.transform_text(
                "Hello this is running 123!"
            )

    assert transformed == "hello run 123"


def test_preprocess_df_success():
    raw_df = pd.DataFrame(
        {
            "text": ["HELLO WORLD", "HELLO WORLD"],
            "target": ["spam", "spam"],
        }
    )

    with patch.object(
        data_preprocessing, "transform_text", new=lambda text: text.lower()
    ):
        processed = data_preprocessing.preprocess_df(raw_df.copy())

    assert len(processed) == 1
    assert processed.iloc[0]["text"] == "hello world"
    assert int(processed.iloc[0]["target"]) == 0


def test_preprocess_df_missing_column():
    raw_df = pd.DataFrame({"text": ["sample"]})
    with pytest.raises(KeyError):
        data_preprocessing.preprocess_df(raw_df)


def test_preprocess_df_generic_error():
    raw_df = pd.DataFrame({"text": ["sample"], "target": ["spam"]})
    with patch.object(
        data_preprocessing.LabelEncoder,
        "fit_transform",
        side_effect=Exception("encoding failed"),
    ):
        with pytest.raises(Exception):
            data_preprocessing.preprocess_df(raw_df)


def test_main_success():
    train_df = pd.DataFrame({"text": ["a"], "target": ["spam"]})
    test_df = pd.DataFrame({"text": ["b"], "target": ["ham"]})
    processed_train = pd.DataFrame({"text": ["a"], "target": [1]})
    processed_test = pd.DataFrame({"text": ["b"], "target": [0]})

    with patch.object(
        data_preprocessing.pd,
        "read_csv",
        side_effect=[train_df, test_df],
    ):
        with patch.object(
            data_preprocessing,
            "preprocess_df",
            side_effect=[processed_train, processed_test],
        ):
            with patch.object(data_preprocessing.os, "makedirs"):
                with patch.object(pd.DataFrame, "to_csv") as mock_to_csv:
                    data_preprocessing.main()
                    assert mock_to_csv.call_count == 2


def test_main_file_not_found():
    with patch.object(
        data_preprocessing.pd,
        "read_csv",
        side_effect=FileNotFoundError("missing"),
    ):
        data_preprocessing.main()


def test_main_empty_data_error():
    with patch.object(
        data_preprocessing.pd,
        "read_csv",
        side_effect=pd.errors.EmptyDataError("empty"),
    ):
        data_preprocessing.main()


def test_main_generic_exception_prints():
    with patch.object(
        data_preprocessing.pd,
        "read_csv",
        side_effect=Exception("boom"),
    ):
        with patch("builtins.print") as mock_print:
            data_preprocessing.main()
            assert mock_print.called
