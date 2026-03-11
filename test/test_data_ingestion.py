import pytest
import pandas as pd
import yaml
from unittest.mock import patch, mock_open
import os
import sys


# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

from data_ingestion import (
    load_params,
    load_data,
    preprocess_data,
    save_data,
    main,
)


class TestLoadParams:
    """Test cases for load_params function."""

    def test_load_params_success(self):
        """Test successful loading of parameters from YAML."""
        mock_params = {"data_ingestion": {"test_size": 0.2}}
        with patch("builtins.open", mock_open(read_data=yaml.dump(mock_params))):
            result = load_params("params.yaml")
            assert result == mock_params

    def test_load_params_file_not_found(self):
        """Test FileNotFoundError when params file doesn't exist."""
        with patch("builtins.open", side_effect=FileNotFoundError):
            with pytest.raises(FileNotFoundError):
                load_params("nonexistent.yaml")

    def test_load_params_yaml_error(self):
        """Test YAMLError when YAML is malformed."""
        with patch("builtins.open", mock_open(read_data="invalid: yaml: content:")):
            with patch("yaml.safe_load", side_effect=yaml.YAMLError):
                with pytest.raises(yaml.YAMLError):
                    load_params("params.yaml")

    def test_load_params_empty_file(self):
        """Test ValueError when params file is empty."""
        with patch("builtins.open", mock_open(read_data="")):
            with patch("yaml.safe_load", return_value=None):
                with pytest.raises(ValueError):
                    load_params("params.yaml")

    def test_load_params_not_a_dict(self):
        """Test ValueError when YAML content is not a dict."""
        with patch("builtins.open", mock_open(read_data="- item1\n- item2")):
            with patch("yaml.safe_load", return_value=["item1", "item2"]):
                with pytest.raises(ValueError):
                    load_params("params.yaml")

    def test_load_params_generic_exception(self):
        """Test generic exception handling."""
        with patch("builtins.open", side_effect=Exception("Generic error")):
            with pytest.raises(Exception):
                load_params("params.yaml")


class TestLoadData:
    """Test cases for load_data function."""

    def test_load_data_success(self):
        """Test successful loading of data from CSV."""
        mock_data = {"col1": [1, 2], "col2": [3, 4]}
        mock_df = pd.DataFrame(mock_data)
        with patch("pandas.read_csv", return_value=mock_df):
            result = load_data("data.csv")
            assert isinstance(result, pd.DataFrame)
            assert list(result.columns) == ["col1", "col2"]

    def test_load_data_parser_error(self):
        """Test ParserError when CSV is malformed."""
        with patch("pandas.read_csv", side_effect=pd.errors.ParserError):
            with pytest.raises(pd.errors.ParserError):
                load_data("malformed.csv")

    def test_load_data_generic_exception(self):
        """Test generic exception handling."""
        with patch("pandas.read_csv", side_effect=Exception("Generic error")):
            with pytest.raises(Exception):
                load_data("data.csv")


class TestPreprocessData:
    """Test cases for preprocess_data function."""

    def test_preprocess_data_success(self):
        """Test successful data preprocessing."""
        data = {
            "v1": ["spam", "ham"],
            "v2": ["text1", "text2"],
            "Unnamed: 2": [1, 2],
            "Unnamed: 3": [3, 4],
            "Unnamed: 4": [5, 6],
        }
        df = pd.DataFrame(data)
        result = preprocess_data(df)

        # Check columns are renamed
        assert "target" in result.columns
        assert "text" in result.columns
        # Check unnamed columns are dropped
        assert "Unnamed: 2" not in result.columns
        assert "Unnamed: 3" not in result.columns
        assert "Unnamed: 4" not in result.columns

    def test_preprocess_data_missing_columns(self):
        """Test KeyError when expected columns are missing."""
        df = pd.DataFrame({"colA": [1, 2], "colB": [3, 4]})
        with pytest.raises(KeyError):
            preprocess_data(df)

    def test_preprocess_data_generic_exception(self):
        """Test generic exception handling."""
        df = pd.DataFrame({"v1": [1, 2], "v2": [3, 4]})
        with patch.object(df, "drop", side_effect=Exception("Generic error")):
            with pytest.raises(Exception):
                preprocess_data(df)


class TestSaveData:
    """Test cases for save_data function."""

    def test_save_data_success(self):
        """Test successful saving of train and test data."""
        train_data = pd.DataFrame({"col1": [1, 2]})
        test_data = pd.DataFrame({"col1": [3, 4]})

        with patch("os.makedirs"):
            with patch.object(pd.DataFrame, "to_csv") as mock_to_csv:
                save_data(train_data, test_data, "./data")
                assert mock_to_csv.call_count == 2

    def test_save_data_directory_creation(self):
        """Test that raw data directory is created."""
        train_data = pd.DataFrame({"col1": [1, 2]})
        test_data = pd.DataFrame({"col1": [3, 4]})

        with patch("os.makedirs") as mock_makedirs:
            with patch.object(pd.DataFrame, "to_csv"):
                save_data(train_data, test_data, "./data")
                mock_makedirs.assert_called()

    def test_save_data_generic_exception(self):
        """Test generic exception handling."""
        train_data = pd.DataFrame({"col1": [1, 2]})
        test_data = pd.DataFrame({"col1": [3, 4]})

        with patch("os.makedirs", side_effect=Exception("Generic error")):
            with pytest.raises(Exception):
                save_data(train_data, test_data, "./data")


class TestMain:
    """Test cases for main function."""

    def test_main_success(self):
        """Test successful execution of main function."""
        mock_params = {"data_ingestion": {"test_size": 0.2}}
        mock_df = pd.DataFrame(
            {
                "v1": ["spam", "ham", "spam"],
                "v2": ["text1", "text2", "text3"],
                "Unnamed: 2": [1, 2, 3],
                "Unnamed: 3": [4, 5, 6],
                "Unnamed: 4": [7, 8, 9],
            }
        )

        with patch("builtins.open", mock_open(read_data=yaml.dump(mock_params))):
            with patch("yaml.safe_load", return_value=mock_params):
                with patch("pandas.read_csv", return_value=mock_df):
                    with patch("os.makedirs"):
                        with patch.object(pd.DataFrame, "to_csv"):
                            # Should not raise any exception
                            main()

    def test_main_load_params_failure(self):
        """Test main handles load_params failure."""
        with patch("data_ingestion.load_params", side_effect=FileNotFoundError):
            with patch("builtins.print"):
                main()

    def test_main_load_data_failure(self):
        """Test main handles load_data failure."""
        mock_params = {"data_ingestion": {"test_size": 0.2}}

        with patch("data_ingestion.load_params", return_value=mock_params):
            with patch(
                "data_ingestion.load_data", side_effect=Exception("Data load error")
            ):
                with patch("builtins.print"):
                    main()

    def test_main_preprocess_data_failure(self):
        """Test main handles preprocess_data failure."""
        mock_params = {"data_ingestion": {"test_size": 0.2}}
        mock_df = pd.DataFrame({"col1": [1, 2]})

        with patch("data_ingestion.load_params", return_value=mock_params):
            with patch("data_ingestion.load_data", return_value=mock_df):
                with patch(
                    "data_ingestion.preprocess_data",
                    side_effect=Exception("Preprocessing error"),
                ):
                    with patch("builtins.print"):
                        main()

    def test_main_save_data_failure(self):
        """Test main handles save_data failure."""
        mock_params = {"data_ingestion": {"test_size": 0.2}}
        mock_df = pd.DataFrame(
            {
                "v1": ["spam", "ham"],
                "v2": ["text1", "text2"],
                "Unnamed: 2": [1, 2],
                "Unnamed: 3": [3, 4],
                "Unnamed: 4": [5, 6],
            }
        )

        with patch("data_ingestion.load_params", return_value=mock_params):
            with patch("data_ingestion.load_data", return_value=mock_df):
                with patch("data_ingestion.preprocess_data", return_value=mock_df):
                    with patch(
                        "data_ingestion.save_data", side_effect=Exception("Save error")
                    ):
                        with patch("builtins.print"):
                            main()


class TestIntegration:
    """Integration tests for the complete data ingestion pipeline."""

    def test_end_to_end_pipeline(self):
        """Test the complete pipeline from params to saved data."""
        # Create sample data
        sample_data = {
            "v1": ["spam"] * 70 + ["ham"] * 30,
            "v2": [f"message_{i}" for i in range(100)],
            "Unnamed: 2": list(range(100)),
            "Unnamed: 3": list(range(100, 200)),
            "Unnamed: 4": list(range(200, 300)),
        }
        mock_df = pd.DataFrame(sample_data)
        mock_params = {"data_ingestion": {"test_size": 0.2}}

        with patch("builtins.open", mock_open(read_data=yaml.dump(mock_params))):
            with patch("yaml.safe_load", return_value=mock_params):
                with patch("pandas.read_csv", return_value=mock_df):
                    with patch("os.makedirs"):
                        with patch.object(pd.DataFrame, "to_csv") as mock_to_csv:
                            main()
                            # Verify to_csv was called twice (train and test)
                            assert mock_to_csv.call_count == 2


# Run tests with: pytest test_data_ingestion.py -v
