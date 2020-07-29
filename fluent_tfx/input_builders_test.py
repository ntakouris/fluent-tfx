import pytest
from unittest import mock

from fluent_tfx.input_builders import from_csv
from tfx.types import standard_artifacts


def test_from_csv_works(monkeypatch):
    # Arrange
    mock_csv_example_gen = mock.Mock()
    monkeypatch.setattr(
        'fluent_tfx.input_builders.CsvExampleGen', mock_csv_example_gen)

    # Act
    x = from_csv('some_uri', name='some_example',
                 input_config=2, output_config=3)
    _, kwargs = mock_csv_example_gen.call_args

    # Assert
    assert x is mock_csv_example_gen.return_value
    assert kwargs['input'].type is standard_artifacts.ExternalArtifact
    assert kwargs['input'].get()[0].uri == 'some_uri'
    assert kwargs['instance_name'] == 'some_example'
    assert kwargs['input_config'] == 2
    assert kwargs['output_config'] == 3
