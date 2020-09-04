import pytest
from unittest import mock

from fluent_tfx.input_builders import from_csv, from_tfrecord, from_bigquery, \
    with_base_model, with_hyperparameters, with_imported_schema, get_latest_blessed_model_resolver
from tfx.types import standard_artifacts, Channel
from tfx.dsl.experimental.latest_blessed_model_resolver import LatestBlessedModelResolver


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
    assert mock_csv_example_gen.called
    assert x is mock_csv_example_gen.return_value
    assert type(kwargs['input_base']) is str
    assert kwargs['input_base'] == 'some_uri'
    assert kwargs['instance_name'] == 'some_example'
    assert kwargs['input_config'] == 2
    assert kwargs['output_config'] == 3


def test_from_csv_does_not_apply_optional_arguments_to_example_gen(monkeypatch):
    # Arrange
    mock_csv_example_gen = mock.Mock()
    monkeypatch.setattr(
        'fluent_tfx.input_builders.CsvExampleGen', mock_csv_example_gen)

    # Act
    x = from_csv('some_uri')
    _, kwargs = mock_csv_example_gen.call_args

    # Assert
    assert mock_csv_example_gen.called
    assert 'instance_name' not in kwargs
    assert 'input_config' not in kwargs
    assert 'output_config' not in kwargs


def test_from_tfrecord_works(monkeypatch):
    # Arrange
    mock_import_example_gen = mock.Mock()
    monkeypatch.setattr(
        'fluent_tfx.input_builders.ImportExampleGen', mock_import_example_gen)

    # Act
    x = from_tfrecord('some_uri', name='some_example',
                      input_config=2, output_config=3)
    _, kwargs = mock_import_example_gen.call_args

    # Assert
    assert mock_import_example_gen.called
    assert x is mock_import_example_gen.return_value
    assert type(kwargs['input_base']) is str
    assert kwargs['input_base'] == 'some_uri'
    assert kwargs['instance_name'] == 'some_example'
    assert kwargs['input_config'] == 2
    assert kwargs['output_config'] == 3


def test_from_tfrecord_does_not_apply_optional_arguments_to_example_gen(monkeypatch):
    # Arrange
    mock_import_example_gen = mock.Mock()
    monkeypatch.setattr(
        'fluent_tfx.input_builders.ImportExampleGen', mock_import_example_gen)

    # Act
    x = from_tfrecord('some_uri')
    _, kwargs = mock_import_example_gen.call_args

    # Assert
    assert mock_import_example_gen.called
    assert 'instance_name' not in kwargs
    assert 'input_config' not in kwargs
    assert 'output_config' not in kwargs


def test_from_bigquery_works(monkeypatch):
    # Arrange
    mock_bigquery_example_gen = mock.Mock()
    monkeypatch.setattr(
        'fluent_tfx.input_builders.BigQueryExampleGen', mock_bigquery_example_gen)

    # Act
    x = from_bigquery('some_query', name='some_example',
                      input_config=2, output_config=3)
    _, kwargs = mock_bigquery_example_gen.call_args

    # Assert
    assert mock_bigquery_example_gen.called
    assert x is mock_bigquery_example_gen.return_value
    assert kwargs['query'] == 'some_query'
    assert kwargs['instance_name'] == 'some_example'
    assert kwargs['input_config'] == 2
    assert kwargs['output_config'] == 3


def test_from_bigquery_does_not_apply_optional_arguments_to_example_gen(monkeypatch):
    # Arrange
    mock_bigquery_example_gen = mock.Mock()
    monkeypatch.setattr(
        'fluent_tfx.input_builders.BigQueryExampleGen', mock_bigquery_example_gen)

    # Act
    x = from_bigquery('query')
    _, kwargs = mock_bigquery_example_gen.call_args

    # Assert
    assert mock_bigquery_example_gen.called
    assert 'instance_name' not in kwargs
    assert 'input_config' not in kwargs
    assert 'output_config' not in kwargs


def test_with_imported_schema_works(monkeypatch):
    # Arrange
    mock_importer_node = mock.Mock()
    monkeypatch.setattr(
        'fluent_tfx.input_builders.ImporterNode', mock_importer_node)

    # Act
    x = with_imported_schema('some_uri')
    _, kwargs = mock_importer_node.call_args

    # Assert
    assert mock_importer_node.called
    assert kwargs['instance_name'] == 'with_imported_schema'
    assert kwargs['source_uri'] == 'some_uri'
    assert kwargs['artifact_type'] is standard_artifacts.Schema


def test_with_base_model_works(monkeypatch):
    # Arrange
    mock_importer_node = mock.Mock()
    monkeypatch.setattr(
        'fluent_tfx.input_builders.ImporterNode', mock_importer_node)

    # Act
    x = with_base_model('some_uri')
    _, kwargs = mock_importer_node.call_args

    # Assert
    assert mock_importer_node.called
    assert kwargs['instance_name'] == 'with_base_model'
    assert kwargs['source_uri'] == 'some_uri'
    assert kwargs['artifact_type'] is standard_artifacts.Model


def test_with_hyperparameters_works(monkeypatch):
    # Arrange
    mock_importer_node = mock.Mock()
    monkeypatch.setattr(
        'fluent_tfx.input_builders.ImporterNode', mock_importer_node)

    # Act
    x = with_hyperparameters('some_uri')
    _, kwargs = mock_importer_node.call_args

    # Assert
    assert mock_importer_node.called
    assert kwargs['instance_name'] == 'with_hyperparameters'
    assert kwargs['source_uri'] == 'some_uri'
    assert kwargs['artifact_type'] is standard_artifacts.HyperParameters


def test_get_latest_blessed_model_resolver_works(monkeypatch):
    # Arrange
    mock_resolver_node = mock.Mock()
    monkeypatch.setattr(
        'fluent_tfx.input_builders.ResolverNode', mock_resolver_node)

    # Act
    x = get_latest_blessed_model_resolver()
    _, kwargs = mock_resolver_node.call_args

    # Assert
    assert mock_resolver_node.called
    assert kwargs['instance_name'] == 'latest_blessed_model_resolver'
    assert kwargs['resolver_class'] is LatestBlessedModelResolver
    assert isinstance(kwargs['model'], Channel)
    assert isinstance(kwargs['model_blessing'], Channel)
