import pytest
import os
from unittest import mock

from tfx.components.base.base_component import BaseComponent

from fluent_tfx import PipelineDef, \
    ExampleInputs, SchemaInputs, HyperParameterInputs

from fluent_tfx.pipeline_def import build_step
from tfx.components.base import executor_spec
from tfx.extensions.google_cloud_ai_platform.trainer \
    import executor as ai_platform_trainer_executor


def test_pipeline_def_from_csv_works(pipeline_def, monkeypatch):
    # Arrange
    mock_from_csv = mock.Mock()
    monkeypatch.setattr(
        'fluent_tfx.pipeline_def.input_builders.from_csv', mock_from_csv)

    # Act
    pipeline_def = pipeline_def.from_csv(
        uri='some_uri', input_config=1, output_config=2)
    _, kwargs = mock_from_csv.call_args

    # Assert
    assert mock_from_csv.called
    assert kwargs['uri'] == 'some_uri'
    assert kwargs['input_config'] == 1
    assert kwargs['output_config'] == 2

    assert pipeline_def.components['example_gen'] is mock_from_csv.return_value
    assert pipeline_def.example_gen is mock_from_csv.return_value


def test_pipeline_def_from_tfrecord_works(pipeline_def, monkeypatch):
    # Arrange
    mock_from_tfrecord = mock.Mock()
    monkeypatch.setattr(
        'fluent_tfx.pipeline_def.input_builders.from_tfrecord', mock_from_tfrecord)

    # Act
    pipeline_def = pipeline_def.from_tfrecord(
        uri='some_uri', input_config=1, output_config=2)
    _, kwargs = mock_from_tfrecord.call_args

    # Assert
    assert mock_from_tfrecord.called
    assert kwargs['uri'] == 'some_uri'
    assert kwargs['input_config'] == 1
    assert kwargs['output_config'] == 2

    assert pipeline_def.components['example_gen'] is mock_from_tfrecord.return_value
    assert pipeline_def.example_gen is mock_from_tfrecord.return_value


def test_pipeline_def_from_bigquery_works(pipeline_def, monkeypatch):
    # Arrange
    mock_from_bigquery = mock.Mock()
    monkeypatch.setattr(
        'fluent_tfx.pipeline_def.input_builders.from_bigquery', mock_from_bigquery)

    # Act
    pipeline_def = pipeline_def.from_bigquery(
        query='some_query', input_config=1, output_config=2)
    _, kwargs = mock_from_bigquery.call_args

    # Assert
    assert mock_from_bigquery.called
    assert kwargs['query'] == 'some_query'
    assert kwargs['input_config'] == 1
    assert kwargs['output_config'] == 2

    assert pipeline_def.components['example_gen'] is mock_from_bigquery.return_value
    assert pipeline_def.example_gen is mock_from_bigquery.return_value


def test_pipeline_def_from_custom_example_gen_component_works(pipeline_def):
    # Arrange
    mock_base_component = mock.MagicMock(spec=BaseComponent)

    # Act
    pipeline_def = pipeline_def.from_custom_example_gen_component(
        mock_base_component)

    # Assert
    assert pipeline_def.components['example_gen'] is mock_base_component
    assert pipeline_def.example_gen is mock_base_component
