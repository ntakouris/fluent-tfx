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


def test_pipeline_def_with_imported_schema_works(pipeline_def, monkeypatch):
    # Arrange
    mock_with_imported_schema = mock.Mock()
    monkeypatch.setattr(
        'fluent_tfx.pipeline_def.input_builders.with_imported_schema', mock_with_imported_schema)

    # Act
    pipeline_def = pipeline_def.with_imported_schema(uri='schema_uri')
    _, kwargs = mock_with_imported_schema.call_args

    # Assert
    assert mock_with_imported_schema.called
    assert kwargs['uri'] == 'schema_uri'
    assert pipeline_def.components['user_schema_importer'] is mock_with_imported_schema.return_value
    assert pipeline_def.user_schema_importer is mock_with_imported_schema.return_value


def test_pipeline_def_infer_schema_works(pipeline_def, monkeypatch):
    # Arrange
    mock_statistics_gen = mock.MagicMock()
    pipeline_def.statistics_gen = mock_statistics_gen

    mock_schema_gen = mock.Mock()
    monkeypatch.setattr(
        'fluent_tfx.pipeline_def.SchemaGen', mock_schema_gen)

    mock_infer_feature_shape = mock.Mock()

    # Act
    pipeline_def = pipeline_def.infer_schema(
        infer_feature_shape=mock_infer_feature_shape)
    _, kwargs = mock_schema_gen.call_args

    # Assert
    assert mock_schema_gen.called
    assert kwargs['statistics'] is mock_statistics_gen.outputs['statistics']
    assert kwargs['infer_feature_shape'] is mock_infer_feature_shape
    assert pipeline_def.components['schema_gen'] is mock_schema_gen.return_value
    assert pipeline_def.schema_gen is mock_schema_gen.return_value
