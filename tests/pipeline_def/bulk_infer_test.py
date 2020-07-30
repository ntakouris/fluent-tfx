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

from tfx.proto import pusher_pb2


def test_bulk_infer_works(pipeline_def, monkeypatch):
    # Arrange
    mock_bulk_inferrer = mock.Mock()
    monkeypatch.setattr(
        'fluent_tfx.pipeline_def.BulkInferrer', mock_bulk_inferrer)

    mock_trainer = mock.MagicMock()
    pipeline_def.trainer = mock_trainer

    pipeline_def.model_evaluator = None

    mock_example_provider_component = mock.MagicMock(spec=BaseComponent)

    # Act
    pipeline_def = pipeline_def.bulk_infer(
        example_provider_component=mock_example_provider_component)
    _, kwargs = mock_bulk_inferrer.call_args

    # Assert
    assert mock_bulk_inferrer.called
    assert kwargs['model'] is mock_trainer.outputs['model']
    assert 'model_blessing' not in kwargs
    assert 'data_spec' not in kwargs
    assert 'model_spec' not in kwargs

    assert pipeline_def.components['bulk_inferrer_example_provider'] is mock_example_provider_component

    assert pipeline_def.components['bulk_inferrer'] is mock_bulk_inferrer.return_value
    assert pipeline_def.bulk_inferrer is mock_bulk_inferrer.return_value


def test_bulk_infer_works_with_optional_args(pipeline_def, monkeypatch):
    # Arrange
    mock_bulk_inferrer = mock.Mock()
    monkeypatch.setattr(
        'fluent_tfx.pipeline_def.BulkInferrer', mock_bulk_inferrer)

    mock_trainer = mock.MagicMock()
    pipeline_def.trainer = mock_trainer

    mock_evaluator = mock.MagicMock()
    pipeline_def.model_evaluator = mock_evaluator

    mock_example_provider_component = mock.MagicMock(spec=BaseComponent)

    mock_data_spec = mock.Mock()
    mock_model_spec = mock.Mock()

    # Act
    pipeline_def = pipeline_def.bulk_infer(
        example_provider_component=mock_example_provider_component,
        data_spec=mock_data_spec,
        model_spec=mock_model_spec)
    _, kwargs = mock_bulk_inferrer.call_args

    # Assert
    assert mock_bulk_inferrer.called
    assert kwargs['model'] is mock_trainer.outputs['model']
    assert kwargs['model_blessing'] is mock_evaluator.outputs['blessing']
    assert kwargs['data_spec'] is mock_data_spec
    assert kwargs['model_spec'] is mock_model_spec

    assert pipeline_def.components['bulk_inferrer_example_provider'] is mock_example_provider_component

    assert pipeline_def.components['bulk_inferrer'] is mock_bulk_inferrer.return_value
    assert pipeline_def.bulk_inferrer is mock_bulk_inferrer.return_value
