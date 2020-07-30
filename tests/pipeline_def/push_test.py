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


def test_push_to_works(pipeline_def, monkeypatch):
    # Arrange
    mock_pusher = mock.Mock()
    monkeypatch.setattr('fluent_tfx.pipeline_def.Pusher', mock_pusher)

    mock_trainer = mock.MagicMock()
    pipeline_def.trainer = mock_trainer

    mock_push_dest = mock.Mock()

    pipeline_def.model_evaluator = None
    # Act
    pipeline_def = pipeline_def.push_to(push_destination=mock_push_dest)
    _, kwargs = mock_pusher.call_args

    # Assert
    assert mock_pusher.called
    assert kwargs['model'] is mock_trainer.outputs['model']
    assert kwargs['push_destination'] is mock_push_dest
    assert 'model_blessing' not in kwargs
    assert 'custom_config' not in kwargs
    assert 'custom_executor_spec' not in kwargs
    assert 'infra_blessing' not in kwargs
    assert pipeline_def.components['pusher'] is mock_pusher.return_value
    assert pipeline_def.pusher is mock_pusher.return_value


def test_push_to_works_with_optional_args(pipeline_def, monkeypatch):
    # Arrange
    mock_pusher = mock.Mock()
    monkeypatch.setattr('fluent_tfx.pipeline_def.Pusher', mock_pusher)

    mock_trainer = mock.MagicMock()
    pipeline_def.trainer = mock_trainer

    mock_evaluator = mock.MagicMock()
    pipeline_def.model_evaluator = mock_evaluator

    mock_infra_validator = mock.MagicMock()
    pipeline_def.infra_validator = mock_infra_validator

    mock_config = mock.Mock()
    mock_exec = mock.Mock()

    # Act
    pipeline_def = pipeline_def.push_to(
        relative_push_uri='some_uri',
        custom_config=mock_config,
        custom_executor_spec=mock_exec)
    _, kwargs = mock_pusher.call_args

    # Assert
    assert mock_pusher.called
    assert kwargs['model'] is mock_trainer.outputs['model']
    assert kwargs['model_blessing'] is mock_evaluator.outputs['blessing']
    assert kwargs['custom_config'] is mock_config
    assert kwargs['custom_executor_spec'] is mock_exec
    assert kwargs['infra_blessing'] is mock_infra_validator.outputs['blessing']

    assert isinstance(kwargs['push_destination'], pusher_pb2.PushDestination)
    assert(
        kwargs['push_destination'].filesystem.base_directory.endswith('some_uri'))

    assert pipeline_def.components['pusher'] is mock_pusher.return_value
    assert pipeline_def.pusher is mock_pusher.return_value
