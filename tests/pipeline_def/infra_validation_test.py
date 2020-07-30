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


def test_infra_validator_works(pipeline_def, monkeypatch):
    # Arrange
    mock_infra_validator = mock.Mock()
    monkeypatch.setattr(
        'fluent_tfx.pipeline_def.InfraValidator', mock_infra_validator)

    mock_example_gen = mock.MagicMock()
    pipeline_def.example_gen = mock_example_gen

    mock_trainer = mock.MagicMock()
    pipeline_def.trainer = mock_trainer

    mock_serving_spec = mock.Mock()

    # Act
    pipeline_def = pipeline_def.infra_validate(serving_spec=mock_serving_spec)
    _, kwargs = mock_infra_validator.call_args

    # Assert
    assert mock_infra_validator.called
    assert kwargs['serving_spec'] is mock_serving_spec
    assert 'validation_spec' not in kwargs
    assert 'request_spec' not in kwargs
    assert kwargs['model'] is mock_trainer.outputs['model']
    assert kwargs['examples'] is ExampleInputs.RAW_EXAMPLES(pipeline_def)
    assert pipeline_def.components['infra_validator'] is mock_infra_validator.return_value
    assert pipeline_def.infra_validator is mock_infra_validator.return_value


def test_infra_validator_passes_works_with_optional_args(pipeline_def, monkeypatch):
    # Arrange
    mock_infra_validator = mock.Mock()
    monkeypatch.setattr(
        'fluent_tfx.pipeline_def.InfraValidator', mock_infra_validator)

    mock_transform = mock.MagicMock()
    pipeline_def.transform = mock_transform

    mock_trainer = mock.MagicMock()
    pipeline_def.trainer = mock_trainer

    mock_serving_spec = mock.Mock()
    mock_validation_spec = mock.Mock()
    mock_request_spec = mock.Mock()

    # Act
    pipeline_def = pipeline_def.infra_validate(
        serving_spec=mock_serving_spec,
        validation_spec=mock_validation_spec,
        request_spec=mock_request_spec,
        example_input=ExampleInputs.PREPROCESSED_EXAMPLES)
    _, kwargs = mock_infra_validator.call_args

    # Assert
    assert mock_infra_validator.called
    assert kwargs['request_spec'] is mock_request_spec
    assert kwargs['validation_spec'] is mock_validation_spec
    assert kwargs['examples'] is ExampleInputs.PREPROCESSED_EXAMPLES(
        pipeline_def)
    assert pipeline_def.components['infra_validator'] is mock_infra_validator.return_value
    assert pipeline_def.infra_validator is mock_infra_validator.return_value


def test_infra_validator_can_use_cached_input(pipeline_def, monkeypatch):
    # Arrange
    mock_infra_validator = mock.Mock()
    monkeypatch.setattr(
        'fluent_tfx.pipeline_def.InfraValidator', mock_infra_validator)

    mock_cached_input = mock.Mock()
    pipeline_def.cached_example_input = mock_cached_input

    mock_trainer = mock.MagicMock()
    pipeline_def.trainer = mock_trainer

    mock_serving_spec = mock.Mock()

    # Act
    pipeline_def = pipeline_def.infra_validate(
        serving_spec=mock_serving_spec)
    _, kwargs = mock_infra_validator.call_args

    # Assert
    assert mock_infra_validator.called
    assert kwargs['examples'] is mock_cached_input
    assert pipeline_def.components['infra_validator'] is mock_infra_validator.return_value
    assert pipeline_def.infra_validator is mock_infra_validator.return_value
