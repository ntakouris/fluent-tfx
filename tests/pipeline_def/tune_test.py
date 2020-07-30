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


def test_pipeline_def_with_hyperparameters_works(pipeline_def, monkeypatch):
    # Arrange
    mock_with_hyperparameters = mock.Mock()
    monkeypatch.setattr(
        'fluent_tfx.pipeline_def.input_builders.with_hyperparameters', mock_with_hyperparameters)

    # Act
    pipeline_def = pipeline_def.with_hyperparameters('some_uri')
    _, kwargs = mock_with_hyperparameters.call_args

    # Assert
    assert mock_with_hyperparameters.called
    assert kwargs['uri'] == 'some_uri'
    assert pipeline_def.components['user_hyperparameters_importer'] is mock_with_hyperparameters.return_value
    assert pipeline_def.user_hyperparameters_importer is mock_with_hyperparameters.return_value


def test_pipeline_def_tune_works(pipeline_def, monkeypatch):
    # Arrange
    mock_schema = mock.MagicMock()
    pipeline_def.schema_gen = mock_schema

    mock_transform = mock.MagicMock()
    pipeline_def.transform = mock_transform

    mock_module = mock.Mock()

    mock_tuner = mock.Mock()
    monkeypatch.setattr('fluent_tfx.pipeline_def.Tuner', mock_tuner)

    # Act
    pipeline_def = pipeline_def.tune(
        module_file=mock_module, train_args=1, eval_args=2)

    _, kwargs = mock_tuner.call_args

    # Assert
    assert mock_tuner.called
    assert kwargs['module_file'] is mock_module
    assert kwargs['schema'] is SchemaInputs.SCHEMA_CHANNEL(pipeline_def)
    assert kwargs['train_args'] == 1
    assert kwargs['eval_args'] == 2
    assert kwargs['transform_graph'] == mock_transform.outputs['transform_graph']
    assert kwargs['examples'] is ExampleInputs.PREPROCESSED_EXAMPLES(
        pipeline_def)
    assert pipeline_def.components['tuner'] is mock_tuner.return_value
    assert pipeline_def.tuner is mock_tuner.return_value


def test_pipeline_def_tune_example_uses_example_channel_if_specified(pipeline_def, monkeypatch):
    # Arrange
    mock_schema = mock.MagicMock()
    pipeline_def.schema_gen = mock_schema

    mock_transform = mock.MagicMock()
    pipeline_def.transform = mock_transform

    mock_module = mock.Mock()

    example_in = mock.MagicMock()

    def example_in_fn(pipeline_instance):
        return example_in(pipeline_instance)

    mock_tuner = mock.Mock()
    monkeypatch.setattr('fluent_tfx.pipeline_def.Tuner', mock_tuner)

    # Act
    pipeline_def = pipeline_def.tune(
        module_file=mock_module, train_args=1, eval_args=2,
        example_input=example_in_fn)

    _, kwargs = mock_tuner.call_args
    args, _ = example_in.call_args
    # Assert
    assert example_in.called
    assert args[0] is pipeline_def
    assert pipeline_def.cached_example_input is example_in.return_value
    assert pipeline_def.components['tuner'] is mock_tuner.return_value
    assert pipeline_def.tuner is mock_tuner.return_value


def test_pipeline_def_tune_does_not_specify_absent_optional_arguments(pipeline_def, monkeypatch):
    # Arrange
    mock_example_gen = mock.MagicMock()
    pipeline_def.example_gen = mock_example_gen

    mock_schema = mock.MagicMock()
    pipeline_def.schema_gen = mock_schema

    pipeline_def.transform = None

    mock_tuner = mock.Mock()
    monkeypatch.setattr('fluent_tfx.pipeline_def.Tuner', mock_tuner)

    mock_module = mock.Mock()

    # Act
    pipeline_def = pipeline_def.tune(
        module_file=mock_module,
        example_input=ExampleInputs.RAW_EXAMPLES)

    _, kwargs = mock_tuner.call_args

    # Assert
    assert mock_tuner.called
    assert 'train_args' not in kwargs
    assert 'eval_args' not in kwargs
    assert 'transform_graph' not in kwargs
    assert kwargs['examples'] is ExampleInputs.RAW_EXAMPLES(pipeline_def)
    assert pipeline_def.cached_example_input is ExampleInputs.RAW_EXAMPLES(
        pipeline_def)
    assert pipeline_def.components['tuner'] is mock_tuner.return_value
    assert pipeline_def.tuner is mock_tuner.return_value
