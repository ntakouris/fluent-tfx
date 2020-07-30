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


def test_pipeline_def_with_base_model_works(pipeline_def, monkeypatch):
    # Arrange
    mock_with_base_model = mock.Mock()
    monkeypatch.setattr(
        'fluent_tfx.pipeline_def.input_builders.with_base_model', mock_with_base_model)

    # Act
    pipeline_def = pipeline_def.with_base_model('some_uri')
    _, kwargs = mock_with_base_model.call_args

    # Assert
    assert mock_with_base_model.called
    assert kwargs['uri'] == 'some_uri'
    assert pipeline_def.components['train_base_model'] is mock_with_base_model.return_value
    assert pipeline_def.train_base_model is mock_with_base_model.return_value


def test_pipeline_def_train_minimally_works(pipeline_def, monkeypatch):
    # Arrange
    mock_schema = mock.MagicMock()
    pipeline_def.schema_gen = mock_schema

    mock_transform = mock.MagicMock()
    pipeline_def.transform = mock_transform

    pipeline_def.cached_example_input = None

    pipeline_def.tuner = None
    pipeline_def.user_hyperparameters_importer = None

    mock_trainer = mock.Mock()
    monkeypatch.setattr('fluent_tfx.pipeline_def.Trainer', mock_trainer)

    mock_module = mock.Mock()

    # Act
    pipeline_def = pipeline_def.train(module_file=mock_module)
    _, kwargs = mock_trainer.call_args

    # Assert
    assert mock_trainer.called
    assert 'train_args' not in kwargs
    assert 'eval_args' not in kwargs
    assert kwargs['schema'] is SchemaInputs.SCHEMA_CHANNEL(pipeline_def)
    assert kwargs['transform_graph'] is mock_transform.outputs['transform_graph']
    assert 'custom_config' not in kwargs
    assert 'hyperparameters' not in kwargs
    assert isinstance(kwargs['custom_executor_spec'],
                      executor_spec.ExecutorClassSpec)
    assert kwargs['transformed_examples'] is ExampleInputs.PREPROCESSED_EXAMPLES(
        pipeline_def)
    assert pipeline_def.components['trainer'] is mock_trainer.return_value
    assert pipeline_def.trainer is mock_trainer.return_value


def test_pipeline_def_train_works_with_optional_args(pipeline_def, monkeypatch):
    # Arrange
    mock_schema = mock.MagicMock()
    pipeline_def.schema_gen = mock_schema

    mock_hparams = mock.MagicMock()
    pipeline_def.user_hyperparameters_importer = mock_hparams

    mock_transform = mock.MagicMock()
    pipeline_def.transform = mock_transform

    mock_example_input = mock.MagicMock()
    pipeline_def.example_gen = mock_example_input
    pipeline_def.cached_example_input = None

    pipeline_def.tuner = None

    mock_trainer = mock.Mock()
    monkeypatch.setattr('fluent_tfx.pipeline_def.Trainer', mock_trainer)

    mock_module = mock.Mock()
    mock_config = mock.Mock()
    mock_executor_spec = mock.Mock()

    # Act
    pipeline_def = pipeline_def.train(
        module_file=mock_module,
        train_args=1, eval_args=2,
        custom_config=mock_config,
        custom_executor_spec=mock_executor_spec,
        example_input=ExampleInputs.RAW_EXAMPLES
    )
    _, kwargs = mock_trainer.call_args

    # Assert
    assert mock_trainer.called
    assert kwargs['train_args'] == 1
    assert kwargs['eval_args'] == 2
    assert kwargs['transformed_examples'] is ExampleInputs.RAW_EXAMPLES(
        pipeline_def)
    assert pipeline_def.cached_example_input is ExampleInputs.RAW_EXAMPLES(
        pipeline_def)
    assert kwargs['schema'] is SchemaInputs.SCHEMA_CHANNEL(pipeline_def)
    assert kwargs['transform_graph'] is mock_transform.outputs['transform_graph']
    assert kwargs['custom_config'] is mock_config
    assert kwargs['custom_executor_spec'] is mock_executor_spec
    assert kwargs['hyperparameters'] is HyperParameterInputs.BEST_HYPERPARAMETERS(
        pipeline_def)
    assert pipeline_def.components['trainer'] is mock_trainer.return_value
    assert pipeline_def.trainer is mock_trainer.return_value


def test_pipeline_def_train_uses_cached_example_input(pipeline_def, monkeypatch):
    # Arrange
    mock_schema = mock.MagicMock()
    pipeline_def.schema_gen = mock_schema

    pipeline_def.transform = None

    mock_input = mock.MagicMock()
    pipeline_def.cached_example_input = mock_input

    pipeline_def.tuner = None
    pipeline_def.user_hyperparameters_importer = None

    mock_trainer = mock.Mock()
    monkeypatch.setattr('fluent_tfx.pipeline_def.Trainer', mock_trainer)

    mock_module = mock.Mock()

    # Act
    pipeline_def = pipeline_def.train(module_file=mock_module)
    _, kwargs = mock_trainer.call_args

    # Assert
    assert mock_trainer.called
    assert kwargs['transformed_examples'] is mock_input
    assert pipeline_def.components['trainer'] is mock_trainer.return_value
    assert pipeline_def.trainer is mock_trainer.return_value


def test_pipeline_def_train_can_use_example_inputs(pipeline_def, monkeypatch):
    # Arrange
    mock_schema = mock.MagicMock()
    pipeline_def.schema_gen = mock_schema

    pipeline_def.transform = None

    mock_example_gen = mock.MagicMock()
    pipeline_def.example_gen = mock_example_gen
    pipeline_def.cached_example_input = None

    pipeline_def.tuner = None
    pipeline_def.user_hyperparameters_importer = None

    mock_trainer = mock.Mock()
    monkeypatch.setattr('fluent_tfx.pipeline_def.Trainer', mock_trainer)

    mock_module = mock.Mock()

    # Act
    pipeline_def = pipeline_def.train(
        module_file=mock_module, example_input=ExampleInputs.RAW_EXAMPLES)
    _, kwargs = mock_trainer.call_args

    # Assert
    assert mock_trainer.called
    assert kwargs['transformed_examples'] is ExampleInputs.RAW_EXAMPLES(
        pipeline_def)
    assert pipeline_def.components['trainer'] is mock_trainer.return_value
    assert pipeline_def.trainer is mock_trainer.return_value


def test_pipeline_def_train_updates_config_with_ai_platform_args(pipeline_def, monkeypatch):
    # Arrange
    mock_schema = mock.MagicMock()
    pipeline_def.schema_gen = mock_schema

    mock_transform = mock.MagicMock()
    pipeline_def.transform = mock_transform

    pipeline_def.cached_example_input = None

    pipeline_def.tuner = None
    pipeline_def.user_hyperparameters_importer = None

    mock_trainer = mock.Mock()
    monkeypatch.setattr('fluent_tfx.pipeline_def.Trainer', mock_trainer)

    mock_module = mock.Mock()

    ai_platform_args = mock.Mock()
    custom_config = mock.Mock()
    # Act
    pipeline_def = pipeline_def.train(
        module_file=mock_module, ai_platform_args=ai_platform_args, custom_config=custom_config)
    _, kwargs = mock_trainer.call_args

    # Assert
    assert mock_trainer.called
    assert kwargs['custom_config'][ai_platform_trainer_executor.TRAINING_ARGS_KEY] is ai_platform_args
    assert pipeline_def.components['trainer'] is mock_trainer.return_value
    assert pipeline_def.trainer is mock_trainer.return_value
