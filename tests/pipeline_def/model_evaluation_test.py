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


def test_evaluate_model_works(pipeline_def, monkeypatch):
    # Arrange
    mock_latest_blessed = mock.MagicMock()
    latest_blessed_ret_val = mock.MagicMock()
    mock_latest_blessed.return_value = latest_blessed_ret_val
    monkeypatch.setattr(
        'fluent_tfx.pipeline_def.input_builders.get_latest_blessed_model_resolver', mock_latest_blessed)

    mock_evaluator = mock.MagicMock()
    mock_evaluator_ret_val = mock.MagicMock()
    mock_evaluator.return_value = mock_evaluator_ret_val
    monkeypatch.setattr('fluent_tfx.pipeline_def.Evaluator', mock_evaluator)

    mock_trainer = mock.MagicMock()
    pipeline_def.trainer = mock_trainer

    mock_raw_examples = mock.MagicMock()
    pipeline_def.example_gen = mock_raw_examples
    pipeline_def.cached_example_input = None

    mock_eval_conf = mock.Mock()

    # Act
    pipeline_def = pipeline_def.evaluate_model(eval_config=mock_eval_conf)
    _, kwargs = mock_evaluator.call_args

    # Assert
    assert mock_evaluator.called
    assert kwargs['eval_config'] is mock_eval_conf
    assert kwargs['model'] is mock_trainer.outputs['model']
    assert kwargs['examples'] is ExampleInputs.RAW_EXAMPLES(pipeline_def)
    assert kwargs['baseline_model'] is latest_blessed_ret_val.outputs['model']

    assert pipeline_def.components['model_evaluator'] is mock_evaluator.return_value
    assert pipeline_def.model_evaluator is mock_evaluator.return_value

    assert pipeline_def.components['latest_blessed_model_resolver'] is mock_latest_blessed.return_value
    assert pipeline_def.latest_blessed_model_resolver is mock_latest_blessed.return_value


def test_pipeline_def_evaluate_model_can_use_custom_example_component(pipeline_def, monkeypatch):
    # Arrange
    mock_latest_blessed = mock.MagicMock()
    latest_blessed_ret_val = mock.MagicMock()
    mock_latest_blessed.return_value = latest_blessed_ret_val
    monkeypatch.setattr(
        'fluent_tfx.pipeline_def.input_builders.get_latest_blessed_model_resolver', mock_latest_blessed)

    mock_evaluator = mock.MagicMock()
    mock_evaluator_ret_val = mock.MagicMock()
    mock_evaluator.return_value = mock_evaluator_ret_val
    monkeypatch.setattr('fluent_tfx.pipeline_def.Evaluator', mock_evaluator)

    mock_trainer = mock.MagicMock()
    pipeline_def.trainer = mock_trainer

    mock_eval_conf = mock.Mock()

    mock_example_provider_component = mock.MagicMock(spec=BaseComponent)

    # Act
    pipeline_def = pipeline_def.evaluate_model(
        eval_config=mock_eval_conf, example_provider_component=mock_example_provider_component)
    _, kwargs = mock_evaluator.call_args

    # Assert
    assert mock_evaluator.called

    assert kwargs['examples'] is mock_example_provider_component.outputs['examples']
    assert pipeline_def.components['model_evaluator_example_provider'] is mock_example_provider_component

    assert pipeline_def.components['model_evaluator'] is mock_evaluator.return_value
    assert pipeline_def.model_evaluator is mock_evaluator.return_value


def test_pipeline_def_evaluate_model_can_use_cached_example_inputs(pipeline_def, monkeypatch):
    # Arrange
    mock_latest_blessed = mock.MagicMock()
    latest_blessed_ret_val = mock.MagicMock()
    mock_latest_blessed.return_value = latest_blessed_ret_val
    monkeypatch.setattr(
        'fluent_tfx.pipeline_def.input_builders.get_latest_blessed_model_resolver', mock_latest_blessed)

    mock_evaluator = mock.MagicMock()
    mock_evaluator_ret_val = mock.MagicMock()
    mock_evaluator.return_value = mock_evaluator_ret_val
    monkeypatch.setattr('fluent_tfx.pipeline_def.Evaluator', mock_evaluator)

    mock_trainer = mock.MagicMock()
    pipeline_def.trainer = mock_trainer

    pipeline_def.example_gen = None

    mock_eval_conf = mock.Mock()

    mock_cached_input = mock.Mock()
    pipeline_def.cached_example_input = mock_cached_input

    # Act
    pipeline_def = pipeline_def.evaluate_model(
        eval_config=mock_eval_conf)
    _, kwargs = mock_evaluator.call_args

    # Assert
    assert mock_evaluator.called

    assert kwargs['examples'] is mock_cached_input
    assert pipeline_def.components['model_evaluator'] is mock_evaluator.return_value
    assert pipeline_def.model_evaluator is mock_evaluator.return_value


def test_pipeline_def_evaluate_model_can_cache_provided_example_input(pipeline_def, monkeypatch):
    # Arrange
    mock_latest_blessed = mock.MagicMock()
    latest_blessed_ret_val = mock.MagicMock()
    mock_latest_blessed.return_value = latest_blessed_ret_val
    monkeypatch.setattr(
        'fluent_tfx.pipeline_def.input_builders.get_latest_blessed_model_resolver', mock_latest_blessed)

    mock_evaluator = mock.MagicMock()
    mock_evaluator_ret_val = mock.MagicMock()
    mock_evaluator.return_value = mock_evaluator_ret_val
    monkeypatch.setattr('fluent_tfx.pipeline_def.Evaluator', mock_evaluator)

    mock_trainer = mock.MagicMock()
    pipeline_def.trainer = mock_trainer

    mock_example_gen = mock.MagicMock()
    pipeline_def.example_gen = mock_example_gen
    pipeline_def.cached_example_input = None

    mock_eval_conf = mock.Mock()

    # Act
    pipeline_def = pipeline_def.evaluate_model(
        eval_config=mock_eval_conf, example_input=ExampleInputs.RAW_EXAMPLES)
    _, kwargs = mock_evaluator.call_args

    # Assert
    assert mock_evaluator.called

    assert kwargs['examples'] is mock_example_gen.outputs['examples']
    assert pipeline_def.cached_example_input is ExampleInputs.RAW_EXAMPLES(
        pipeline_def)
    assert pipeline_def.components['model_evaluator'] is mock_evaluator.return_value
    assert pipeline_def.model_evaluator is mock_evaluator.return_value
