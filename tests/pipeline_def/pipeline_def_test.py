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


def test_pipeline_def_generate_statistics_works(pipeline_def, monkeypatch):
    # Arrange
    mock_statistics_gen = mock.Mock()
    monkeypatch.setattr(
        'fluent_tfx.pipeline_def.StatisticsGen', mock_statistics_gen)

    mock_example_gen = mock.MagicMock()
    pipeline_def.example_gen = mock_example_gen

    # Act
    pipeline_def = pipeline_def.generate_statistics()
    _, kwargs = mock_statistics_gen.call_args

    # Assert
    assert mock_statistics_gen.called
    assert kwargs['examples'] is mock_example_gen.outputs['examples']
    assert 'schema' not in kwargs
    assert pipeline_def.components['statistics_gen'] is mock_statistics_gen.return_value
    assert pipeline_def.statistics_gen is mock_statistics_gen.return_value


def test_pipeline_def_generate_statistics_uses_imported_schema_if_specified(pipeline_def, monkeypatch):
    # Arrange
    mock_statistics_gen = mock.Mock()
    monkeypatch.setattr(
        'fluent_tfx.pipeline_def.StatisticsGen', mock_statistics_gen)

    mock_example_gen = mock.MagicMock()
    pipeline_def.example_gen = mock_example_gen

    mock_schema_importer = mock.MagicMock()
    pipeline_def.user_schema_importer = mock_schema_importer

    # Act
    pipeline_def = pipeline_def.generate_statistics()
    _, kwargs = mock_statistics_gen.call_args

    # Assert
    assert mock_statistics_gen.called
    assert 'schema' in kwargs
    assert kwargs['schema'] is mock_schema_importer.outputs['result']
    assert pipeline_def.components['statistics_gen'] is mock_statistics_gen.return_value
    assert pipeline_def.statistics_gen is mock_statistics_gen.return_value


def test_pipeline_def_validate_input_data_works(pipeline_def, monkeypatch):
    # Arrange
    mock_statistics_gen = mock.MagicMock()
    pipeline_def.statistics_gen = mock_statistics_gen
    mock_schema_gen = mock.MagicMock()
    pipeline_def.schema_gen = mock_schema_gen

    mock_example_validator = mock.Mock()
    monkeypatch.setattr(
        'fluent_tfx.pipeline_def.ExampleValidator', mock_example_validator)

    # Act
    pipeline_def = pipeline_def.validate_input_data()
    _, kwargs = mock_example_validator.call_args

    # Assert
    assert mock_example_validator.called
    assert kwargs['statistics'] is mock_statistics_gen.outputs['statistics']
    assert kwargs['schema'] is SchemaInputs.SCHEMA_CHANNEL(pipeline_def)
    assert 'exclude_splits' not in kwargs
    assert pipeline_def.components['example_validator'] is mock_example_validator.return_value
    assert pipeline_def.example_validator is mock_example_validator.return_value


def test_pipeline_def_validate_passes_exclude_splits_arg(pipeline_def, monkeypatch):
    # Arrange
    mock_statistics_gen = mock.MagicMock()
    pipeline_def.statistics_gen = mock_statistics_gen
    mock_schema_gen = mock.MagicMock()
    pipeline_def.schema_gen = mock_schema_gen

    mock_example_validator = mock.Mock()
    monkeypatch.setattr(
        'fluent_tfx.pipeline_def.ExampleValidator', mock_example_validator)

    mock_exclude_split = mock.Mock()

    # Act
    pipeline_def = pipeline_def.validate_input_data(
        exclude_splits=mock_exclude_split)
    _, kwargs = mock_example_validator.call_args

    # Assert
    assert 'exclude_splits' in kwargs
    assert kwargs['exclude_splits'] is mock_exclude_split\



def test_pipeline_def_preprocess_works(pipeline_def, monkeypatch):
    # Arrange
    mock_examples = mock.MagicMock()
    pipeline_def.example_gen = mock_examples

    mock_schema = mock.MagicMock()
    pipeline_def.schema_gen = mock_schema

    mock_transform = mock.Mock()
    monkeypatch.setattr(
        'fluent_tfx.pipeline_def.Transform', mock_transform)

    mock_module = mock.Mock()

    # Act
    pipeline_def = pipeline_def.preprocess(mock_module)
    _, kwargs = mock_transform.call_args

    # Assert
    assert mock_transform.called
    assert kwargs['examples'] is mock_examples.outputs['examples']
    assert kwargs['module_file'] is mock_module
    assert kwargs['schema'] is SchemaInputs.SCHEMA_CHANNEL(pipeline_def)
    assert kwargs['materialize'] == True
    assert pipeline_def.components['transform'] is mock_transform.return_value
    assert pipeline_def.transform is mock_transform.return_value


def test_pipeline_def_preprocess_passes_materialize_arg(pipeline_def, monkeypatch):
    # Arrange
    mock_examples = mock.MagicMock()
    pipeline_def.example_gen = mock_examples

    mock_schema = mock.MagicMock()
    pipeline_def.schema_gen = mock_schema

    mock_transform = mock.Mock()
    monkeypatch.setattr(
        'fluent_tfx.pipeline_def.Transform', mock_transform)

    mock_module = mock.Mock()

    # Act
    pipeline_def = pipeline_def.preprocess(mock_module, materialize=False)
    _, kwargs = mock_transform.call_args

    # Assert
    assert mock_transform.called
    assert kwargs['materialize'] == False


def test_pipeline_def_enable_cache_works(pipeline_def):
    # Arrange
    mock_cache = mock.Mock()

    # Act
    pipeline_def = pipeline_def.cache(mock_cache)

    # Assert
    assert pipeline_def.enable_cache is mock_cache


def test_pipeline_def_with_sqlite_ml_metadata_works(pipeline_def):
    # Act
    pipeline_def = pipeline_def.with_sqlite_ml_metadata()
    metadata_uri = pipeline_def.metadata_connection_config.sqlite.filename_uri

    # Assert
    assert metadata_uri.endswith('metadata.db')
    assert pipeline_def.pipeline_name in metadata_uri
    assert pipeline_def.pipeline_bucket in metadata_uri


def test_pipeline_def_assigns_beam_args(pipeline_def):
    # Arrange
    beam_args = mock.Mock()

    # Act
    pipeline_def = pipeline_def.with_beam_pipeline_args(beam_args)

    # Assert
    assert pipeline_def.beam_pipeline_args is beam_args


def test_pipeline_build_works(pipeline_def, monkeypatch):
    # Arrange
    mock_components = mock.Mock()
    pipeline_def.components = mock_components
    mock_cache = mock.Mock()
    pipeline_def.enable_cache = mock_cache
    mock_metadata_conf = mock.Mock()
    pipeline_def.metadata_connection_config = mock_metadata_conf
    mock_beam_args = mock.Mock()
    pipeline_def.beam_pipeline_args = mock_beam_args

    mock_tfx_pipeline = mock.Mock()
    monkeypatch.setattr('fluent_tfx.pipeline_def.Pipeline', mock_tfx_pipeline)

    # Act
    pipeline = pipeline_def.build()
    _, kwargs = mock_tfx_pipeline.call_args

    # Assert
    assert mock_tfx_pipeline.called
    assert pipeline is mock_tfx_pipeline.return_value

    assert kwargs['components'] is mock_components.values()
    assert kwargs['enable_cache'] is mock_cache

    assert 'metadata_connection_config' in kwargs
    assert kwargs['metadata_connection_config'] is mock_metadata_conf
    assert 'beam_pipeline_args' in kwargs
    assert kwargs['beam_pipeline_args'] is mock_beam_args


def test_pipeline_build_does_not_provide_optional_args_if_not_specified_and_does_not_use_cache_by_default(pipeline_def, monkeypatch):
    # Arrange
    mock_components = mock.Mock()
    pipeline_def.components = mock_components
    pipeline_def.enable_cache = None
    pipeline_def.metadata_connection_config = None
    pipeline_def.beam_pipeline_args = None

    mock_tfx_pipeline = mock.Mock()
    monkeypatch.setattr('fluent_tfx.pipeline_def.Pipeline', mock_tfx_pipeline)

    # Act
    pipeline = pipeline_def.build()
    _, kwargs = mock_tfx_pipeline.call_args

    # Assert
    assert mock_tfx_pipeline.called
    assert pipeline is mock_tfx_pipeline.return_value

    assert kwargs['components'] is mock_components.values()
    assert kwargs['enable_cache'] == False

    assert 'metadata_connection_config' not in kwargs
    assert 'beam_pipeline_args' not in kwargs


def test_build_step_works():
    # Arrange
    class BuildStepTester:
        def __init__(self):
            self.components = {}

        @ build_step('some_component_name')
        def some_build_step(self, ret):
            ret(1)
            return ret

        @ build_step('other_component_name')
        def other_build_step(self, ret):
            ret(2)
            return ret

    bs_tester = BuildStepTester()
    some_comp = mock.Mock()
    other_comp = mock.Mock()

    # Act
    bs_tester.some_build_step(some_comp)
    bs_tester.other_build_step(other_comp)

    # Assert
    assert len(bs_tester.components) == 2
    assert 'some_component_name' in bs_tester.components
    assert bs_tester.components['some_component_name'] is some_comp

    assert 'other_component_name' in bs_tester.components
    assert bs_tester.components['other_component_name'] is other_comp

    some_comp.assert_called_once_with(1)
    other_comp.assert_called_once_with(2)
