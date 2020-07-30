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


def test_pipeline_def_add_custom_component_BaseComponent_works(pipeline_def):
    # Arrange
    fake_components = {'a': 'a_instance', 'b': 'b_instance'}
    pipeline_def.components = fake_components

    mock_base_component = mock.Mock(spec=BaseComponent)

    # Act
    pipeline_def = pipeline_def.add_custom_component(
        name='base_component', component=mock_base_component)

    # Assert
    assert 'base_component' in pipeline_def.components
    assert pipeline_def.components['base_component'] is mock_base_component


def test_pipeline_def_add_custom_component_callable_works(pipeline_def):
    # Arrange
    fake_components = {'a': 'a_instance', 'b': 'b_instance'}
    pipeline_def.components = fake_components

    mock_component_ctor = mock.Mock()

    def callable(components):
        return mock_component_ctor(components)

    # Act
    pipeline_def = pipeline_def.add_custom_component(
        name='some_component', component=callable)

    args, _ = mock_component_ctor.call_args
    existing_components = args[0]

    # Assert
    assert mock_component_ctor.called
    assert existing_components['a'] == 'a_instance'
    assert existing_components['b'] == 'b_instance'

    assert 'some_component' in pipeline_def.components
    assert pipeline_def.components['some_component'] is mock_component_ctor.return_value


def test_pipeline_def_add_custom_component_incompatible_type_raises_value_error(pipeline_def):
    # Act Assert
    with pytest.raises(ValueError):
        pipeline_def.add_custom_component(
            name='asdf_component', component='neither BaseComponent nor callable')
