import pytest
from unittest import mock

from fluent_tfx.pipeline_def import PipelineDef, \
    ExampleInputs, SchemaInputs, HyperParameterInputs, \
    build_step


def test_build_step_works():
    # Arrange
    class BuildStepTester:
        def __init__(self):
            self.components = {}

        @build_step('some_component_name')
        def some_build_step(self, ret):
            ret(1)
            return ret

        @build_step('other_component_name')
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


def test_example_inputs_raw_work():
    pass


def test_example_inputs_preprocessed_work():
    pass


def test_hyperparameter_inputs_best_work():
    pass


def test_schema_inputs_schema_channel_work():
    pass
