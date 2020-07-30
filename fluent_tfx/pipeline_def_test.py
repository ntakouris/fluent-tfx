import pytest
from unittest import mock

from .pipeline_def import PipelineDef, \
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


def test_raw_example_inputs_raw_work():
    # Arrange
    mock_pipeline_def = mock.MagicMock()
    mock_pipeline_def.example_gen.outputs = {
        'examples': 'example_channel'}

    # Act
    examples = ExampleInputs.RAW_EXAMPLES(mock_pipeline_def)

    # Assert
    assert examples == 'example_channel'


def test_preprocessed_example_inputs_work():
    # Arrange
    mock_pipeline_def = mock.MagicMock()
    mock_pipeline_def.transform.outputs = {
        'transformed_examples': 'example_channel'}

    # Act
    examples = ExampleInputs.PREPROCESSED_EXAMPLES(mock_pipeline_def)

    # Assert
    assert examples == 'example_channel'


def test_best_hyperparameter_inputs_returns_user_provided_if_no_tuner():
    # Arrange
    mock_pipeline_def = mock.MagicMock()
    mock_pipeline_def.user_hyperparameters_importer.outputs = {
        'result': 'hparams'}

    # Act
    hparams = HyperParameterInputs.BEST_HYPERPARAMETERS(mock_pipeline_def)

    # Assert
    assert hparams == 'hparams'


def test_best_hyperparameter_inputs_returns_user_provided_even_if_tuner_exists():
    # Arrange
    mock_pipeline_def = mock.MagicMock()
    mock_pipeline_def.tuner.outputs = {
        'best_hyperparameters': 'tuner_hparams'}

    mock_pipeline_def.user_hyperparameters_importer.outputs = {
        'result': 'imported_hparams'}

    # Act
    hparams = HyperParameterInputs.BEST_HYPERPARAMETERS(mock_pipeline_def)

    # Assert
    assert hparams == 'imported_hparams'


def test_best_hyperparameter_inputs_returns_tuner_if_not_user_provided():
    # Arrange
    mock_pipeline_def = mock.MagicMock()
    mock_pipeline_def.user_hyperparameters_importer = None
    mock_pipeline_def.tuner.outputs = {
        'best_hyperparameters': 'hparams'}

    # Act
    hparams = HyperParameterInputs.BEST_HYPERPARAMETERS(mock_pipeline_def)

    # Assert
    assert hparams == 'hparams'


def test_schema_channel_inputs_returns_user_provided_if_no_schema_gen():
    # Arrange
    mock_pipeline_def = mock.MagicMock()
    mock_pipeline_def.user_schema_importer.outputs = {
        'result': 'schema'}

    # Act
    hparams = SchemaInputs.SCHEMA_CHANNEL(mock_pipeline_def)

    # Assert
    assert hparams == 'schema'


def test_schema_channel_inputs_returns_user_provided_even_if_schema_gen_exists():
    # Arrange
    mock_pipeline_def = mock.MagicMock()
    mock_pipeline_def.schema_gen.outputs = {
        'schema': 'schema_gen'}
    mock_pipeline_def.user_schema_importer.outputs = {
        'result': 'schema_provided'}

    # Act
    hparams = SchemaInputs.SCHEMA_CHANNEL(mock_pipeline_def)

    # Assert
    assert hparams == 'schema_provided'


def test_schema_channel_inputs_returns_schema_gen_if_not_user_provided():
    # Arrange
    mock_pipeline_def = mock.MagicMock()
    mock_pipeline_def.user_schema_importer = None
    mock_pipeline_def.schema_gen.outputs = {
        'schema': 'schema'}

    # Act
    hparams = SchemaInputs.SCHEMA_CHANNEL(mock_pipeline_def)

    # Assert
    assert hparams == 'schema'
