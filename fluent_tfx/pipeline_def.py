import os
import logging
from typing import Optional, Text, List, Dict, Any, Union
from functools import wraps

import tensorflow_model_analysis as tfma
from tfx.orchestration.pipeline import Pipeline
from tfx.orchestration import metadata, data_types

from tfx.proto import example_gen_pb2, trainer_pb2, \
    infra_validator_pb2, pusher_pb2, bulk_inferrer_pb2
from ml_metadata.proto import metadata_store_pb2

from tfx.dsl.experimental import latest_blessed_model_resolver

from tfx.components.base.base_component import BaseComponent

from tfx.components import CsvExampleGen, ImportExampleGen, \
    StatisticsGen, SchemaGen, Transform, ExampleValidator, ImporterNode, \
    ResolverNode, Trainer, Evaluator, InfraValidator, Pusher, BulkInferrer, Tuner

from tfx.extensions.google_cloud_big_query.example_gen.component import BigQueryExampleGen

from tfx.components.base import executor_spec
from tfx.components.trainer import executor as trainer_executor

from tfx.types import standard_artifacts, Channel
from tfx.extensions.google_cloud_ai_platform.trainer \
    import executor as ai_platform_trainer_executor

import fluent_tfx.input_builders as input_builders


def build_step(component_name: Text):
    """Wraps a function inside a `PipelineDef`.
    Adds the component returned to the dict of used components (`self.components`)
    and then returns self as a fluent builder pattern.

    Args:
        func (function): Function to be wrapped

    Returns: self
    """
    def inner_func(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            component = func(self, *args, **kwargs)
            self.components[component_name] = component
            return self
        return wrapper

    return inner_func


class PipelineDef:
    """The fluent definition class that empowers fluent-tfx.
    Initialise the constructor with a name and use the
    factory methods to construct the pipeline.
    Obtain the tfx pipeline instance by using `.build()`
    """

    def __init__(self, name: Text, bucket: Optional[Text] = None,
                 metadata_connection_config: Optional[metadata_store_pb2.ConnectionConfig] = None):
        """
        Args:
            name (Text): The pipeline's name
            bucket (Text, optional): Intermediate artifacts and staging binaries are going
            to be saved under `{name}/{bucket}/..`. Defaults to './bucket'.
            metadata_connection_config (Optional[metadata_store_pb2.ConnectionConfig], optional):
            Optional ML Metadata configuration for rapid local prototyping you can use `with_sqlite_ml_metadata`. Defaults to None.
        """
        if not bucket:
            bucket = os.path.join(os.getcwd(), 'bucket')

        self.components = {}

        self.pipeline_name = name
        self.pipeline_bucket = bucket
        self.metadata_connection_config = metadata_connection_config
        self.enable_cache = False
        self.beam_pipeline_args = None

        self.cached_example_input = None

        self.example_gen = None
        self.statistics_gen = None
        self.example_validator = None
        self.user_schema_importer = None
        self.schema_gen = None
        self.transform = None
        self.user_hyperparameters_importer = None
        self.trainer = None
        self.tuner = None
        self.infra_validator = None
        self.latest_blessed_model_resolver = None
        self.model_evaluator = None
        self.pusher = None
        self.bulk_inferrer = None

    def with_sqlite_ml_metadata(self):
        """WIll use a sqlite database under {bucket}/{name}/metadata.db as a
        backend for ML Metadata.

        Returns: self
        """
        metadata_connection_string = os.path.join(
            self.pipeline_bucket, self.pipeline_name, 'metadata.db')

        self.metadata_connection_config = metadata. \
            sqlite_metadata_connection_config(metadata_connection_string)

        return self

    def add_custom_component(self, name: Text, component):
        """Adds a custom component given a name to the current components pipeline dict.
        The name should not be one of tensorflow extended's components names, converted to snake_case.

        If the component is not an instance of BaseComponent, it should be a
        callable that expects the self.components dict as input

        Raises:
            ValueError: If provided component is neither a BaseComponent nor a callable

        Returns: self
        """
        if name in self.components:
            logging.info(
                f'{name} is already in the components dict, overriding.')

        if isinstance(component, BaseComponent):
            self.components[name] = component
            return self

        if hasattr(component, '__call__'):
            self.components[name] = component(self.components)
            return self

        raise ValueError(
            f'add_custom_component: Expected callable or BaseComponent, got {component} instead.')
        return self

    @build_step('example_gen')
    def from_csv(self, uri: Text,
                 input_config: Optional[example_gen_pb2.Input] = None,
                 output_config: Optional[example_gen_pb2.Output] = None):
        """Constructs a CsvExampleGen component by using `uri`

        Args:
            uri (Text): Csv file(s) uri
            input_config (Optional[example_gen_pb2.Input], optional): Defaults to None.
            output_config (Optional[example_gen_pb2.Output], optional): Defaults to None.

        Returns: self
        """

        self.example_gen = input_builders.from_csv(
            uri=uri, input_config=input_config, output_config=output_config)
        return self.example_gen

    @build_step('example_gen')
    def from_tfrecord(self, uri: Text,
                      input_config: Optional[example_gen_pb2.Input] = None,
                      output_config: Optional[example_gen_pb2.Output] = None):
        """Constructs an ImportExampleGen component by using `uri`

        Args:
            uri (Text): TFRecord file(s) uri
            input_config (Optional[example_gen_pb2.Input], optional): Defaults to None.
            output_config (Optional[example_gen_pb2.Output], optional): Defaults to None.

        Returns: self
        """
        self.example_gen = input_builders.from_tfrecord(
            uri=uri, input_config=input_config, output_config=output_config)
        return self.example_gen

    @build_step('example_gen')
    def from_bigquery(self, query: Text,
                      input_config: Optional[example_gen_pb2.Input] = None,
                      output_config: Optional[example_gen_pb2.Output] = None):
        """Constructs a BigQueryExampleGen component by using `uri`

        Args:
            query (Text): The query to run
            input_config (Optional[example_gen_pb2.Input], optional): Defaults to None.
            output_config (Optional[example_gen_pb2.Output], optional): Defaults to None.

        Returns: self
        """
        self.example_gen = input_builders.from_bigquery(
            query=query, input_config=input_config, output_config=output_config)
        return self.example_gen

    @build_step('example_gen')
    def from_custom_example_gen_component(self, component: BaseComponent):
        """Uses a custom tfx component to generate example files.
        The component should comply with the naming conventions of the other
        example generating components of tfx. (for example, contain a `.outputs['examples']` attribute)

        Args:
            component (BaseComponent): Your custom, compatible component.

        Returns: self
        """
        self.example_gen = component

        return self.example_gen

    @build_step('user_schema_importer')
    def with_imported_schema(self, uri: Text):
        """Constructs an ImporterNode component that imports
        the schema in the pipelineas an artifact.

        If infer_schema is called, the subsequent components will still use this
        use provided schema, but the SchemaGen component will still produce inferred
        schema artifacts.

        Args:
            uri (Text): Schema .pbtxt file uri

        Returns: self
        """
        self.user_schema_importer = input_builders.with_imported_schema(
            uri=uri)

        return self.user_schema_importer

    @build_step('statistics_gen')
    def generate_statistics(self):
        """Constructs a StatisticsGen component on the example_gen output files

        Returns: self
        """
        args = {
            'examples': self.example_gen.outputs['examples']
        }

        if self.user_schema_importer:
            args['schema'] = self.user_schema_importer.outputs['result']

        self.statistics_gen = StatisticsGen(**args)

        return self.statistics_gen

    @build_step('schema_gen')
    def infer_schema(self, infer_feature_shape: Optional[bool] = False):
        """Constructs a SchemaGen component. a StatisticsGen component via `generate_statistics`
        is required as an input.

        Args:
            infer_feature_shape (bool, optional): Defaults to False.

        Returns: self
        """
        self.schema_gen = SchemaGen(
            statistics=self.statistics_gen.outputs['statistics'],
            infer_feature_shape=infer_feature_shape)

        return self.schema_gen

    @build_step('example_validator')
    def validate_input_data(self, exclude_splits: Optional[List[Text]] = None):
        """Constructs an ExampleValidator component that uses a schema and the output
        of StatisticsGen via `generate_statistics` to validate input data.

        If a user provided schema is specified, it will be used.

        Returns: self
        """
        args = {
            'statistics': self.statistics_gen.outputs['statistics'],
            'schema': SchemaInputs.SCHEMA_CHANNEL(self),
        }

        if exclude_splits:
            args['exclude_splits'] = exclude_splits

        self.example_validator = ExampleValidator(**args)
        return self.example_validator

    @build_step('transform')
    def preprocess(self, module_file: Union[Text, data_types.RuntimeParameter], materialize: bool = True):
        """Constructs a Transform component using examples generated and a schema.

        If a user provided schema is specified, it will be used.

        Args:
            module_file (Union[Text, data_types.RuntimeParameter]): The module file (a/b/c.py)
            that contains the preprocessing_fn function. The signature should be
            `def preprocessing_fn(inputs: Dict[Text, Any]) -> Dict[Text, Any]`

        Returns: self
        """
        args = {
            'examples': self.example_gen.outputs['examples'],
            'module_file': module_file,
            'schema': SchemaInputs.SCHEMA_CHANNEL(self),
            'materialize': materialize
        }

        self.transform = Transform(**args)
        return self.transform

    @build_step('train_base_model')
    def with_base_model(self, uri: Text):
        """Constructs an ImporterNode component that imports a `standard_artifacts.Model`
        artifact to use as a starting point for training.

        Args:
            uri (Text): Model artifact's uri

        Returns: self
        """
        self.train_base_model = input_builders.with_base_model(uri=uri)

        return self.train_base_model

    @build_step('user_hyperparameters_importer')
    def with_hyperparameters(self, uri: Text):
        """Constructs an ImporterNode component that imports a `standard_artifacts.HyperParameters`
        artifact to use for future runs.

        Args:
            uri (Text): Hyperparameter artifact's uri

        Returns: self
        """
        self.user_hyperparameters_importer = input_builders.with_hyperparameters(
            uri=uri)

        return self.user_hyperparameters_importer

    @build_step('tuner')
    def tune(self, module_file:
             Union[Text, data_types.RuntimeParameter],
             train_args: Optional[trainer_pb2.TrainArgs] = None,
             eval_args: Optional[trainer_pb2.EvalArgs] = None,
             example_input=None):
        args = {
            'module_file': module_file,
            'schema': SchemaInputs.SCHEMA_CHANNEL(self)
        }

        if train_args:
            args['train_args'] = train_args

        if eval_args:
            args['eval_args'] = eval_args

        if self.transform:
            args['transform_graph'] = self.transform.outputs['transform_graph']

        if example_input:
            inputs = example_input(self)
            args['examples'] = inputs

            self.cached_example_input = inputs
        else:
            args['examples'] = ExampleInputs.PREPROCESSED_EXAMPLES(
                self)

        self.tuner = Tuner(**args)

        return self.tuner

    @build_step('trainer')
    def train(self, module_file: Union[Text, data_types.RuntimeParameter],
              train_args: Optional[trainer_pb2.TrainArgs] = None,
              eval_args: Optional[trainer_pb2.EvalArgs] = None,
              example_input: Optional = None,
              custom_executor_spec: Optional[executor_spec.ExecutorSpec] = None,
              ai_platform_args: Optional[Dict[Text, Text]] = None,
              custom_config: Optional[Dict[Text, Any]] = None):
        """Constructs a Trainer component given the arguments

        If there is a user provided schema, it will be used.
        Similarly, if there are hyperparameters explicitly defined, they will be used.

        No need to explicitly re-specify example inputs if you've already specified them in `tune()`.

        By default, an trainer_executor.GenericExecutor is used.

        If ai platform training arguments are provided, the executor is set to ai_platform_trainer_executor.GenericExecutor

        Args:
            module_file (Union[Text, data_types.RuntimeParameter]): The module file that contains run_fn.
            The signature of the method must be `def run_fn(fn_args: TrainerFnArgs)`.
            example_input ([type], optional): A `ExampleInputs.{RAW_EXAMPLES, PREPROCESSED_EXAMPLES}` function reference which provides
            the example input channel. Defaults to None, where the input channel is set to the transformed/preprocessed examples.

        Returns: self
        """
        args = {
            'module_file': module_file,
            'schema': SchemaInputs.SCHEMA_CHANNEL(self)
        }

        hparams = HyperParameterInputs.BEST_HYPERPARAMETERS(self)
        if hparams:
            args['hyperparameters'] = hparams

        if not custom_executor_spec:
            custom_executor_spec = executor_spec.ExecutorClassSpec(
                trainer_executor.GenericExecutor)

        args['custom_executor_spec'] = custom_executor_spec

        if custom_config:
            args['custom_config'] = custom_config

        if train_args:
            args['train_args'] = train_args

        if eval_args:
            args['eval_args'] = eval_args

        if self.transform:
            args['transform_graph'] = self.transform.outputs['transform_graph']

        if not example_input and self.cached_example_input:
            args['transformed_examples'] = self.cached_example_input
        elif example_input:
            inputs = example_input(self)
            args['transformed_examples'] = inputs

            self.cached_example_input = self.cached_example_input or inputs
        else:
            args['transformed_examples'] = ExampleInputs.PREPROCESSED_EXAMPLES(
                self)

        if ai_platform_args is not None:
            args.update({
                'custom_executor_spec':
                executor_spec.ExecutorClassSpec(
                    ai_platform_trainer_executor.GenericExecutor
                ),
                    'custom_config': {
                    ai_platform_trainer_executor.TRAINING_ARGS_KEY:
                    ai_platform_args,
                }
            })

        self.trainer = Trainer(**args)
        return self.trainer

    @build_step('model_evaluator')
    def evaluate_model(self, eval_config: tfma.EvalConfig,
                       example_input: Optional = None,
                       example_provider_component: Optional[BaseComponent] = None):
        """Constructs an Evaluator component

        No need to specify example_input if you've already specified it in `train()` or `tune()`.

        Args:
            example_input (Optional, optional): A `ExampleInputs.{RAW_EXAMPLES, PREPROCESSED_EXAMPLES}` function reference which provides
            the example input channel. Defaults to None, where the input channel is set to the transformed/preprocessed examples. Defaults
            to RAW_EXAMPLES.


            example_provider_component (Optional[BaseComponent], optional): An external example input component,
            which should output examples at the `outputs['examples']` attribute.

        Returns: self
        """

        self.latest_blessed_model_resolver = input_builders.get_latest_blessed_model_resolver()
        self.components['latest_blessed_model_resolver'] = self.latest_blessed_model_resolver

        args = {
            'model': self.trainer.outputs['model'],
            'baseline_model': self.latest_blessed_model_resolver.outputs['model'],
            'eval_config': eval_config
        }

        if example_provider_component:
            self.components['model_evaluator_example_provider'] = example_provider_component
            args['examples'] = example_provider_component.outputs['examples']

        if not example_provider_component:
            if not example_input and self.cached_example_input:
                args['examples'] = self.cached_example_input
            elif example_input:
                inputs = example_input(self)
                args['examples'] = inputs

                self.cached_example_input = self.cached_example_input or inputs
            else:
                args['examples'] = ExampleInputs.RAW_EXAMPLES(
                    self)

        self.model_evaluator = Evaluator(**args)

        return self.model_evaluator

    @build_step('infra_validator')
    def infra_validate(self, serving_spec: infra_validator_pb2.ServingSpec,
                       validation_spec: Optional[infra_validator_pb2.ValidationSpec] = None,
                       request_spec: Optional[infra_validator_pb2.RequestSpec] = None,
                       example_input=None):
        """Constructs an InfraValidator component.

        No need to specify example_input if you've already specified it in `train()`, `tune()` or `evaluate_model()`.


        Args:
            example_input ([type], optional): A `ExampleInputs.{RAW_EXAMPLES, PREPROCESSED_EXAMPLES}` function reference which provides
            the example input channel. Defaults to None, where the input channel is set to the transformed/preprocessed examples.

        Returns: self
        """
        args = {
            'model': self.trainer.outputs['model'],
            'serving_spec': serving_spec,
        }

        if validation_spec:
            args['validation_spec'] = validation_spec

        if request_spec:
            args['request_spec'] = request_spec

        if not example_input and self.cached_example_input:
            args['examples'] = self.cached_example_input
        elif example_input:
            inputs = example_input(self)
            args['examples'] = inputs

            # unnecessary to assign `inputs` to `self.cached_example_input` here
            # this is probably the last stop for the cached example input train
        else:
            args['examples'] = ExampleInputs.RAW_EXAMPLES(
                self)

        self.infra_validator = InfraValidator(**args)
        return self.infra_validator

    @build_step('pusher')
    def push_to(self, relative_push_uri: Optional[Text] = None,
                push_destination: Optional[pusher_pb2.PushDestination] = None,
                custom_config: Optional[Dict[Text, Any]] = None,
                custom_executor_spec: Optional[executor_spec.ExecutorSpec] = None):
        """Constructs a Pusher component

        Args:
            relative_push_uri (Optional[Text], optional): The relative to `{bucket}/{name}/` uri to push models to.
            Defaults to None, where a `push_destination` `pusher_pb2` is expected.

        Returns: self
        """
        args = {
            'model': self.trainer.outputs['model'],
        }

        if push_destination:
            args['push_destination'] = push_destination
        elif relative_push_uri:
            args['push_destination'] = pusher_pb2.PushDestination(
                filesystem=pusher_pb2.PushDestination.Filesystem(
                    base_directory=os.path.join(
                        self.pipeline_bucket, self.pipeline_name, relative_push_uri)))

        if self.model_evaluator:
            args['model_blessing'] = self.model_evaluator.outputs['blessing']

        if custom_config:
            args['custom_config'] = custom_config

        if custom_executor_spec:
            args['custom_executor_spec'] = custom_executor_spec

        if self.infra_validator:
            args['infra_blessing'] = self.infra_validator.outputs['blessing']

        self.pusher = Pusher(**args)
        return self.pusher

    @build_step('bulk_inferrer')
    def bulk_infer(self, example_provider_component: BaseComponent,
                   data_spec: Optional[Union[bulk_inferrer_pb2.DataSpec,
                                             Dict[Text, Any]]] = None,
                   model_spec: Optional[Union[bulk_inferrer_pb2.ModelSpec,
                                              Dict[Text, Any]]] = None):
        """Generates a BulkInferrer component that uses trainer model output and model evaluator
        blessing.

        Args:
            example_provider_component (BaseComponent): A user-provided component that provides example tfrecord inputs to the
            `.outputs['examples']` attribute.

        Returns: self
        """
        self.components['bulk_inferrer_example_provider'] = example_provider_component

        args = {
            'examples': example_provider_component.outputs['examples'],
            'model': self.trainer.outputs['model']
        }

        if self.model_evaluator:
            args['model_blessing'] = self.model_evaluator.outputs['blessing']

        if data_spec:
            args['data_spec'] = data_spec

        if model_spec:
            args['model_spec'] = model_spec

        self.bulk_inferrer = BulkInferrer(**args)
        return self.bulk_inferrer

    def cache(self, enable_cache=True):
        """
        Args:
            enable_cache (bool, optional): Defaults to True.

        Returns: self
        """
        self.enable_cache = enable_cache

        return self

    def with_beam_pipeline_args(self, args: Optional[List[Text]]):
        """

        Args:
            args (Optional[List[Text]]): Beam Pipeline Arguments

        Returns: self
        """
        self.beam_pipeline_args = args
        return self

    def build(self) -> Pipeline:
        """Builds the pipeline.

        Returns:
            Pipeline: The native TFX pipeline
        """
        args = {
            'pipeline_name': self.pipeline_name,
            'pipeline_root': os.path.join(self.pipeline_bucket, self.pipeline_name, 'staging'),
            'components': self.components.values(),
            'enable_cache': self.enable_cache or False
        }

        if self.metadata_connection_config:
            args['metadata_connection_config'] = self.metadata_connection_config

        if self.beam_pipeline_args:
            args['beam_pipeline_args'] = self.beam_pipeline_args

        return Pipeline(**args)


class ExampleInputs:
    """Provides accessor functions for channel artifact access, regarding example tf records.
    RAW EXAMPLES -> import example_gen files
    PREPROCESSED_EXAMPLES -> import transformed examples
    """
    @staticmethod
    def _get_raw_examples_channel(pipeline_def: PipelineDef) -> Channel:
        return pipeline_def.example_gen.outputs['examples']

    @staticmethod
    def _get_preprocessed_examples_channel(pipeline_def: PipelineDef) -> Channel:
        return pipeline_def.transform.outputs['transformed_examples']

    RAW_EXAMPLES = _get_raw_examples_channel
    PREPROCESSED_EXAMPLES = _get_preprocessed_examples_channel


class HyperParameterInputs:
    """Provides accessor functions for channel artifact access, regarding hyperparameter artifacts.
    BEST_HYPERPARAMETERS -> if they are specified explicitly, return them. Else, return tuner outputs.
    """
    @staticmethod
    def _get_best_hyperparameters(pipeline_def: PipelineDef):
        if pipeline_def.user_hyperparameters_importer:
            return pipeline_def.user_hyperparameters_importer.outputs['result']

        if pipeline_def.tuner:
            return pipeline_def.tuner.outputs['best_hyperparameters']

        return None

    BEST_HYPERPARAMETERS = _get_best_hyperparameters


class SchemaInputs:
    """Provides accessor functions for channel artifact access, regarding schema artifacts.
    SCHEMA_CHANNEL -> if they schema is specified explicitly, return it. Else, return inferred schema.
    """
    @staticmethod
    def _get_schema_channel(pipeline_def: PipelineDef):
        if pipeline_def.user_schema_importer:
            return pipeline_def.user_schema_importer.outputs['result']

        if pipeline_def.schema_gen:
            return pipeline_def.schema_gen.outputs['schema']

        return None

    SCHEMA_CHANNEL = _get_schema_channel
