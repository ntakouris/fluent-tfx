import logging
from typing import Optional, Text, List, Dict, Any, Union
from functools import wraps, partial

import tensorflow_model_analysis as tfma
from tfx.orchestration import metadata, pipeline, data_types

from tfx.proto import example_gen_pb2, trainer_pb2, infra_validator_pb2, pusher_pb2
from ml_metadata.proto import metadata_store_pb2

from tfx.dsl.experimental import latest_blessed_model_resolver
from tfx.utils.dsl_utils import csv_input, tfrecord_input, external_input

from tfx.components.base.base_component import BaseComponent

from tfx.components import CsvExampleGen, ImportExampleGen, BigQueryExampleGen, StatisticsGen, SchemaGen, Transform, ExampleValidator, ImporterNode, ResolverNode, Trainer, Evaluator, InfraValidator, Pusher, BulkInferrer

from tfx.components.base import executor_spec
from tfx.components.trainer import executor as trainer_executor

from tfx.components.tuner.component import Tuner

from tfx.types import standard_artifacts, Channel
from tfx.extensions.google_cloud_ai_platform.trainer import executor as ai_platform_trainer_executor


def build_step(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        component = func(self, *args, **kwargs)
        self.components.append(component)
        return self
    return wrapper


class PipelineDef:

    def __init__(self, name: Text, bucket: Text = './bucket', metadata_connection_config: Optional[
            metadata_store_pb2.ConnectionConfig] = None):
        self.components = []

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
        self.movdel_evaluator = None
        self.pusher = None
        self.bulk_inferrer = None

    def with_sqlite_ml_metadata(self):
        metadata_connection_string = f'{self.pipeline_bucket}/{self.pipeline_name}/metadata.db'
        self.metadata_connection_config = metadata.sqlite_metadata_connection_config(
            metadata_connection_string)

        return self

    @build_step
    def from_csv(self, uri: Text, input_config: Optional[example_gen_pb2.Input] = None, output_config: Optional[example_gen_pb2.Output] = None):
        args = {
            'input': csv_input(uri),
        }

        if input_config:
            args['input_config'] = input_config

        if output_config:
            args['output_config'] = output_config

        self.example_gen = CsvExampleGen(**args)
        return self.example_gen

    @build_step
    def from_tfrecord(self, uri: Text, input_config: Optional[example_gen_pb2.Input] = None, output_config: Optional[example_gen_pb2.Output] = None):
        args = {
            'input': tfrecord_input(uri),
        }

        if input_config:
            args['input_config'] = input_config

        if output_config:
            args['output_config'] = output_config

        self.example_gen = ImportExampleGen(**args)
        return self.example_gen

    @build_step
    def from_bigquery(self, query: Text, input_config: Optional[example_gen_pb2.Input] = None, output_config: Optional[example_gen_pb2.Output] = None):
        args = {
            'query': query,
        }

        if input_config:
            args['input_config'] = input_config

        if output_config:
            args['output_config'] = output_config

        self.example_gen = BigQueryExampleGen(**args)
        return self.example_gen

    @build_step
    def from_custom_example_gen_component(self, component: BaseComponent):
        self.example_gen = component

        return self.example_gen

    @build_step
    def with_imported_schema(self, uri: Text):
        self.user_schema_importer = ImporterNode(
            instance_name='with_imported_schema',
            source_uri=uri,
            artifact_type=standard_artifacts.Schema)

        return self.user_schema_importer

    @build_step
    def generate_statistics(self):
        args = {
            'examples': self.example_gen.outputs['examples']
        }

        if self.user_schema_importer:
            args['schema'] = self.user_schema_importer.outputs['result']

        self.statistics_gen = StatisticsGen(**args)

        return self.statistics_gen

    @build_step
    def infer_schema(self):
        self.schema_gen = SchemaGen(
            statistics=self.statistics_gen.outputs['statistics'])

        return self.schema_gen

    @build_step
    def validate_input_data(self):
        args = {
            'statistics': self.statistics_gen.outputs['statistics'],
            'schema': SchemaInputs.SCHEMA_CHANNEL(self)
        }

        self.example_validator = ExampleValidator(**args)
        return self.example_validator

    @build_step
    def preprocess(self, preprocessing_fn):
        args = {
            'examples': self.example_gen.outputs['examples'],
            'preprocessing_fn': preprocessing_fn,
            'schema': SchemaInputs.SCHEMA_CHANNEL(self)
        }

        self.transform = Transform(**args)
        return self.transform

    @build_step
    def with_base_model(self, uri: Text):
        self.train_base_model = ImporterNode(
            instance_name='with_base_model',
            source_uri=uri,
            artifact_type=standard_artifacts.Model)

        return self.train_base_model

    @build_step
    def with_hyperparameters(self, uri: Text):
        self.user_hyperparameters_importer = ImporterNode(
            instance_name='with_hyperparameters',
            source_uri=uri,
            artifact_type=standard_artifacts.HyperParameters)

        return self.user_hyperparameters_importer

    @build_step
    def tune(self, tuner_fn: Optional[Union[Text, data_types.RuntimeParameter]], train_args: Optional[trainer_pb2.TrainArgs] = None, eval_args: Optional[trainer_pb2.EvalArgs] = None, example_input=None):
        args = {
            'tuner_fn': tuner_fn,
            'schema': SchemaInputs.SCHEMA_CHANNEL(self)
        }

        if train_args:
            args['train_args'] = train_args

        if eval_args:
            args['eval_args'] = eval_args

        if self.transform:
            args['transform_graph'] = self.transform.outputs['transform_graph']

        if not example_input and self.cached_example_input:
            args['examples'] = self.cached_example_input
        elif example_input:
            inputs = example_input(self)
            args['examples'] = inputs

            self.cached_example_input = self.cached_example_input or inputs
        else:
            args['examples'] = ExampleInputs.PREPROCESSED_EXAMPLES(
                self)

        self.tuner = Tuner(**args)

        return self.tuner

    @build_step
    def train(self, train_fn: Optional[Union[Text, data_types.RuntimeParameter]], train_args: Optional[trainer_pb2.TrainArgs] = None, eval_args: Optional[trainer_pb2.EvalArgs] = None, example_input=None, custom_executor_spec: Optional[executor_spec.ExecutorSpec] = None, ai_platform_args: Optional[Dict[Text, Text]] = None, custom_config: Optional[Dict[Text, Any]] = None):
        args = {
            'run_fn': train_fn,
            'schema': SchemaInputs.SCHEMA_CHANNEL(self),
            'hyperparameters': HyperParameterInputs.BEST_HYPERPARAMETERS(self)
        }

        args['custom_executor_spec'] = custom_executor_spec or executor_spec.ExecutorClassSpec(
            trainer_executor.GenericExecutor),

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

    @build_step
    def evaluate_model(self, eval_config: tfma.EvalConfig, example_input=None):
        if not example_input and self.cached_example_input:
            example_input = self.cached_example_input

        self.lasest_blessed_model_resolver = ResolverNode(
            resolver_class=latest_blessed_model_resolver.LatestBlessedModelResolver,
            model=Channel(type=standard_artifacts.Model),
            model_blessing=Channel(type=standard_artifacts.ModelBlessing))

        inputs = ExampleInputs.RAW_EXAMPLES(self)

        if example_input:
            inputs = example_input(self)
            self.cached_example_input = self.cached_example_input or inputs

        self.model_evaluator = Evaluator(
            examples=inputs,
            model=self.trainer.outputs['model'],
            eval_config=eval_config)

        return self.model_evaluator

    @build_step
    def infra_validate(self, serving_spec: infra_validator_pb2.ServingSpec, validation_spec: Optional[infra_validator_pb2.ValidationSpec] = None, request_spec: Optional[infra_validator_pb2.RequestSpec] = None, example_input=None):
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

            self.cached_example_input = self.cached_example_input or inputs
        else:
            args['examples'] = ExampleInputs.RAW_EXAMPLES(
                self)

        self.infra_validator = InfraValidator(**args)
        return self.infra_validator

    @build_step
    def push_to(self, push_destination: pusher_pb2.PushDestination, custom_config: Optional[Dict[Text, Any]] = None, custom_executor_spec: Optional[executor_spec.ExecutorSpec] = None):
        args = {
            'model': self.trainer.outputs['model'],
            'model_blessing': self.model_evaluator.outputs['blessing'],
            'push_destination': push_destination
        }

        if custom_config:
            args['custom_config'] = custom_config

        if custom_executor_spec:
            args['custom_executor_spec'] = custom_executor_spec

        if self.infra_validator:
            args['infra_blessing'] = self.infra_validator.outputs['blessing']

        self.pusher = Pusher(**args)
        return self.pusher

    def cache(self, enable_cache=True):
        self.enable_cache = enable_cache

        return self

    def with_beam_pipeline_args(self, args: Optional[List[Text]]):
        self.beam_pipeline_args = args
        return self

    def build(self) -> pipeline.Pipeline:
        args = {
            'pipeline_name': self.pipeline_name,
            'pipeline_root': f'{self.pipeline_bucket}/{self.pipeline_name}/staging',
            'components': self.components,
            'enable_cache': self.enable_cache or False,
            'metadata_connection_config': self.metadata_connection_config,
        }

        if self.beam_pipeline_args:
            args['beam_pipeline_args'] = self.beam_pipeline_args

        return pipeline.Pipeline(**args)


class ExampleInputs:

    @staticmethod
    def _get_raw_examples_channel(pipeline_def: PipelineDef):
        return pipeline_def.example_gen['examples']

    @staticmethod
    def _get_preprocessed_examples_channel(pipeline_def: PipelineDef):
        return pipeline_def.transform.outputs['transformed_examples']

    RAW_EXAMPLES = _get_preprocessed_examples_channel
    PREPROCESSED_EXAMPLES = _get_preprocessed_examples_channel


class HyperParameterInputs:
    
    @staticmethod
    def _get_best_hyperparameters(pipeline_def: PipelineDef):
        if pipeline_def.tuner:
            return pipeline_def.tuner.outputs['best_hyperparameters']

        if pipeline_def.user_hyperparameters_importer:
            return pipeline_def.user_hyperparameters_importer.outputs['result']

        return None

    BEST_HYPERPARAMETERS = _get_best_hyperparameters


class SchemaInputs:
    @staticmethod
    def _get_schema_channel(pipeline_def: PipelineDef):
        if pipeline_def.user_schema_importer:
            return pipeline_def.user_schema_importer.outputs['result']

        if pipeline_def.schema_gen:
            return pipeline_def.schema_gen.output['schema']
        
        return None

    SCHEMA_CHANNEL = _get_schema_channel