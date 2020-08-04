# MIT License

# Copyright(c) 2020 Theodoros Ntakouris

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files(the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and / or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import os
import logging
from typing import Dict, Text, Any, Optional, List

import absl
import fluent_tfx as ftfx
import tensorflow as tf
import kerastuner
from kerastuner.tuners import Hyperband
import tensorflow_transform as tft
import tensorflow_model_analysis as tfma
from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner

from tfx.dsl.component.experimental.decorators import component
from tfx.dsl.component.experimental.annotations import InputArtifact
from tfx.types.standard_artifacts import Model, HyperParameters, Schema

from tfx.components.trainer.fn_args_utils import FnArgs
from tfx.components.trainer.executor import TrainerFnArgs
from tfx.components.tuner.component import TunerFnResult

from tfx.proto import trainer_pb2, evaluator_pb2, pusher_pb2, infra_validator_pb2

from google.protobuf import text_format
from tensorflow.python.lib.io import file_io
from tensorflow_metadata.proto.v0 import schema_pb2
from tensorflow_transform.tf_metadata import schema_utils

from examples.usage_guide.model_code import LABEL_KEY

from tfx.dsl.component.experimental.decorators import component
from tfx.dsl.component.experimental.annotations import InputArtifact
from tfx.types.standard_artifacts import Model, HyperParameters, Schema

"""
This file contains code the pipeline orchestration that uses user provided code from
`model_code.py`. Only the data sources and evaluation configuration is declared with high fidelity
here, basically.
"""


@component
def tips_printer(model: InputArtifact[Model],
                 hyperparameters: InputArtifact[HyperParameters]) -> None:
    # just showcase that you can hook onto any pre-existing component
    logging.info(
        f'for next runs, you can use with_base_model({model.uri})')
    logging.info(
        f'for next runs, you can use with_hyperparameters({hyperparameters.uri})')


def tips_printer_build_fn(components):
    return tips_printer(hyperparameters=components['tuner'].outputs['best_hyperparameters'],
                        model=components['trainer'].outputs['model'])


def _get_eval_config() -> tfma.EvalConfig:
    return tfma.EvalConfig(
        model_specs=[tfma.ModelSpec(
            label_key=LABEL_KEY)],
        slicing_specs=[tfma.SlicingSpec()],
        metrics_specs=[
            tfma.MetricsSpec(metrics=[
                tfma.MetricConfig(
                    class_name='BinaryAccuracy',
                    threshold=tfma.MetricThreshold(
                        value_threshold=tfma.GenericValueThreshold(
                            lower_bound={'value': 0.01}),
                        change_threshold=tfma.GenericChangeThreshold(
                            direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                            absolute={'value': 1e-10})))
            ])
        ])


def get_pipeline(pipeline_def: ftfx.PipelineDef) -> ftfx.PipelineDef:
    current_dir = os.path.dirname(
        os.path.realpath(__file__))

    user_code_file = os.path.join(os.path.dirname(
        os.path.realpath(__file__)), 'model_code.py')
    logging.info(
        f'Using {user_code_file} for preprocessing, training and tuning functions')

    return pipeline_def.from_csv(os.path.join(current_dir, 'data')) \
        .generate_statistics() \
        .infer_schema(infer_feature_shape=True) \
        .validate_input_data() \
        .preprocess(user_code_file) \
        .tune(user_code_file,
              train_args=trainer_pb2.TrainArgs(num_steps=5),
              eval_args=trainer_pb2.EvalArgs(num_steps=3)) \
        .train(user_code_file,
               train_args=trainer_pb2.TrainArgs(num_steps=10),
               eval_args=trainer_pb2.EvalArgs(num_steps=5)) \
        .evaluate_model(eval_config=_get_eval_config()) \
        .infra_validate(serving_spec=infra_validator_pb2.ServingSpec(
            tensorflow_serving=infra_validator_pb2.TensorFlowServing(
                tags=['latest']),
            local_docker=infra_validator_pb2.LocalDockerConfig()
        ),
        request_spec=infra_validator_pb2.RequestSpec(
            tensorflow_serving=infra_validator_pb2.TensorFlowServingRequestSpec()
        )) \
        .push_to(relative_push_uri='serving') \
        .bulk_infer(example_provider_component=ftfx.input_builders.from_csv(
            uri=os.path.join(current_dir, 'to_infer'),
            name='bulk_infer_example_gen'
        )) \
        .add_custom_component(name='tips_printer', component=tips_printer_build_fn)


if __name__ == '__main__':
    absl.logging.set_verbosity(absl.logging.ERROR)

    bucket_uri = os.path.join(os.path.dirname(__file__), 'bucket')

    pipeline_def = ftfx.PipelineDef(name='simple_e2e', bucket=bucket_uri) \
        .with_sqlite_ml_metadata()

    pipeline_def = get_pipeline(pipeline_def)

    # you can also do:
    print('Exposed pipeline components dict:')
    print(pipeline_def.components)

    pipeline = pipeline_def.build()
    BeamDagRunner().run(pipeline)
