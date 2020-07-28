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

from tfx.proto import trainer_pb2, evaluator_pb2, pusher_pb2

from google.protobuf import text_format
from tensorflow.python.lib.io import file_io
from tensorflow_metadata.proto.v0 import schema_pb2
from tensorflow_transform.tf_metadata import schema_utils

# pipeline definition is at the bottom of the file

LABEL_KEY = 'lbl'
DENSE_FEATURES = ['a', 'b']
BINARY_FEATURES = ['c']


def _gzip_reader_fn(filenames):
    return tf.data.TFRecordDataset(filenames, compression_type='GZIP')


def _get_serve_tf_examples_fn(model, tf_transform_output):
    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples):
        feature_spec = tf_transform_output.raw_feature_spec()
        feature_spec.pop(LABEL_KEY)
        parsed_features = tf.io.parse_example(
            serialized_tf_examples, feature_spec)

        transformed_features = model.tft_layer(parsed_features)

        return model(transformed_features)

    return serve_tf_examples_fn


def preprocessing_fn(inputs: Dict[Text, Any]) -> Dict[Text, Any]:
    outputs = {}
    for feat in DENSE_FEATURES:
        outputs[f'{feat}_xf'] = tft.scale_to_z_score(inputs[feat])

    for feat in BINARY_FEATURES:
        outputs[feat] = inputs[feat]

    outputs[LABEL_KEY] = inputs[LABEL_KEY]
    return outputs


def _input_fn(file_pattern: List[Text],
              tf_transform_output: tft.TFTransformOutput,
              batch_size: int = 32) -> tf.data.Dataset:
    transformed_feature_spec = (
        tf_transform_output.transformed_feature_spec().copy())

    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transformed_feature_spec,
        reader=_gzip_reader_fn,
        label_key=LABEL_KEY)

    return dataset


H_SIZE = 'h_size'


def _get_hyperparameters() -> kerastuner.HyperParameters:
    hp = kerastuner.HyperParameters()
    hp.Choice(H_SIZE, [5, 10])
    return hp


def _build_keras_model(hparams: kerastuner.HyperParameters) -> tf.keras.Model:
    features_in = []
    features_in.extend(DENSE_FEATURES)
    features_in.extend(BINARY_FEATURES)

    features_in = [f'{x}_xf' for x in features_in]

    input_layers = {
        colname: tf.keras.layers.Input(
            name=colname, shape=(None, 1), dtype=tf.float32)
        for colname in features_in
    }

    x = tf.keras.layers.Concatenate(axis=-1)(input_layers.values())

    h = int(hparams.get(H_SIZE))
    x = tf.keras.layers.Dense(
        units=h, activation='relu')(x)

    out = tf.keras.layers.Dense(units=1, activation='sigmoid')(x)

    model = tf.keras.Model(input_layers, out)

    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=[tf.keras.metrics.BinaryAccuracy()])

    model.summary(print_fn=logging.info)
    return model


def tuner_fn(fn_args: FnArgs) -> TunerFnResult:
    train_files = fn_args.train_files
    eval_files = fn_args.eval_files

    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

    hparams = _get_hyperparameters()

    tuner = kerastuner.Hyperband(
        hypermodel=_build_keras_model,
        hyperparameters=hparams,
        objective=kerastuner.Objective('binary_accuracy', 'max'),
        factor=3,
        max_epochs=2,
        directory=fn_args.working_dir,
        project_name='ftfx:simple_e2e')

    train_dataset = _input_fn(train_files, tf_transform_output)
    eval_dataset = _input_fn(eval_files, tf_transform_output)

    return TunerFnResult(
        tuner=tuner,
        fit_kwargs={
            'x': train_dataset,
            'validation_data': eval_dataset,
            'steps_per_epoch': fn_args.train_steps,
            'validation_steps': fn_args.eval_steps
        })


def run_fn(fn_args: TrainerFnArgs):
    hparams = fn_args.hyperparameters
    if type(hparams) is dict and 'values' in hparams.keys():
        hparams = hparams['values']

    schema = schema_pb2.Schema()
    schema_text = file_io.read_file_to_string(fn_args.schema_file)
    text_format.Parse(schema_text, schema)
    feature_spec = schema_utils.schema_as_feature_spec(schema).feature_spec

    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

    train_dataset = _input_fn(fn_args.train_files, tf_transform_output)
    eval_dataset = _input_fn(fn_args.eval_files, tf_transform_output)

    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        model = _build_keras_model(hparams=hparams)
    try:
        log_dir = fn_args.model_run_dir
    except KeyError:
        log_dir = os.path.join(os.path.dirname(
            fn_args.serving_model_dir), 'logs')

    # Write logs to path
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, update_freq='batch')

    model.fit(
        train_dataset,
        steps_per_epoch=fn_args.train_steps,
        validation_data=eval_dataset,
        validation_steps=fn_args.eval_steps,
        callbacks=[tensorboard_callback])

    signatures = {
        'serving_default':
        _get_serve_tf_examples_fn(model,
                                  tf_transform_output).get_concrete_function(
            tf.TensorSpec(
                shape=[None],
                dtype=tf.string,
                name='examples'))
    }
    model.save(fn_args.serving_model_dir,
               save_format='tf', signatures=signatures)


# @component
# def tips_printer(model: InputArtifact[Model],
#                  hyperparameters: InputArtifact[HyperParameters]) -> None:
#     logging.info(
#         f'for next runs, you can use with_base_model({model.uri})')
#     logging.info(
#         f'for next runs, you can use with_hyperparameters({hyperparameters.uri})')


# def tips_printer_build_fn(components):
#     return tips_printer(hyperparameters=components['tuner'].outputs['best_hyperparameters'],
#                         model=components['trainer'].outputs['model'])


if __name__ == '__main__':
    absl.logging.set_verbosity(absl.logging.ERROR)

    current_dir = os.path.dirname(
        os.path.realpath(__file__))

    this_file = os.path.realpath(__file__)
    print(f'Using {this_file} for preprocessing, training and tuning functions')

    bucket_uri = os.path.join(current_dir, 'bucket')

    eval_config = tfma.EvalConfig(
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

    pipeline = ftfx.PipelineDef(name='simple_e2e', bucket=bucket_uri) \
        .with_sqlite_ml_metadata() \
        .from_csv(os.path.join(current_dir, 'data/')) \
        .generate_statistics() \
        .infer_schema(infer_feature_shape=True) \
        .preprocess(this_file) \
        .tune(this_file,
              train_args=trainer_pb2.TrainArgs(num_steps=5),
              eval_args=trainer_pb2.EvalArgs(num_steps=3)) \
        .train(this_file,
               train_args=trainer_pb2.TrainArgs(num_steps=10),
               eval_args=trainer_pb2.EvalArgs(num_steps=5)) \
        .evaluate_model(eval_config=eval_config,
                        example_provider_component=ftfx.input_builders.from_csv(
                            os.path.join(current_dir, 'data/'),
                            name='eval_example_gen')) \
        .push_to(relative_push_uri='serving') \
        .build()

    BeamDagRunner().run(pipeline)
