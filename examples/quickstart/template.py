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
"""Copy this file and uncomment lines based on your needs"""

from typing import List, Text, Dict, Any

from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner

import os
import absl
from functools import partial

import tensorflow as tf
import tensorflow_transform as tft
import tensorflow_model_analysis as tfma
import fluent_tfx as ftfx
import kerastuner

from tfx.proto import pusher_pb2
from tfx.proto import trainer_pb2

from tensorflow_transform.tf_metadata import schema_utils
from tfx.components.trainer.fn_args_utils import FnArgs
from tfx.components.trainer.executor import TrainerFnArgs
from tfx.components.tuner.component import TunerFnResult

from google.protobuf import text_format
from tensorflow.python.lib.io import file_io
from tensorflow_metadata.proto.v0 import schema_pb2

from kerastuner.engine import base_tuner


def _gzip_reader_fn(filenames):
    return tf.data.TFRecordDataset(filenames, compression_type='GZIP')


def _transformed_name(key):
    return key + '_xf'


def _transformed_names(keys):
    return [_transformed_name(key) for key in keys]


def _get_serve_tf_examples_fn(model, tf_transform_output):
    """Returns a function that parses a serialized tf.Example and applies TFT."""

    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples):
        """Returns the output to be used in the serving signature."""
        feature_spec = tf_transform_output.raw_feature_spec()
        # todo: remove label key from serving data (does not exist here)
        # feature_spec.pop(LABEL_KEY)
        parsed_features = tf.io.parse_example(
            serialized_tf_examples, feature_spec)

        transformed_features = model.tft_layer(parsed_features)

        return model(transformed_features)

    return serve_tf_examples_fn


# todo: determine batch size

def _input_fn(file_pattern: List[Text],
              tf_transform_output: tft.TFTransformOutput,
              batch_size: int = 32) -> tf.data.Dataset:
    """Generates features and label for tuning/training.

    Args:
            file_pattern: List of paths or patterns of input tfrecord files.
            tf_transform_output: A TFTransformOutput.
            batch_size: representing the number of consecutive elements of returned
                    dataset to combine in a single batch

    Returns:
            A dataset that contains (features, indices) tuple where features is a
                    dictionary of Tensors, and indices is a single Tensor of label indices.
    """
    transformed_feature_spec = (
        tf_transform_output.transformed_feature_spec().copy())

    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transformed_feature_spec,
        reader=_gzip_reader_fn,
        label_key=_transformed_name(_LABEL_KEY))

    return dataset

# todo: use model with constants or add hyperparameter based config?
# def _build_keras_model() -> tf.keras.Model:
#     """Builds a keras model to use for your application"""
# 		model = None
# 		return model

# def _get_hyperparameters() -> kerastuner.HyperParameters:
#     hp = kerastuner.HyperParameters()
#     # hp.Int('hidden_layer_num', 1, 3, default=2)
#     #	hp.Choice('dropout', [0.2, 0.3, 0.5], default=0.2)
#     return hp

# def _build_keras_model(hparams: kerastuner.HyperParameters) -> tf.keras.Model:
#     """Builds a keras model to use for your application"""
# 		model = None
#			.. use hparams
# 		return model

# TFX Transform will call this function.


def preprocessing_fn(inputs: Dict[Text, Any]) -> Dict[Text, Any]:
    """tf.transform's callback function for preprocessing inputs.

    Args:
            inputs: map from feature keys to raw not-yet-transformed features.

    Returns:
            Map from string feature key to transformed feature operations.
    """
    outputs = {}
    # todo: ...
    return outputs

# def tuner_fn(fn_args: FnArgs) -> TunerFnResult:
#     """Build the tuner using the KerasTuner API.
#     Args:
#       fn_args: Holds args as name/value pairs.
#         - working_dir: working dir for tuning.
#         - train_files: List of file paths containing training tf.Example data.
#         - eval_files: List of file paths containing eval tf.Example data.
#         - train_steps: number of train steps.
#         - eval_steps: number of eval steps.
#         - schema_path: optional schema of the input data.
#         - transform_graph_path: optional transform graph produced by TFT.
#     Returns:
#       A namedtuple contains the following:
#         - tuner: A BaseTuner that will be used for tuning.
#         - fit_kwargs: Args to pass to tuner's run_trial function for fitting the
#                       model , e.g., the training and validation dataset. Required
#                       args depend on the above tuner's implementation.
#     """

#     train_files = fn_args.train_files
#     eval_files = fn_args.eval_files

#     tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

#     hparams = _get_hyperparameters()

# 		# todo: many tuners are available
#     # tuner = kerastuner.Hyperband(
#     #     hypermodel=_build_keras_model,
#     #     hyperparameters=hparams,
#     #     objective=kerastuner.Objective('mean_absolute_error', 'min'),
#     #     factor=3,
#     #     max_epochs=5,
#     #     directory=fn_args.working_dir,
#     #     project_name='<fill in>')

#     # train_dataset = _input_fn(train_files, tf_transform_output)
#     # eval_dataset = _input_fn(eval_files, tf_transform_output)

#     # return TunerFnResult(
#     #     tuner=tuner,
#     #     fit_kwargs={
#     #         'x': train_dataset,
#     #         'validation_data': eval_dataset,
#     #         'steps_per_epoch': fn_args.train_steps,
#     #         'validation_steps': fn_args.eval_steps
#     #     })


# TFX Trainer will call this function.
def run_fn(fn_args: TrainerFnArgs):
    """Train the model based on given args.

    Args:
            fn_args: Holds args used to train the model as name/value pairs.
    """

    # todo: if your model uses hyperparameters uncomment below
    # hparams = fn_args.hyperparameters
    # if type(hparams) is dict and 'values' in hparams.keys():
    # 		hparams = hparams['values']

    # need schema ? uncomment below
    # schema = schema_pb2.Schema()
    # schema_text = file_io.read_file_to_string(fn_args.schema_file)
    # text_format.Parse(schema_text, schema)
    # feature_spec = schema_utils.schema_as_feature_spec(schema).feature_spec

    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

    train_dataset = _input_fn(fn_args.train_files, tf_transform_output)
    eval_dataset = _input_fn(fn_args.eval_files, tf_transform_output)

    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        # uncommend depending on hparams
        # model = _build_keras_model()
        # model = _build_keras_model(hparams=hparams)
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


def get_pipeline():
    # todo: specify evalualtion configuration (optional)
    # eval_config = tfma.EvalConfig(
    #     model_specs=[tfma.ModelSpec(label_key=LABEL_KEY)],
    #     slicing_specs=[tfma.SlicingSpec()],
    #     metrics_specs=[
    #         tfma.MetricsSpec(metrics=[
    #             tfma.MetricConfig(
    #                 class_name='BinaryAccuracy',
    #                 threshold=tfma.MetricThreshold(
    #                     value_threshold=tfma.GenericValueThreshold(
    #                         lower_bound={'value': 0.6}),
    #                     change_threshold=tfma.GenericChangeThreshold(
    #                         direction=tfma.MetricDirection.HIGHER_IS_BETTER,
    #                         absolute={'value': -1e-10})))
    #         ])
    #     ])

    # this is relative to the root of the python process dir that runs the pipeline
    # todo:
    # fn_file = 'examples/chicago_taxi_pipeline/pipeline.py'

    # todo: mix and match
    # data sources:
    # from csv
    # from tfrecords
    # from bigquery
    # from custom component

    # if you want to infer a schema, you have to generate statistics

    # model evaluation is not required for pusher, but strongly recommended

    # depending on your running platform of choice, you can add custom parameters
    # to train, tuner and infra validator for example (gcp ai platform supports it)

    return ftfx.PipelineDef(name='<fill in>') \
        #     .from_csv(uri=<specify>) \
    #     .generate_statistics() \
    #     .infer_schema() \
    #     .preprocess(fn_file) \
    #     .train(fn_file,
    #            train_args=trainer_pb2.TrainArgs(num_steps=1000),
    #            eval_args=trainer_pb2.EvalArgs(num_steps=150)) \
    #     .evaluate_model(eval_config=eval_config) \
    #     .push_to(relative_push_uri='serving_model') \
    #     .cache() \
    #     .with_sqlite_ml_metadata() \
    #     .with_beam_pipeline_args([
    #         '--direct_running_mode=multi_processing',
    #         '--direct_num_workers=0',
    #     ]) \
    .build()


if __name__ == '__main__':
    # absl.logging.set_verbosity(absl.logging.INFO)
    pipeline = get_pipeline()
    BeamDagRunner().run(pipeline)
