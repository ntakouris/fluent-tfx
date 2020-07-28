# Lint as: python2, python3
# Copyright 2019 Google LLC. All Rights Reserved.
# Modifications Copyright 2020 Theodoros Ntakouris
# Modified parts: pipeline assembly via fluent_tfx got added
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Python source file include taxi pipeline functions and necesasry utils.

The utilities in this file are used to build a model with native Keras.
This module file will be used in Transform and generic Trainer.

The below example is end to end, provided by google as an example at
https://github.com/tensorflow/tfx/blob/master/tfx/examples/chicago_taxi_pipeline/taxi_utils_native_keras.py

The following file is modified to showcase how to create a sample taxi pipeline with fluent_tfx.
The pipeline definition is located at the bottom
"""

import fluent_tfx as ftfx

from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner

import os
from typing import List, Text, Dict, Any

import absl
import tensorflow as tf
import tensorflow_transform as tft
import tensorflow_model_analysis as tfma
from tfx.proto import pusher_pb2
from tfx.proto import trainer_pb2
from tfx.components.trainer.executor import TrainerFnArgs

# Categorical features are assumed to each have a maximum value in the dataset.
_MAX_CATEGORICAL_FEATURE_VALUES = [24, 31, 12]

_CATEGORICAL_FEATURE_KEYS = [
    'trip_start_hour', 'trip_start_day', 'trip_start_month',
    'pickup_census_tract', 'dropoff_census_tract', 'pickup_community_area',
    'dropoff_community_area'
]

_DENSE_FLOAT_FEATURE_KEYS = ['trip_miles', 'fare', 'trip_seconds']

# Number of buckets used by tf.transform for encoding each feature.
_FEATURE_BUCKET_COUNT = 10

_BUCKET_FEATURE_KEYS = [
    'pickup_latitude', 'pickup_longitude', 'dropoff_latitude',
    'dropoff_longitude'
]

# Number of vocabulary terms used for encoding VOCAB_FEATURES by tf.transform
_VOCAB_SIZE = 1000

# Count of out-of-vocab buckets in which unrecognized VOCAB_FEATURES are hashed.
_OOV_SIZE = 10

_VOCAB_FEATURE_KEYS = [
    'payment_type',
    'company',
]

# Keys
_LABEL_KEY = 'big_tipper'


def _transformed_name(key):
    return key + '_xf'


def _transformed_names(keys):
    return [_transformed_name(key) for key in keys]


def _gzip_reader_fn(filenames):
    """Small utility returning a record reader that can read gzip'ed files."""
    return tf.data.TFRecordDataset(
        filenames,
        compression_type='GZIP')


def _fill_in_missing(x):
    """Replace missing values in a SparseTensor.

    Fills in missing values of `x` with '' or 0, and converts to a dense tensor.

    Args:
      x: A `SparseTensor` of rank 2.  Its dense shape should have size at most 1
        in the second dimension.

    Returns:
      A rank 1 tensor where missing values of `x` have been filled in.
    """
    if isinstance(x, tf.sparse.SparseTensor):
        default_value = '' if x.dtype == tf.string else 0
        dense_tensor = tf.sparse.to_dense(
            tf.SparseTensor(x.indices, x.values, [x.dense_shape[0], 1]),
            default_value)
    else:
        dense_tensor = x

    return tf.squeeze(dense_tensor, axis=1)


def _get_serve_tf_examples_fn(model, tf_transform_output):
    """Returns a function that parses a serialized tf.Example and applies TFT."""

    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples):
        """Returns the output to be used in the serving signature."""
        feature_spec = tf_transform_output.raw_feature_spec()
        feature_spec.pop(_LABEL_KEY)
        parsed_features = tf.io.parse_example(
            serialized_tf_examples, feature_spec)

        transformed_features = model.tft_layer(parsed_features)

        return model(transformed_features)

    return serve_tf_examples_fn


def _input_fn(file_pattern: List[Text],
              tf_transform_output: tft.TFTransformOutput,
              batch_size: int = 200) -> tf.data.Dataset:
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


def _build_keras_model(hidden_units: List[int] = None) -> tf.keras.Model:
    """Creates a DNN Keras model for classifying taxi data.

    Args:
      hidden_units: [int], the layer sizes of the DNN (input layer first).

    Returns:
      A keras Model.
    """
    real_valued_columns = [
        tf.feature_column.numeric_column(key, shape=())
        for key in _transformed_names(_DENSE_FLOAT_FEATURE_KEYS)
    ]
    categorical_columns = [
        tf.feature_column.categorical_column_with_identity(
            key, num_buckets=_VOCAB_SIZE + _OOV_SIZE, default_value=0)
        for key in _transformed_names(_VOCAB_FEATURE_KEYS)
    ]
    categorical_columns += [
        tf.feature_column.categorical_column_with_identity(
            key, num_buckets=_FEATURE_BUCKET_COUNT, default_value=0)
        for key in _transformed_names(_BUCKET_FEATURE_KEYS)
    ]
    categorical_columns += [
        tf.feature_column.categorical_column_with_identity(  # pylint: disable=g-complex-comprehension
            key,
            num_buckets=num_buckets,
            default_value=0) for key, num_buckets in zip(
                _transformed_names(_CATEGORICAL_FEATURE_KEYS),
                _MAX_CATEGORICAL_FEATURE_VALUES)
    ]
    indicator_column = [
        tf.feature_column.indicator_column(categorical_column)
        for categorical_column in categorical_columns
    ]

    model = _wide_and_deep_classifier(
        wide_columns=indicator_column,
        deep_columns=real_valued_columns,
        dnn_hidden_units=hidden_units or [100, 70, 50, 25])
    return model


def _wide_and_deep_classifier(wide_columns, deep_columns, dnn_hidden_units):
    """Build a simple keras wide and deep model.

    Args:
      wide_columns: Feature columns wrapped in indicator_column for wide (linear)
        part of the model.
      deep_columns: Feature columns for deep part of the model.
      dnn_hidden_units: [int], the layer sizes of the hidden DNN.

    Returns:
      A Wide and Deep Keras model
    """
    # Following values are hard coded for simplicity in this example,
    # However prefarably they should be passsed in as hparams.

    # Keras needs the feature definitions at compile time.
    input_layers = {
        colname: tf.keras.layers.Input(
            name=colname, shape=(), dtype=tf.float32)
        for colname in _transformed_names(_DENSE_FLOAT_FEATURE_KEYS)
    }
    input_layers.update({
        colname: tf.keras.layers.Input(name=colname, shape=(), dtype='int32')
        for colname in _transformed_names(_VOCAB_FEATURE_KEYS)
    })
    input_layers.update({
        colname: tf.keras.layers.Input(name=colname, shape=(), dtype='int32')
        for colname in _transformed_names(_BUCKET_FEATURE_KEYS)
    })
    input_layers.update({
        colname: tf.keras.layers.Input(name=colname, shape=(), dtype='int32')
        for colname in _transformed_names(_CATEGORICAL_FEATURE_KEYS)
    })

    deep = tf.keras.layers.DenseFeatures(deep_columns)(input_layers)
    for numnodes in dnn_hidden_units:
        deep = tf.keras.layers.Dense(numnodes)(deep)
    wide = tf.keras.layers.DenseFeatures(wide_columns)(input_layers)

    output = tf.keras.layers.Dense(
        1, activation='sigmoid')(
            tf.keras.layers.concatenate([deep, wide]))
    output = tf.squeeze(output, -1)

    model = tf.keras.Model(input_layers, output)
    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(lr=0.001),
        metrics=[tf.keras.metrics.BinaryAccuracy()])
    model.summary(print_fn=absl.logging.info)
    return model


# TFX Transform will call this function.
def preprocessing_fn(inputs: Dict[Text, Any]) -> Dict[Text, Any]:
    """tf.transform's callback function for preprocessing inputs.

    Args:
      inputs: map from feature keys to raw not-yet-transformed features.

    Returns:
      Map from string feature key to transformed feature operations.
    """
    outputs = {}
    for key in _DENSE_FLOAT_FEATURE_KEYS:
        # Preserve this feature as a dense float, setting nan's to the mean.
        outputs[_transformed_name(key)] = tft.scale_to_z_score(
            _fill_in_missing(inputs[key]))

    for key in _VOCAB_FEATURE_KEYS:
        # Build a vocabulary for this feature.
        outputs[_transformed_name(key)] = tft.compute_and_apply_vocabulary(
            _fill_in_missing(inputs[key]),
            top_k=_VOCAB_SIZE,
            num_oov_buckets=_OOV_SIZE)

    for key in _BUCKET_FEATURE_KEYS:
        outputs[_transformed_name(key)] = tft.bucketize(
            _fill_in_missing(inputs[key]),
            _FEATURE_BUCKET_COUNT)

    for key in _CATEGORICAL_FEATURE_KEYS:
        outputs[_transformed_name(key)] = _fill_in_missing(inputs[key])

    outputs[_transformed_name(_LABEL_KEY)] = inputs[_LABEL_KEY]

    return outputs


# TFX Trainer will call this function.
def run_fn(fn_args: TrainerFnArgs):
    """Train the model based on given args.

    Args:
      fn_args: Holds args used to train the model as name/value pairs.
    """
    # Number of nodes in the first layer of the DNN
    first_dnn_layer_size = 100
    num_dnn_layers = 4
    dnn_decay_factor = 0.7

    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

    train_dataset = _input_fn(fn_args.train_files, tf_transform_output, 40)
    eval_dataset = _input_fn(fn_args.eval_files, tf_transform_output, 40)

    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        model = _build_keras_model(
            # Construct layers sizes with exponetial decay
            hidden_units=[
                max(2, int(first_dnn_layer_size * dnn_decay_factor**i))
                for i in range(num_dnn_layers)
            ])

    try:
        log_dir = fn_args.model_run_dir
    except KeyError:
        # TODO(b/158106209): use ModelRun instead of Model artifact for logging.
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
                                              name='examples')),
    }
    model.save(fn_args.serving_model_dir,
               save_format='tf', signatures=signatures)


def get_pipeline():
    eval_config = tfma.EvalConfig(
        model_specs=[tfma.ModelSpec(label_key='big_tipper')],
        slicing_specs=[tfma.SlicingSpec()],
        metrics_specs=[
            tfma.MetricsSpec(metrics=[
                tfma.MetricConfig(
                    class_name='BinaryAccuracy',
                    threshold=tfma.MetricThreshold(
                        value_threshold=tfma.GenericValueThreshold(
                            lower_bound={'value': 0.6}),
                        change_threshold=tfma.GenericChangeThreshold(
                            direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                            absolute={'value': -1e-10})))
            ])
        ])

    _data_root = os.path.join(os.path.dirname(
        __file__), 'data', 'big_tipper_label')
    print(f'Loading csv data from: {_data_root}')

    return ftfx.PipelineDef(name='chicago_taxi_pipeline') \
        .from_csv(uri=_data_root) \
        .generate_statistics() \
        .infer_schema() \
        .preprocess(os.path.realpath(__file__)) \
        .train(fn_file,
               train_args=trainer_pb2.TrainArgs(num_steps=1000),
               eval_args=trainer_pb2.EvalArgs(num_steps=150)) \
        .evaluate_model(eval_config=eval_config) \
        .push_to(relative_push_uri='serving_model') \
        .cache() \
        .with_sqlite_ml_metadata() \
        .with_beam_pipeline_args([
            '--direct_running_mode=multi_processing',
            '--direct_num_workers=0',
        ]) \
        .build()


if __name__ == '__main__':
    absl.logging.set_verbosity(absl.logging.INFO)
    pipeline = get_pipeline()
    BeamDagRunner().run(pipeline)