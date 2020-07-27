# fluent-tfx

Fluent TFX provides an easy-to-use api over TFX. If you are already using tensorflow or keras for your models, this is an easy to use api that will have your model up and running in an e2e scenario quickly, without the need to waste a significant amount of time of learning the internals of tfx to get something working.

![Tests](https://github.com/ntakouris/fluent-tfx/workflows/Test%20Python%20Package/badge.svg)

![PyPi](https://github.com/ntakouris/fluent-tfx/workflows/Upload%20Python%20Package/badge.svg)

`pip install fluent-tfx`

```python
import fluent_tfx as ftfx
```

## Goals of this package

Create an easy to use API for the creation, running and orchestration of TFX pipelines.

## Usage

This is a lightweight api to aid with the construction of tfx pipelines. Every side-effect and produced artifact is 100% compatible with the rest of the tfx sybsystem, including but not limited to: all the supported Beam runners (ex. Local with `BeamDagRunner`, Kubeflow, Airflow, Dataflow, etc.), custom components, ML Metadata artifacts.

It provides several shortcut functions and utilities for easier, more readable, compact and expressive pipeline definitions, with sensible defaults. Some assumptions are made about specific inputs and outputs of the componens which are described after this small example:

This is what you need to get started by using fluent tfx, instead of ~ 20 files produced with the tfx template cli commands:

### Pipeline Creation

```python
# file pipeline.py
import fluent_tfx as ftfx

def get_pipeline():
    return ftfx.PipelineDef(name='taxi_pipeline') \
        .with_sqlite_ml_metadata() \ # or provide a different configuration in the constructor
        .from_csv(uri='./examples/taxi/data') \ # or use bigquery/tfrecord/custom components
        .generate_statistics() \ # or not (optional)
        .infer_schema() \ # or use with_imported_schema(<uri>) to load your schema and detect anomalies
        .preprocess(<preprocessing_fn>) \
        .tune(<tune_args>) \ # or use with_hyperparameters(<uri>) to import a best set of hyperparameters straight to the model--or skip tuning and just use constants on .train()
        .train(<trainer_fn>, <train_args>) \
        .evaluate_model(<eval_args>) \ # evaluate against baseline and bless model
        .infra_validate(<args>) \
        .push_to(<pusher_args>) \
        .cache() \ # optional
        .with_beam_pipeline_args(<args>) \ # optional too
        .build()


# run normally with:
# python -m pipeline
if __name__ == '__main__':
    pipeline = get_pipeline()
    BeamDagRunner().run(pipeline)

```

## Assumptions and Degrees of Freedom

Assumptions are related to component dag wiring, paths and naming.

**Paths**

- `PipelineDef` needs a `pipeline_name` and an optional `bucket` path.
- Binary/Temporary/Staging artifacts are stored under `{bucket}/{name}/staging`
- Default ml metadata sqlite path is set to `{bucket}/{name}/metadata.db` unless specified otherwise
- `bucket` defaults to `./bucket`
- Pusher's `relative_push_uri` will publish the model to `{bucket}/{name}/{relative_push_uri}`

**Component IO**

- An input, or an `example_gen` component provides `.tfrecord`s (probably in gzipped format) to next components
- Fluent TFX follows the TFX naming of default components for everything. When providing custom components, make sure that inputs and outputs are on par with TFX.
- For example, your custom `example_gen` component should have a `.outputs['examples']` attribute

**Component Wiring Defaults**

- If a user provided schema uri is provided, it will be used for data validation, transform, etc. The generated schema component will still generate artifacts if declared
- If a user did not provide a model evaluation step, it will not be wired to the pusher
- The default training input source are transform outputs. The user can specify if he wants the raw tf records instead
- If hyperparameters are specified and tuner is declared, tuner will still test configurations and produce hyperparameter artifacts, but the provided hyperparameters will be used

## Utilities

WIP


### Artifact browsing

WIP

### Runners

There is no extra effort required to run the pipeline on different runners, nor extra dependencies required: `PipelineDef` produces a vanilla tfx pipeline.

However, if you are using `ftfx` utilities inside your pipeline functions, [be sure to include this package in your requirements.txt beam argument](https://beam.apache.org/documentation/sdks/python-pipeline-dependencies/)

## Examples

Example usages are available under `<repo root>/examples`.

There is an ongoing effort to keep up with most of [tensorflow/tfx/examples](https://github.com/tensorflow/tfx/tree/master/tfx/examples/).

### What made me create this fluent api

    - The TFX api is aimed for maximum flexibility. A very big portion of machine learning pipelines can be created with a much less verbose and ultimately, simpler api.
    - By the time of creation (mid 2020), the TFX demos are all over the place, with deprecated usage in many places, the documentation is lacking and there are a lot of issues that need fixing.
    - The default scaffolding is horrible. On the one hand, it makes it easy to get started, but on the other hand, it creates 10~20 files that make it hard to keep track of everything if you are new to this kind of engineering.
    - Why use scaffolding anyway? The default TFX api is flexible as stated above, but there is (1) too much magic going on and (2) lot's of components IOs could be routed automatically.
    - The pipeline definition is simply too much LOC for no apparent reason and the examples are lengthy

### What this package contains/is going to do

    - Provide an easy to use, fluent API for configuration, instead of scaffolding
    - Support all the runners and the functionality of tfx
    - Use as much code from tfx as possible, including components, internally
    - Keep the pb2 (protocol buffers) on the extenral api because they are a powerful tool
    - Keep usage restrictions to a sensible minimum, while enabling users to specify custom components and logic wherever possible
    - Provide utility functions and components for some tfx edge cases and boilerplate code

### What this package is not

    - First of all, it is not a non-opinionated way of making machine learning end to end pipelines
    - This does not automatically solve all data engineering and good statistical practises problems
    - There is going to be no estimator support for anything, sorry. Please migrate to native keras.

Please be advised, this project is aimed to make the majority of machine learning deployment to production tasks easier. Some parts of the API might take an opinionated approach to specific problems.

If you've got suggestions for improvement, please create a new issue and we can chat about it :).

### Licensing Notice

Main source code of this package, under the directory `fluent_tfx` is released under the MIT License.

Every example under the directory `/examples` is released under the APACHE 2.0 license, copyrighted by TFX Authors and Google LLC, with any modified parts being stated at the top of the notice on each file.
