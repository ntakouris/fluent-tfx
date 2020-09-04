from typing import Text, Any, Optional, List, Dict
from tfx.components import CsvExampleGen, ImportExampleGen, ImporterNode, ResolverNode
from tfx.extensions.google_cloud_big_query.example_gen.component import BigQueryExampleGen
from tfx.types import Channel, standard_artifacts
from tfx.dsl.experimental import latest_blessed_model_resolver
from tfx.proto import example_gen_pb2


def from_csv(uri: Text, name: Optional[Text] = None,
             input_config: Optional[example_gen_pb2.Input] = None,
             output_config: Optional[example_gen_pb2.Output] = None):
    """Constructs a CsvExampleGen component by using external_input(uri)

    Args:
        uri (Text): Csv file(s) uri
        name(Optional[Text]): The optional instance_name of the component. Please use this to avoid
        same-name errors.
        input_config (Optional[example_gen_pb2.Input], optional): Defaults to None.
        output_config (Optional[example_gen_pb2.Output], optional): Defaults to None.

    Returns: CsvExampleGen
    """
    args = {
        'input_base': uri,
    }

    if name:
        args['instance_name'] = name

    if input_config:
        args['input_config'] = input_config

    if output_config:
        args['output_config'] = output_config

    return CsvExampleGen(**args)


def from_tfrecord(uri: Text, name: Optional[Text] = None,
                  input_config: Optional[example_gen_pb2.Input] = None,
                  output_config: Optional[example_gen_pb2.Output] = None):
    """Constructs an ImportExampleGen component by using external_input(uri)

    Args:
        uri (Text): TFRecord file(s) uri
        input_config (Optional[example_gen_pb2.Input], optional): Defaults to None.
        output_config (Optional[example_gen_pb2.Output], optional): Defaults to None.

    Returns: ImportExampleGen
    """
    args = {
        'input_base': uri,
    }

    if name:
        args['instance_name'] = name

    if input_config:
        args['input_config'] = input_config

    if output_config:
        args['output_config'] = output_config

    return ImportExampleGen(**args)


def from_bigquery(query: Text, name: Optional[Text] = None,
                  input_config: Optional[example_gen_pb2.Input] = None,
                  output_config: Optional[example_gen_pb2.Output] = None):
    """Constructs a BigQueryExampleGen component by using external_input(uri)

    Args:
        query (Text): The query to run
        input_config (Optional[example_gen_pb2.Input], optional): Defaults to None.
        output_config (Optional[example_gen_pb2.Output], optional): Defaults to None.

    Returns: BigQueryExampleGen
    """
    args = {
        'query': query,
    }

    if name:
        args['instance_name'] = name

    if input_config:
        args['input_config'] = input_config

    if output_config:
        args['output_config'] = output_config

    return BigQueryExampleGen(**args)


def with_imported_schema(uri: Text):
    """Constructs an ImporterNode component that imports
    the schema in the pipelineas an artifact.

    If infer_schema is called, the subsequent components will still use this
    use provided schema, but the SchemaGen component will still produce inferred
    schema artifacts.

    Args:
        uri (Text): Schema .pbtxt file uri

    Returns: ImporterNode
    """
    return ImporterNode(
        instance_name='with_imported_schema',
        source_uri=uri,
        artifact_type=standard_artifacts.Schema)


def with_base_model(uri: Text):
    """Constructs an ImporterNode component that imports a `standard_artifacts.Model`
    artifact to use as a starting point for training.

    Args:
        uri (Text): Model artifact's uri

    Returns: ImporterNode
    """
    return ImporterNode(
        instance_name='with_base_model',
        source_uri=uri,
        artifact_type=standard_artifacts.Model)


def with_hyperparameters(uri: Text):
    """Constructs an ImporterNode component that imports a `standard_artifacts.HyperParameters`
    artifact to use for future runs.

    Args:
        uri (Text): Hyperparameter artifact's uri

    Returns: ImporterNode
    """
    return ImporterNode(
        instance_name='with_hyperparameters',
        source_uri=uri,
        artifact_type=standard_artifacts.HyperParameters)


def get_latest_blessed_model_resolver():
    """Constructs a latest blessed model resolver node

    Returns: ResolverNode
    """
    return ResolverNode(
        instance_name='latest_blessed_model_resolver',
        resolver_class=latest_blessed_model_resolver.LatestBlessedModelResolver,
        model=Channel(type=standard_artifacts.Model),
        model_blessing=Channel(type=standard_artifacts.ModelBlessing))
