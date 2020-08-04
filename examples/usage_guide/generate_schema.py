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
from operator import attrgetter

import absl
import fluent_tfx as ftfx
from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner

from tfx.orchestration import metadata
from tfx.types import standard_artifacts


def get_pipeline(pipeline_def: ftfx.PipelineDef) -> ftfx.PipelineDef:
    current_dir = os.path.dirname(
        os.path.realpath(__file__))

    return pipeline_def \
        .from_csv(os.path.join(current_dir, 'data')) \
        .generate_statistics() \
        .infer_schema()


if __name__ == '__main__':
    absl.logging.set_verbosity(absl.logging.ERROR)

    bucket_uri = os.path.join(
        os.path.realpath(__file__), 'bucket')

    pipeline_def = ftfx.PipelineDef(name='schema_generation', bucket=bucket_uri) \
        .with_sqlite_ml_metadata() \

    pipeline_def = get_pipeline(pipeline_def)
    pipeline = pipeline_def.build()

    print(pipeline.components)

    BeamDagRunner().run(pipeline)

    # if you are runnign this first time
    # now, a schema should be saved by default under
    # ./bucket/SchemaGen/schema/3/schema.pbtxt
    # run numbers =
    # 1 = example gen
    # 2 = stasistics
    # 3 = schema gen
    schema_output = os.path.join(
        bucket_uri, 'schema_generation', 'staging', 'SchemaGen', 'schema', '3', 'schema.pbtxt')
    print()
    print(f'Schema at {schema_output}:')
    with open(schema_output, 'r') as f:
        print(f.read())

    # alternatively, you can use our sqlite ml metadata.db to view
    # latest schema artifact directly, by using it's uri
    metadata_connection_config = metadata.sqlite_metadata_connection_config(
        os.path.join(bucket_uri, 'schema_generation', 'metadata.db'))  # defaults to this

    with metadata.Metadata(metadata_connection_config) as store:
        schema_artifacts = store.get_artifacts_by_type(
            standard_artifacts.Schema.TYPE_NAME)

        latest_schema_uri = max(schema_artifacts, key=attrgetter(
            'last_update_time_since_epoch')).uri

        latest_schema_uri = os.path.join(latest_schema_uri, 'schema.pbtxt')

        print(f'Schema at {latest_schema_uri}:')
        with open(latest_schema_uri, 'r') as f:
            print(f.read())
