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

from tfx.dsl.component.experimental.decorators import component
from tfx.dsl.component.experimental.annotations import InputArtifact
from tfx.types.standard_artifacts import Schema


@component
def schema_printer(schema_generated: InputArtifact[Schema],
                   schema_provided: InputArtifact[Schema]) -> None:
    print(f'uri of schema_generated: {schema_generated.uri}')
    print(f'uri of schema_provided: {schema_provided.uri}')


def schema_printer_build_fn(components):
    return schema_printer(schema_generated=components['schema_gen'].outputs['schema'],
                          schema_provided=components['user_schema_importer'].outputs['result'])


def get_pipeline(pipeline_def: ftfx.PipelineDef) -> ftfx.PipelineDef:
    current_dir = os.path.dirname(
        os.path.realpath(__file__))
    return pipeline_def \
        .from_csv(os.path.join(current_dir, 'data')) \
        .generate_statistics() \
        .infer_schema() \
        .with_imported_schema(os.path.join(current_dir, 'saved', 'schema')) \
        .add_custom_component(name='schema_printer', component=schema_printer_build_fn)


if __name__ == '__main__':
    absl.logging.set_verbosity(absl.logging.ERROR)

    bucket_uri = os.path.join(os.path.dirname(__file__), 'bucket')
    pipeline_def = ftfx.PipelineDef(
        name='custom_component_schema', bucket=bucket_uri).with_sqlite_ml_metadata()

    pipeline_def = get_pipeline(pipeline_def)
    pipeline = pipeline_def.build()

    BeamDagRunner().run(pipeline)
