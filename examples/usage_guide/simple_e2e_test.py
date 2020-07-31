import os
import logging
import shutil
from typing import Text

import tensorflow as tf

from examples.usage_guide.simple import get_pipeline
from fluent_tfx.pipeline_def import PipelineDef
from tfx.orchestration import metadata
from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner
import tfx.components


class SimpleFTFXE2ETest(tf.test.TestCase):

    def setUp(self):
        super(SimpleFTFXE2ETest, self).setUp()
        self._bucket_dir = os.path.join(os.path.dirname(__file__), 'tmpbucket')
        self._clean_bucket_dir()
        self._initial_pipeline_def = PipelineDef(
            name='simple_e2e_test', bucket=self._bucket_dir)

    def _clean_bucket_dir(self):
        if os.path.exists(self._bucket_dir):
            shutil.rmtree(self._bucket_dir)

    def tearDown(self):
        self._clean_bucket_dir()

    def assertComponentExecutedOnce(self, component: Text) -> None:
        component_path = os.path.join(
            self._bucket_dir, self._initial_pipeline_def.pipeline_name, 'staging', component)
        self.assertTrue(tf.io.gfile.exists(component_path))
        outputs = tf.io.gfile.listdir(component_path)
        for output in outputs:
            execution = tf.io.gfile.listdir(
                os.path.join(component_path, output))
            self.assertEqual(1, len(execution))

    def assertPipelineExecution(self) -> None:
        self.assertComponentExecutedOnce('CsvExampleGen')
        self.assertComponentExecutedOnce('StatisticsGen')
        self.assertComponentExecutedOnce('SchemaGen')
        self.assertComponentExecutedOnce('ExampleValidator')
        self.assertComponentExecutedOnce('Tuner')
        self.assertComponentExecutedOnce('Trainer')
        self.assertComponentExecutedOnce('Evaluator')
        self.assertComponentExecutedOnce('Pusher')
        self.assertComponentExecutedOnce('BulkInferrer')

    def assertComponentsExistInDef(self, pipeline_def: PipelineDef, name: Text, type) -> None:
        self.assertTrue(name in pipeline_def.components)
        self.assertIsInstance(pipeline_def.components[name], type)

    def assertAllComponentsExistInDef(self, pipeline_def: PipelineDef) -> None:
        self.assertComponentsExistInDef(
            pipeline_def, 'example_gen', tfx.components.CsvExampleGen)
        self.assertComponentsExistInDef(
            pipeline_def, 'statistics_gen', tfx.components.StatisticsGen)
        self.assertComponentsExistInDef(
            pipeline_def, 'schema_gen', tfx.components.SchemaGen)
        self.assertComponentsExistInDef(
            pipeline_def, 'example_validator', tfx.components.ExampleValidator)
        self.assertComponentsExistInDef(
            pipeline_def, 'transform', tfx.components.Transform)
        self.assertComponentsExistInDef(
            pipeline_def, 'tuner', tfx.components.tuner.component.Tuner)
        self.assertComponentsExistInDef(
            pipeline_def, 'trainer', tfx.components.Trainer)
        self.assertComponentsExistInDef(
            pipeline_def, 'model_evaluator', tfx.components.Evaluator)
        self.assertComponentsExistInDef(
            pipeline_def, 'infra_validator', tfx.components.InfraValidator)
        self.assertComponentsExistInDef(
            pipeline_def, 'pusher', tfx.components.Pusher)
        self.assertComponentsExistInDef(
            pipeline_def, 'tips_printer', tfx.components.base.base_component.BaseComponent)
        self.assertComponentsExistInDef(
            pipeline_def, 'bulk_inferrer', tfx.components.BulkInferrer)

    def testSimpleE2EFTFXPipeline(self):
        pipeline_def = get_pipeline(self._initial_pipeline_def)
        pipeline_def = pipeline_def.with_sqlite_ml_metadata().cache()

        expected_execution_count = 14
        self.assertLen(pipeline_def.components.values(),
                       expected_execution_count)

        self.assertAllComponentsExistInDef(pipeline_def)
        pipeline = pipeline_def.build()

        BeamDagRunner().run(pipeline)

        artifact_root = os.path.join(
            self._bucket_dir, pipeline_def.pipeline_name)
        self.assertTrue(os.path.join(artifact_root, 'metadata.db'))
        metadata_config = pipeline_def.metadata_connection_config
        with metadata.Metadata(metadata_config) as m:
            artifact_count = len(m.store.get_artifacts())
            execution_count = len(m.store.get_executions())
            self.assertGreaterEqual(artifact_count, execution_count)
            self.assertEqual(expected_execution_count, execution_count)

        self.assertPipelineExecution()

        BeamDagRunner().run(pipeline)

        # All executions are cached, except resolvers and blessings
        with metadata.Metadata(metadata_config) as m:
            self.assertEqual(artifact_count + 4, len(m.store.get_artifacts()))
            artifact_count = len(m.store.get_artifacts())
            self.assertEqual(expected_execution_count * 2,
                             len(m.store.get_executions()))

        BeamDagRunner().run(pipeline)

        # Asserts cache execution.
        with metadata.Metadata(metadata_config) as m:
            # Artifact count is unchanged.
            self.assertEqual(artifact_count, len(m.store.get_artifacts()))
            self.assertEqual(expected_execution_count * 3,
                             len(m.store.get_executions()))


if __name__ == '__main__':
    tf.compat.v1.enable_v2_behavior()
    tf.test.main()
