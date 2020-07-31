import pytest

import os
from typing import Text

import tensorflow as tf

from examples.chicago_taxi_pipeline.pipeline import get_pipeline
from fluent_tfx.pipeline_def import PipelineDef
from tfx.orchestration import metadata
from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner


@pytest.mark.skip('default example code from tfx broken?')
class ChicagoPipelineFTFXEndToEndTest(tf.test.TestCase):

    def setUp(self):
        super(ChicagoPipelineFTFXEndToEndTest, self).setUp()

        self._bucket_dir = os.path.join(os.path.dirname(__file__), 'bucket')
        self._initial_pipeline_def = PipelineDef(
            name='chicago_pipeline_e2e', bucket=self._bucket_dir)

    def assertComponentExecutedOnce(self, component: Text) -> None:
        component_path = os.path.join(
            self._bucket_dir, self._initial_pipeline_def.pipeline_name, component)
        self.assertTrue(tf.io.gfile.exists(component_path))
        outputs = tf.io.gfile.listdir(component_path)
        for output in outputs:
            execution = tf.io.gfile.listdir(
                os.path.join(component_path, output))
            self.assertEqual(1, len(execution))

    def assertPipelineExecution(self) -> None:
        self.assertExecutedOnce('CsvExampleGen')
        self.assertExecutedOnce('StatisticsGen')
        self.assertExecutedOnce('SchemaGen')
        self.assertExecutedOnce('Transform')
        self.assertExecutedOnce('Tuner')
        self.assertExecutedOnce('Trainer')
        self.assertExecutedOnce('Evaluator')
        self.assertExecutedOnce('Pusher')

    def testChicagoE2EFTFXPipeline(self):
        pipeline_def = get_pipeline(self._initial_pipeline_def)
        pipeline_def = pipeline_def.with_sqlite_ml_metadata().cache()

        pipeline = pipeline_def.build()
        BeamDagRunner().run(pipeline)

        artifact_root = os.path.join(
            self._bucket_dir, pipeline_def.pipeline_name)
        self.assertTrue(tf.io.gfile.exists(
            os.path.join(artifact_root, 'serving_model')))
        self.assertTrue(os.path.join(artifact_root, 'metadata.db'))
        expected_execution_count = 8  # 7 components + 1 resolver
        metadata_config = pipeline_def.metadata_connection_config
        with metadata.Metadata(metadata_config) as m:
            artifact_count = len(m.store.get_artifacts())
            execution_count = len(m.store.get_executions())
            self.assertGreaterEqual(artifact_count, execution_count)
            self.assertEqual(expected_execution_count, execution_count)

        self.assertPipelineExecution()

        BeamDagRunner().run(pipeline)

        # All executions but Evaluator and Pusher are cached.
        with metadata.Metadata(metadata_config) as m:
            # Artifact count is increased by 3 caused by Evaluator and Pusher.
            self.assertEqual(artifact_count + 3, len(m.store.get_artifacts()))
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
