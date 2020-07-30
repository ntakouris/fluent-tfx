import pytest
import os
from fluent_tfx import PipelineDef


@pytest.fixture
def pipeline_def(tmp_path):
    bucket_uri = os.path.join(tmp_path.absolute(), 'bucket')
    return PipelineDef('pipeline_def_name', bucket=bucket_uri)
