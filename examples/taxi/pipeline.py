import fluent_tfx as ftfx

from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner



def get_pipeline():
    return ftfx.PipelineDef(name='taxi_pipeline') \
        .with_sqlite_ml_metadata() \
        .from_csv(uri='./examples/taxi/data') \
        .generate_statistics() \
        .infer_schema() \
        .preprocess() \
        .tune() \
        .train() \
        .evaluate_model() \
        .push_to() \
        .infra_validate() \
        .build() 


if __name__ == '__main__':
    pipeline = get_pipeline()
    BeamDagRunner().run(pipeline)
