"""
tuning pipelines
author: shiyang lai
email: shiyanglai@uchicago.edu
"""

from pyannote.pipeline import Optimizer


def tune_pipeline(pipeline, protocol, freeze_set=None, initial_params={"onset": 0.6, "offset": 0.4, "min_duration_on": 0.0, "min_duration_off": 0.0}):
    """
    optimizing pipeline hyper-parameters
    @pipeline: the trained origional pipeline
    @protocol: the dataset
    @freeze_set: the hyper-parameters that we don't want tune
    @initial_params: initial hyper-parameters value
    """
    if freeze_set != None:
        pipeline.freeze(freeze_set)
    optimizer = Optimizer(pipeline)
    optimizer.tune(list(protocol.development()),
                    warm_start=initial_params,
                    n_iterations=20,
                    show_progress=True)
    optimized_params = optimizer.best_params
    return optimized_params