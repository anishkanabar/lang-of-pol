"""
models and pipelines evaluation
author: shiyang lai
email: shiyanglai@uchicago.edu
"""

from pyannote.metrics.detection import DetectionErrorRate
from pyannote.metrics.diarization import DiarizationErrorRate
from pyannote.metrics.segmentation import SegmentationPrecision


def calculate_detection_error_rate(test_prediction, protocol, code):
    """
    calculate the detection error rate
    @test_prediction: the prediction given by vad model based on test files
    @protocol: the reference files
    """
    metric = DetectionErrorRate()
    for hypothesis, reference in zip(test_prediction, protocol.test()):
        _ = metric(reference['annotation'], hypothesis, uem=reference['annotated'])
    # save result to a csv file
    metric.report().to_csv(f'/Users/shiyang/Desktop/NIH/git/asr/results/shiyang_test/vad/log-{code}.csv')
    return 'detection error rate', abs(metric)


def calculate_diarization_error_rate(test_prediction, protocol, code):
    """
    calculate the diarization error rate
    @test_prediction: the prediction given by dia model based on test files
    @protocol: the reference files
    """
    metric = DiarizationErrorRate()
    for hypothesis, reference in zip(test_prediction, protocol.test()):
        _ = metric(reference['annotation'], hypothesis, uem=reference['annotated'])
    # save result to a csv file
    metric.report().to_csv(f'/Users/shiyang/Desktop/NIH/git/asr/results/shiyang_test/dia/log-{code}.csv')
    return 'diarization error rate', abs(metric)


def calculate_segmentation_precision(test_prediction, protocol, code):
    """
    calculate the segmentation precision
    @test_prediction: the prediction given by scd model based on test files
    @protocol: the reference files
    """
    metric = SegmentationPrecision()
    for hypothesis, reference in zip(test_prediction, protocol.test()):
        _ = metric(reference['annotation'], hypothesis, uem=reference['annotated'])
    # save result to a csv file
    metric.report().to_csv(f'/Users/shiyang/Desktop/NIH/git/asr/results/shiyang_test/scd/log-{code}.csv')
    return 'segmentation precision', abs(metric)


def test_performance(pipeline, protocol, metric, code):
    """
    test the performance of the model
    @pipeline: the tuned pipeline
    @protocol: test files
    """
    test_prediction = []
    for test_file in protocol.test():
        vad = pipeline(test_file['audio'])
        test_prediction.append(vad)
    # calculate the performance
    name, result = metric(test_prediction, protocol, code)
    print(f"The {name} is {result * 100: .3f}%")