import argparse
from pyannote.audio import Model
from evaluation import calculate_detection_error_rate, test_performance, calculate_segmentation_precision
from data_format import load_data
from tunning import tune_pipeline
from pyannote.audio.pipelines import SpeakerSegmentation as SpeakerSegmentationPipeline
from pyannote.audio.pipelines import VoiceActivityDetection as VoiceActivityDetectionPipeline
from utils import generate_ramdom_code


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test the performance of checkpoints.')
    parser.add_argument('checkpoint', help='the path of the checkpoint')
    parser.add_argument('corpus_name', default='ami', help='the name of the corpus (default: AMI)')
    parser.add_argument('-s', '--sub_corpus', type=str ,default='only_words', help='the name of the subcorpus (default: only_words)')
    parser.add_argument('-t', '--tunning', default=0, help='tune hyperparameters or not.')
    args = parser.parse_args()
    code = generate_ramdom_code()
    print('Start.')
    model = Model.from_pretrained(args.checkpoint)
    print('Model is loaded.')
    protocol = load_data(args.corpus_name, args.sub_corpus)
    print('Data is loaded.')
    if args.checkpoint.split('/')[5] == 'scd':
        pipeline = SpeakerSegmentationPipeline(segmentation=model)
        print('Pipeline is created.')
        if args.tunning == 0:
            pipeline = pipeline.instantiate({'min_duration_on': 0.0, 'min_duration_off': 0.0,
                                            'onset': 0.84, 'offset': 0.46, 'stitch_threshold': 0.39})
        else:
            hyper_params = tune_pipeline(pipeline, protocol, initial_params={'min_duration_on': 0.0, 'min_duration_off': 0.0,
                                            'onset': 0.84, 'offset': 0.46, 'stitch_threshold': 0.39})
            pipeline = pipeline.instantiate(hyper_params)
        test_performance(pipeline, protocol, calculate_segmentation_precision, code)
    elif args.checkpoint.split('/')[5] == 'vad':
        pipeline = VoiceActivityDetectionPipeline(segmentation=model)
        print('Pipeline is created.')
        if args.tunning == 0:
            pipeline = pipeline.instantiate({"onset": 0.767, "offset": 0.377, "min_duration_on": 0.136, "min_duration_off": 0.067})
        else:
            hyper_params = tune_pipeline(pipeline, protocol, initial_params={"onset": 0.767, "offset": 0.377,
                                         "min_duration_on": 0.136, "min_duration_off": 0.067})
            pipeline = pipeline.instantiate(hyper_params)
        test_performance(pipeline, protocol, calculate_detection_error_rate, code)
    print('Finish.')