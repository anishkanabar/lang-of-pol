'''
File: atczero.py
Brief: Loader for ATC0 corpus.
Authors: William Dolan <wdolan@uchicago.edu>
'''

import os
import logging
import pandas as pd
from numbers import Real
from asr_dataset.base import AsrETL
from asr_dataset.constants import DATASET_DIRS, Cluster, DataSizeUnit


logger = logging.getLogger('asr.etl.atczero')


class ATCZeroETL(AsrETL):
    
    def __init__(self, cluster: Cluster=Cluster.RCC):
        super().__init__('atczero')
        self.transcripts_dir = DATASET_DIRS[cluster]['atczero_data']
        self.SAMPLE_RATE = 8000  # Hz
        self.WINDOW_LEN = 0.02  # Sec

    
    def extract(self) -> pd.DataFrame:
        """
        Collect info on audio files and transcripts in their original form.

        Returns: DataFrame with columns:
            audio - (str) full path to audio file
            text - (str) transcript text
            duration - (float) length of audio in seconds
        """
        data = self._create_manifest()
        self.describe(data, "-extracted")
        return data


    def transform(self, data: pd.DataFrame, sample_rate: int=None) -> pd.DataFrame:
        """
        Writes new audio files at the utterance level.

        This function is intended to run ONCE over the lifetime of the dataset,
        but is a NO-OP if called again. To re-transform the dataset, delete the
        output of this function.
        """
        sample_rate = sample_rate if sample_rate is not None else self.SAMPLE_RATE
        # Filter out bad audio
        unfiltered_data = data
        data = self._filter_exists(data, "original_audio")
        data = self._filter_empty(data, sample_rate)
        logger.info("Skipping check for corrupted audio (usually nothing is corrupted)")
        #data = self._filter_corrupt(data, "original_audio")
        # Write new files to disk
        self._write_utterances(data, sample_rate)
        # Filter out bad audio again after writing
        data = self._filter_exists(data, "audio")
        data = self._filter_empty(data, sample_rate)
        logger.info("Skipping check for corrupted audio (usually nothing is corrupted)")
        #data = self._filter_corrupt(data, "audio")
        self.describe(data, "-transformed")
        # Delete bad utterance files
        dropped_data = unfiltered_data.loc[unfiltered_data.index.difference(data.index)]
        for dropped_row in dropped_data.itertuples():
            dropped_row = dropped_row._asdict()
            if os.path.exists(dropped_row['audio']):
                os.remove(dropped_row['audio'])
        return data


    def load(self, 
            data: pd.DataFrame=None, 
            qty:Real=None, 
            units: DataSizeUnit=None) -> pd.DataFrame:
        """
        Collect info on the transformed audio files and transcripts.
        Does NOT load waveforms into memory.

        Returns: DataFrame with same columns as extract()
        """
        if data is None:
            data = self._create_manifest()
        data = self._filter_exists(data, "audio", log=False)
        data = self._sample(data, qty, units)
        self.describe(data, '-loaded')
        return data
                    

    def _create_manifest(self) -> pd.DataFrame:
        """
        Collect info on audio files and transcripts.
        Returns: DataFrame with columns:
            audio - full path to utterance audio file
            text - transcript text
            duration - utterance audio length in seconds
            offset - utterance offset from beginning of context audio
            original_audio - path to audio file with multiple utterances
            location - airport identifier
            speaker - air traffic controller id
            recipient - airplane id
            transcriber - transcriber id
        """
        data = pd.read_csv(self.transcripts_dir + '/atc0.csv')

        # Filter mal-formed start and end columns
        start_filter = data['start'].apply(lambda x: x.replace('.', '', 1).isdigit())
        end_filter = data['end'].apply(lambda x: x.replace('.', '', 1).isdigit())
        data = data.loc[start_filter & end_filter]

        # Create new columns
        msPerSec = 1000
        offset = data['start'].astype(float)
        duration = data['end'].astype(float) - offset
        start_ms = (offset * msPerSec).astype(int).astype('str')  # cast to int for cleaner filenames
        end_ms = ((offset + duration) * msPerSec).astype(int).astype('str')
        audio_names = start_ms.str.cat(end_ms, "_") + ".flac"
        original_paths = self.transcripts_dir + os.sep + data['filePath']
        audio_paths = original_paths.str.replace('audio', 'audio/utterances', regex=False)
        audio_paths = audio_paths.str.replace('\.(sph|wav)', '/', regex=True)
        audio_paths = audio_paths.str.cat(audio_names)
        data = data.assign(
            offset=offset, 
            duration=duration, 
            original_audio=original_paths,
            audio=audio_paths)
            
        # Drop uninteresting columns
        data = data.drop(['start', 'end', 'filePath'], axis = 1)

        data = data.rename(columns={"transcription": "text"})
        return data

