import os
import shutil
import logging
from speechbrain.dataio.dataio import read_audio
import glob


try:
    import pandas as pd
except ImportError:
    err_msg = (
        "The optional dependency pandas must be installed to run this recipe.\n"
    )
    err_msg += "Install using `pip install pandas`.\n"
    raise ImportError(err_msg)

logger = logging.getLogger(__name__)


def prepare_nih(data_folder, save_folder, skip_prep=False):
    """
    This function prepares the NIH dataset.
    If the folder does not exist, the zip file will be extracted.
    @:param
        data_folder : path to NIH dataset.
        save_folder: path where to store the manifest csv files.
        skip_prep: If True, skip data preparation.
    """

    if skip_prep:
        return

    # If the data folders do not exist, we need to extract the data
    if not os.path.isdir(os.path.join(data_folder, "data_nih/wav")):
        # Check for zip file and download if it doesn't exist
        zip_location = os.path.join(data_folder, "data_nih.tgz")
        logger.info("Extracting data_nih.tgz...")
        shutil.unpack_archive(zip_location, data_folder)
        wav_dir = os.path.join(data_folder, "data_nih/wav")
        tgz_list = glob.glob(wav_dir + "/*.tar.gz")
        for tgz in tgz_list:
            shutil.unpack_archive(tgz, wav_dir)
            os.remove(tgz)

    # this is the process of creating train, dev, and test samples.
    split_nih(save_folder, data_folder)


def split_nih(save_folder, data_folder): # this function needs to be rewritten
    """"
    This function generate train, dev, test data set.
    Following the structure of other recipes
    @:param
        save_folder : path where to store the manifest csv files.
        data_folder : path to NIH dataset.
    """
    # Create filename-to-transcript dictionary (this part need to be rewritten)
    filename2transcript = {}
    with open(
            os.path.join(
                data_folder, "data_niih/transcript/nih_transcript_v0.1.txt"
            ),
            "r",
    ) as f:
        lines = f.readlines()
    for line in lines:
        key = line.split()[0]
    value = " ".join(line.split()[1:])
    filename2transcript[key] = value
    splits = ["train", "dev", "test"]
    ID_start = 0  # needed to have a unique ID for each audio
    for split in splits:
        new_filename = os.path.join(save_folder, split) + ".csv"
        if os.path.exists(new_filename):
            continue
        logger.info("Preparing %s..." % new_filename)

        ID = []
        duration = []

        wav = []
        wav_format = []
        wav_opts = []

        # spk_id = []
        # spk_id_format = []
        # spk_id_opts = []

        transcript = []
        transcript_format = []
        transcript_opts = []

        all_wavs = glob.glob(
            os.path.join(data_folder, "data_nih/wav")
            + "/"
            + split
            + "/*/*.wav"
        )
        for i in range(len(all_wavs)):
            filename = all_wavs[i].split("/")[-1].split(".wav")[0]
            if filename not in filename2transcript:
                continue
            transcript_ = filename2transcript[filename]
            transcript.append(transcript_)
            transcript_format.append("string")
            transcript_opts.append(None)

            ID.append(ID_start + i)

            signal = read_audio(all_wavs[i])
            duration.append(signal.shape[0] / 16000)

            wav.append(all_wavs[i])
            wav_format.append("wav")
            wav_opts.append(None)

            # spk_id.append(df.speakerId[i])
            # spk_id_format.append("string")
            # spk_id_opts.append(None)

        new_df = pd.DataFrame(
            {
                "ID": ID,
                "duration": duration,
                "wav": wav,
                "transcript": transcript,
            }
        )
        new_df.to_csv(new_filename, index=False)
        ID_start += len(all_wavs)