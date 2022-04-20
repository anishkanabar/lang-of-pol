import os
import torch
import logging
import librosa
import pandas as pd
from pathlib import Path
import speechbrain as sb
from asr_dataset.police import BpcETL
from asr_dataset.librispeech import LibriSpeechETL
from asr_dataset.atczero import ATCZeroETL
from asr_dataset.constants import DataSizeUnit, Cluster

logger = logging.getLogger(__name__)

def prepare_bpc(cluster: str, 
                dataset_name:str, 
                num_train: int, 
                num_sec: float,
                split_ratios: dict,
                output_folder: str, 
                skip_prep=False):
    """
    @:param
        cluster: cluster name. either ['rcc', 'ai', 'ttic']
        dataset_name: either ['police', 'librispeech', 'atczero']
        num_train: number of samples to prepare from dataset
        num_sec: total seconds of audio to retrieve from dataset
        split_ratios: dictionary of named train/val/test splits and fraction of dataset
        output_folder: path to manifest (output) folder
        skip_prep: If True, skip data preparation.
    """
    if skip_prep:
        return

    cluster = Cluster[cluster.upper()]
    if dataset_name == 'police':
        etl = BpcETL(cluster)
    elif dataset_name == 'librispeech':
        etl = LibriSpeechETL(cluster)
    elif dataset_name == 'atczero':
        etl = ATCZeroETL(cluster)
    else:
        raise NotImplementedError('dataset ' + dataset_name)

    if num_sec is not None:
        qty = num_sec
        units = DataSizeUnit.SECONDS
    else:
        qty = num_train
        units = DataSizeUnit.ROW_COUNT
    data = etl.etl(qty=qty, units=units)

    # Make IDs global to splits
    data = data.reset_index()
    data = data.assign(ID = data.index.to_series())
    
    # Split into hparams-defined splits
    splits = {}
    other_splits = data
    other_frac = 1
    for split, frac in split_ratios.items():
        current_frac = max(0., min(1., frac / other_frac))
        current_split = other_splits.sample(frac=current_frac, random_state=1234)
        other_splits = other_splits.loc[other_splits.index.difference(current_split.index)]
        other_frac = max(0., min(1., other_frac - frac))
        splits[split] = current_split
        
    #test_data = data.sample(frac=.2, random_state=1234)
    #trainval_data = data.iloc[data.index.difference(test_data.index)]
    #val_data = trainval_data.sample(frac=.25, random_state=1234)
    #train_data = trainval_data.iloc[trainval_data.index.difference(val_data.index)]
    #splits = {'train': train_data, 'val': val_data, 'test': test_data}
    
    for split, splitdata in splits.items():
        manifest_path = os.path.join(output_folder, split) + '.csv'
        logger.info("Preparing %s..." % manifest_path)
        
        new_df = pd.DataFrame(
            {
                "ID": splitdata['ID'],
                "duration": splitdata['duration'],
                "wav": splitdata['audio'],
                "transcript": splitdata['text']
            }
        )
        new_df.to_csv(manifest_path, index=False)
        
def dataio_prepare(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions."""

    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_csv"]
    )

    if hparams["sorting"] == "ascending":
        # we sort training data to speed up training and get better results.
        train_data = train_data.filtered_sorted(sort_key="duration")
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "descending":
        train_data = train_data.filtered_sorted(
            sort_key="duration", reverse=True
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "random":
        pass

    else:
        raise NotImplementedError(
            "sorting must be random, ascending or descending"
        )

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["valid_csv"]
    )
    valid_data = valid_data.filtered_sorted(sort_key="duration")

    # test is separate
    test_datasets = {}
    for csv_file in hparams["test_csv"]:
        name = Path(csv_file).stem
        test_datasets[name] = sb.dataio.dataset.DynamicItemDataset.from_csv(
            csv_path=csv_file
        )
        test_datasets[name] = test_datasets[name].filtered_sorted(
            sort_key="duration"
        )

    datasets = [train_data, valid_data] + [i for k, i in test_datasets.items()]

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        sig = sb.dataio.dataio.read_audio(wav)
        sig_np = sig.numpy()
        sig_res = librosa.resample(sig_np, hparams['data_sample_rate'], hparams['model_sample_rate'])
        sig = torch.tensor(sig_res)
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)
    label_encoder = sb.dataio.encoder.CTCTextEncoder()

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("wrd")
    @sb.utils.data_pipeline.provides(
        "wrd", "char_list", "tokens_list", "tokens_bos", "tokens_eos", "tokens"
    )
    def text_pipeline(wrd):
        yield wrd
        char_list = list(wrd)
        yield char_list
        tokens_list = label_encoder.encode_sequence(char_list)
        yield tokens_list
        tokens_bos = torch.LongTensor([hparams["bos_index"]] + (tokens_list))
        yield tokens_bos
        tokens_eos = torch.LongTensor(tokens_list + [hparams["eos_index"]])
        yield tokens_eos
        tokens = torch.LongTensor(tokens_list)
        yield tokens

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    lab_enc_file = os.path.join(hparams["save_folder"], "label_encoder.txt")
    special_labels = {
        "bos_label": hparams["bos_index"],
        "eos_label": hparams["eos_index"],
        "blank_label": hparams["blank_index"],
    }
    label_encoder.load_or_create(
        path=lab_enc_file,
        from_didatasets=[train_data],
        output_key="char_list",
        special_labels=special_labels,
        sequence_input=True,
    )
    label_encoder.add_unk()

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets,
        ["id", "sig", "wrd", "char_list", "tokens_bos", "tokens_eos", "tokens"],
    )
    return train_data, valid_data, test_datasets, label_encoder

 
        
