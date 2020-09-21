#!/usr/bin/env python
# (c) 2020 slegroux@ccrma.stanford.edu

import click
from data import ASRDatasetCSV
from pathlib import Path
from IPython import embed

# @click.option("--dataset", default="/home/syl20/data/es/commonvoice", help="commonvoice data folder")
# @click.option("--output", default="/tmp/es/commonvoice", help="kaldi formatted data folder")

@click.command()
@click.argument("src", default="/home/syl20/data/es/commonvoice")
@click.argument("dst", default="/tmp/es/commonvoice")
@click.option("--lang", default="es")
def format_es_commonvoice(src, dst, lang):
    """Format commonvoice dataset into kaldi compatible data folder"""
    dataset_path = Path(src)
    formatted_dataset_path = Path(dst)
    audio_path = dataset_path / "clips"
    paths = {
        "train": dataset_path / "train.tsv",
<<<<<<< HEAD
<<<<<<< HEAD
        # "dev": dataset_path / "dev.tsv",
        # "test": dataset_path / "test.tsv"
=======
        "dev": dataset_path / "dev.tsv",
        "test": dataset_path / "test.tsv"
>>>>>>> formatting
=======
        "dev": dataset_path / "dev.tsv",
        "test": dataset_path / "test.tsv"
>>>>>>> 57ef79e47a9bc4da3e4d4e5fb644c0e6c8382b79
    }

    for k,v in paths.items():
        ds = ASRDatasetCSV(paths[k], lang=lang, prepend_audio_path=str(audio_path.absolute()))
        ds.export2kaldi(str(formatted_dataset_path / k))

if __name__ == "__main__":
    format_es_commonvoice()