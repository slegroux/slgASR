[![forthebadge](https://forthebadge.com/images/badges/made-with-python.svg)](https://forthebadge.com)
[![forthebadge](https://forthebadge.com/images/badges/built-with-love.svg)](https://forthebadge.com)

[![made-with-bash](https://img.shields.io/badge/Made%20with-Bash-1f425f.svg)](https://www.gnu.org/software/bash/)
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![made-with-VSCode](https://img.shields.io/badge/Made%20with-VSCode-1f425f.svg)](https://code.visualstudio.com/)
[![Generic badge](https://img.shields.io/badge/Made%20for-Kaldi-1f425f.svg)](https://shields.io/)
[![Build Status](https://travis-ci.com/slegroux/slgASR.svg?branch=master)](https://travis-ci.com/slegroux/slgASR)

# slgASR
repository of scripts useful for processing, training and testing speech recognition engines

## installation
- depends on: pandas, pandasql, pytorch, torchaudio and spacy
- dependencies can be installed using [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/)
``` bash
cd slgASR
conda env create -f environment.yml
```
- pre-trained models from Spacy should be downloaded as such:
    - python -m spacy download es_core_news_sm
    - python -m spacy download en_core_web_sm

## Supported Datasets
### English
- commonvoice
### Spanish
- heroico
- dimex
- commonvoice

## Usage
```python
# commonvoice reference data path
common_voice_data = {
    'path': 'data/tests/common_voice/test.tsv'
}

# column names in data
ids = ['sid', 'audio_path', 'transcript', 'up_votes', 'down_votes', 'age', 'gender', 'dialect']

# get pandas style dataset with generic interface
ds = ASRDataset.init_with_csv(common_voice_data['path'], ids, name='common_voice')

# get dataset with specialized commonvoice dataset
cv = CommonVoiceDF(common_voice_data['path'])

# export to kaldi format
ds.export2kaldi('/tmp/kaldi_dir')
```

## License
[GPL](https://www.gnu.org/licenses/gpl-3.0-standalone.html)

## Authors
(c) 2020 Sylvain Le Groux <slegroux@ccrma.stanford.edu>



