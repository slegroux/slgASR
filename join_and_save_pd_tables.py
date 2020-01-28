#!/usr/bin/env python

from IPython import embed
from data import WavFiles
from data_heroico import HeroicoTranscripts, HeroicoWavFile
from data_dimex import DimexTranscripts, DimexWavFile
from data import ASRDataset


class HeroicoDataset(ASRDataset):
    def __init__(self, wav_path, tr_path, name='heroico'):
        ASRDataset.__init__(self, wav_path, tr_path)
        DEFAULT_QUERY = "select {0}.uid, {0}.path as audio_path, {0}.sid, {0}.sr, {0}.duration, {0}.format, {0}.language, \
            {0}.dialect, {1}.path as transcript_path, {1}.transcript \
            from {0} join {1} on {0}.uid={1}.uid and {0}.sid={1}.sid"
        self._name = name
        self._query = DEFAULT_QUERY
    
    @property
    def tr(self):
        return(HeroicoTranscripts(self._tr_path).df)
    
    @property
    def wav(self):
        return(WavFiles(self._audio_path, HeroicoWavFile).df)
    
    @ASRDataset.query.setter
    def query(self, query):
        self._query = query


## HEROICO
## answers
tr_answers = "/home/workfit/Sylvain/Data/LDC/Heroico/LDC2006S37/data/transcripts/heroico-answers.txt"
wav_answers = "/home/workfit/Sylvain/Data/LDC/Heroico/LDC2006S37/data/speech/heroico/Answers_Spanish/*/*.wav"
q = "select {0}.uid, {0}.path as audio_path, {0}.sid, {0}.sr, {0}.duration, {0}.format, {0}.language, \
            {0}.dialect, {1}.path as transcript_path, {1}.transcript \
            from {0} join {1} on {0}.uid={1}.uid and {0}.sid={1}.sid"

answers = HeroicoDataset(wav_answers, tr_answers, 'answers')
answers.query = q
answers.pickle('heroico_answers.pkl')

embed()

## Recordings
tr_recordings = "/home/workfit/Sylvain/Data/LDC/Heroico/LDC2006S37/data/transcripts/heroico-recordings.txt"
wav_recordings = "/home/workfit/Sylvain/Data/LDC/Heroico/LDC2006S37/data/speech/heroico/Recordings_Spanish/*/*.wav"
q_rec = "select {0}.uid, {0}.path as audio_path, {0}.sid, {0}.sr, {0}.duration, {0}.format, {0}.language, \
            {0}.dialect, {1}.path as transcript_path, {1}.transcript \
        from {0} join {1} on {0}.uid={1}.uid"

recordings = HeroicoDataset(wav_recordings, tr_recordings, 'recordings')
recordings.query = q_rec
recordings.pickle('heroico_recordings.pkl')


# USMA
usma = '/home/workfit/Sylvain/Data/LDC/Heroico/LDC2006S37/data/transcripts/usma-prompts.txt'
native_regex = '/home/workfit/Sylvain/Data/LDC/Heroico/LDC2006S37/data/speech/usma/native*/*.wav'
nonnative_regex = '/home/workfit/Sylvain/Data/LDC/Heroico/LDC2006S37/data/speech/usma/nonnative*/*.wav'


# DIMEX
tr_reg_c = '/home/workfit/Sylvain/Data/Spanish/CorpusDimex100/*/*/comunes/*.txt.utf8'
wav_reg_c = '/home/workfit/Sylvain/Data/Spanish/CorpusDimex100/*/audio_16k/comunes/*.wav'
w_c, t_c  = get_wav_tr_df(tr_reg_c, wav_reg_c, DimexWavFile, DimexTranscripts)

tr_reg_i = '/home/workfit/Sylvain/Data/Spanish/CorpusDimex100/*/*/individuales/*.txt.utf8'
wav_reg_i = '/home/workfit/Sylvain/Data/Spanish/CorpusDimex100/*/audio_16k/individuales/*.wav'
w_i, t_i  = get_wav_tr_df(tr_reg_i, wav_reg_i, DimexWavFile, DimexTranscripts)

q = build_join_query("w_c", "t_c")
dimex_common = sqldf(q, locals())
add_uuid(dimex_common)
dimex_common.to_pickle('dimex_common.pkl')

q = build_join_query("w_i", "t_i")
dimex_individual = sqldf(q, locals())
add_uuid(dimex_individual)
dimex_individual.to_pickle('dimex_individual.pkl')




