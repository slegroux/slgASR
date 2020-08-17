from IPython import embed
import glob
from pathlib import Path
import uuid
import torchaudio
import pandas as pd
import spacy
import os
import swifter


class TextNormalizer(object):
    def __init__(self, lang:str='en', country:str='US'):
        # https://en.wikipedia.org/wiki/IETF_language_tag
        # ISO 639-1
        self._lang = lang
        if self._lang=='es':
            self._nlp = spacy.load("es_core_news_sm") 
        elif self._lang=='en':
            self._nlp = spacy.load("en_core_web_sm")
        self._country = country
    
    def normalize(self, text:str)->str:
        text = self.remove_punc(text)
        return(text)

    def remove_punc(self, sentence:str)-> str:
        doc = self._nlp(sentence)
        res = [(w.text, w.pos_) for w in doc]
        return(' '.join([w.lower() for w,att  in res if att!= 'PUNCT']))

class SpeechAsset():
    def __init__(self, path:str, lang:str='en', country:str='US', sid:str=None):
        self._lang = lang
        self._country = country
        self._path = Path(path).absolute()
        self._id = self._path.stem
        self._uuid = str(uuid.uuid4())
        if sid:
            self._sid = sid
        else:
            self._sid = self._id

    @property
    def lang(self)->str:
        return(self._lang)
    
    @lang.setter
    def lang(self, lang:str):
        self._lang = lang
    
    @property
    def country(self)->str:
        return(self._country)

    @country.setter
    def country(self, country:str):
        self._country = country

    @property
    def id(self)->str:
        return(self._id)
    
    @property
    def uuid(self)->str:
        return(self._uuid)

    @property
    def path(self)->str:
        return(str(self._path))


class Transcript(SpeechAsset):
    def __init__(self, path:str, lang:str='en', country:str='US', normalize:bool=True, sid:str=None, encoding='utf-8'):
        SpeechAsset.__init__(self,path=path, lang=lang, country=country, sid=sid)
        self._normalizer = TextNormalizer(self._lang)
        self._encoding = encoding
        with open(self._path, encoding=self._encoding) as f:
            if normalize:
                self._text = self._normalizer.normalize(f.readline().strip())
            else:
                self._text = f.readline().strip()

    def asdict(self):
        return {'text': self._text,
                'path': str(self._path),
                'lang': self._lang,
                'country': self._country,
                'id': self._id,
                'uuid': self._uuid,
                'sid': self._sid,
                'encoding': self._encoding,
                }

    @property
    def text(self)->str:
        return(self._text)
    
    @property
    def encoding(self)->str:
        return(self._encoding)
    

class Audio(SpeechAsset):
    def __init__(self, path:str, lang:str='en', country:str='US', sid:str=None):
        SpeechAsset.__init__(self,path=path, lang=lang, country=country, sid=sid)
        (self._data, self._sr) = Audio.get_data(path)
        self._duration = Audio.get_duration(path)

    def asdict(self):
        return {'data': self._data,
                'sr': self._sr,
                'duration': self._duration,
                'path': str(self._path),
                'lang': self._lang,
                'country': self._country,
                'id': self._id,
                'uuid': self._uuid,
                'sid': self._sid,
                }        
    
    @property
    def data(self):
        return(self._data)

    @property
    def sr(self):
        return(self._sr)
    
    @property
    def duration(self):
        return(self._duration)

    @staticmethod
    def get_data(filename:str):
        try:
            return(torchaudio.load(filename))
        except IOError as e:
            print("I/O error({0}): {1}".format(e.errno, e.strerror))
            return(None,None)            

    @staticmethod
    def get_duration(filename:str)->float:
        try:
            info = torchaudio.info(filename)
            duration = info[0].length / info[0].rate
            return(duration)
        except IOError as e:
            print("I/O error({0}): {1}".format(e.errno, e.strerror))
            return(None)


class Transcripts(object):
    def __init__(self, regex:str, normalize:bool=True, lang:str='en', country:str=None):
        self._paths = glob.glob(regex)
        # convert list of dicts into df
        self._transcripts = pd.DataFrame([Transcript(path, normalize=normalize, lang=lang, country=country).asdict() for path in self._paths])
    
    @property
    def transcripts(self):
        return(self._transcripts)


class Audios(object):
    def __init__(self, regex:str, lang:str='en', country:str='US'):
        self._paths = glob.glob(regex)
        # convert list of dicts into df
        self._audios = pd.DataFrame([Audio(path, lang=lang, country=country).asdict() for path in self._paths])
    
    @property
    def audios(self):
        return(self._audios)


class ASRDataset():
    def __init__(self, audios:pd.DataFrame, transcripts:pd.DataFrame, audio_cols=['id', 'sid', 'path', 'data', 'sr', 'lang', 'country'], transcripts_cols=['id','text','path'], join='id'):
        self._transcripts = transcripts
        self._audios = audios
        df = pd.merge(self._transcripts[transcripts_cols], self._audios[audio_cols], on='id')
        df['uuid'] = [str(uuid.uuid4()) for x in range(df.shape[0])]
        df.rename(columns={'path_x':'transcript_path', 'path_y':'audio_path'}, inplace=True)
        self._df = df
    
    def export2kaldi(self, dir_path:str, sr:int=16000):
        try:
            os.mkdir(dir_path)
        except OSError as error:
            print(error)
        
        # kaldi needs uuid that starts by sid for sorting
        # http://kaldi-asr.org/doc/data_prep.html
        # convert uuid type to string to be able to add sid to it

        self._df['uuid'] = self._df['sid'] + '_' + self._df['uuid']
        # hard copy otherwise it's just a view and then cannot reassign col values

        wav_scp = self._df[['uuid', 'audio_path']].copy()

        wav_scp.audio_path = 'sox ' + wav_scp.audio_path + ' -t wav -r ' + str(sr) + ' -c 1 -b 16 - |'

        wav_scp.to_csv(dir_path + '/wav.scp', sep=' ', index=False, header=None)
        utt2spk = self._df[['uuid','sid']]
        utt2spk.to_csv(dir_path + '/utt2spk', sep=' ', index=False, header=None)
        text = self._df[['uuid','text']]

        try:
            text.to_csv(dir_path + '/text', sep=' ', index=False, header=None)
        except IOError:
            print("File already exists. Delete or change path")

    @property
    def dataset(self):
        return(self._df)

class ASRDatasetCSV(ASRDataset):
    def __init__(self, path:str,
                map:dict={'sid':'client_id','country':'accent','audio_path':'path','text':'sentence'},
                sep:str='\t',
                lang:str='en',
                header:int=0,
                name:str='common_voice',
                prepend_audio_path:str='',
                normalize:bool=True):
    
        self._csv_path = path
        df = pd.read_csv(self._csv_path, sep=sep, header=header)
        names = {value : key for (key, value) in map.items()}

        df.rename(columns=names, inplace=True)
        df['uuid'] = [str(uuid.uuid4()) for x in range(df.shape[0])]
    
        if normalize:
            normalizer = TextNormalizer(lang)
            df['text'] = df['text'].swifter.apply(normalizer.normalize)

        self._df = df
