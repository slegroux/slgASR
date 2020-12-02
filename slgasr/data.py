# (c) 2020 Sylvain Le Groux <slegroux@ccrma.stanford.edu>

from IPython import embed
import glob
from pathlib import Path
import uuid
import torchaudio
import pandas as pd
import spacy
import os
import swifter
import logging
import re
import enchant

logging.basicConfig(level=logging.DEBUG)

class TextNormalizer(object):
    def __init__(self, lang:str='en', country:str='US'):
        # https://en.wikipedia.org/wiki/IETF_language_tag
        # ISO 639-1
        self._lang = lang
        if self._lang=='es':
            self._nlp = spacy.load("es_core_news_sm") 
            self._dictionary = enchant.Dict("es_ES")
        elif self._lang=='en':
            self._nlp = spacy.load("en_core_web_sm")
            self._dictionary = enchant.Dict("en_US")
        elif self._lang=='it':
            self._nlp = spacy.load("it_core_news_sm")
        elif self._lang=='pt':
            self._nlp = spacy.load("pt_core_news_sm")
        elif self._lang=='fr':
            self._nlp = spacy.load("fr_core_news_sm")
            self._dictionary = enchant.Dict("fr_FR")
        else:
            raise Exception("language {} is not supported yet".format(self._lang))
        self._country = country
    
    def normalize(self, text:str)->str:
        text = self.remove_brackets(text)
        text = self.remove_newline(text)
        text = self.remove_punc(text)
        text = self.remove_extra_spaces(text)
        text = text.lstrip()
        text = self.reformat_abbv(text)
        text = self.remove_foreign_language(text,length=10)
        return(text)

    def remove_brackets(self, sentence:str)->str:
        pattern = r'\[.*?\]'
        return(re.sub(pattern,'', sentence))
    
    def remove_extra_spaces(self, sentence:str)->str:
        pattern = r'\s+'
        return(re.sub(pattern,' ', sentence))
    
    def remove_newline(self, sentence:str)->str:
        pattern = r'\\n'
        return(re.sub(pattern, '', sentence))
    
    def reformat_abbv(self, sentence:str)->str:
        pattern = r'\. _'
        return(re.sub(pattern, '._', sentence))

    def remove_punc(self, sentence:str)-> str:
        doc = self._nlp(sentence)
        res = [(w.text, w.pos_) for w in doc]
        return(' '.join([w.lower() for w,att  in res if att!= 'PUNCT']))
    
    def remove_foreign_language(self, sentence:str, length:int=10)-> str:
        buffer = []
        for word in sentence.split():
            if not self._dictionary.check(word):
                buffer.append(word)
                if len(buffer) > length:
                    return("")
        return(sentence)

    def remove_en_language(self, sentence:str)-> str:
        d = enchant.Dict("en_US")
        for word in sentence.split():
            if d.check(word):
                return("")
        return(sentence)
    
    @staticmethod
    def remove_double_quote_from_file(filename:str):
        try:
            with open(filename, 'r+') as f:
                t = f.read().replace('\"', '')
                f.seek(0)
                f.write(t)
                f.truncate()
        except IOError as e:
            logging.error(e, exc_info=True)

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
            logging.exception(str(e))
            return(None,None)            

    @staticmethod
    def get_duration(filename:str)->float:
        try:
            info = torchaudio.info(filename)
            duration = info[0].length / info[0].rate
            return(duration)
        except IOError as e:
            logging.exception(str(e))
            return(None)


class Transcripts(object):
    def __init__(self, regex:str, normalize:bool=True, lang:str='en', country:str=None):
        self._paths = glob.glob(regex)
        # convert list of dicts into df
        self._transcripts = pd.DataFrame([Transcript(path, normalize=normalize, lang=lang, country=country).asdict() for path in self._paths])
    
    @property
    def transcripts(self):
        return(self._transcripts)


class TranscriptsCSV(Transcripts):
    def __init__(self, regex:str, normalize:bool=True, lang:str='en', country:str=None):
        # Transcripts.__init__(self, regex=regex, normalize=normalize, lang=lang, country=country)
        self._paths = glob.glob(regex)
        self._transcripts = pd.DataFrame()

        for path in self._paths:
            tmp = pd.read_csv(path, header=None)
            id = tmp.iloc[:,0].apply(lambda x: x.split()[0])
            text = tmp.iloc[:,0].apply(lambda x: ' '.join(x.split()[1:]))
            p = [path]*len(tmp)
            tmp = pd.DataFrame({'id':id, 'text':text, 'path': p})

            self._transcripts = self._transcripts.append(tmp, ignore_index=True)

        if normalize:
            normalizer = TextNormalizer(lang)
            self._transcripts['text'] = self._transcripts['text'].swifter.apply(normalizer.normalize)


class Audios(object):
    def __init__(self, regex:str, lang:str='en', country:str='US', sid_from_path=None):
        self._paths = glob.glob(regex)
        # convert list of dicts into df
        if sid_from_path:
            self._audios = pd.DataFrame([Audio(path, lang=lang, country=country, sid=sid_from_path(path)).asdict() for path in self._paths])
        else:
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
    
    def export2kaldi(self, dir_path:str, sr:int=16000, ext:str='wav'):
        path = Path(dir_path)
        try:
            path.mkdir(parents=True, exist_ok=False)
        except FileExistsError as e:
            logging.exception(str(e))
        else:
            logging.info("Folder created")
        
        # kaldi needs uuid that starts by sid for sorting
        # http://kaldi-asr.org/doc/data_prep.html
        # convert uuid type to string to be able to add sid to it
        self._df['uuid'] = self._df['sid'] + '-' + self._df['uuid']
        # hard copy otherwise it's just a view and then cannot reassign col values

        wav_scp = self._df[['uuid', 'audio_path']].copy()
        if ext == 'wav':
            wav_scp['audio_path'] = 'sox ' + wav_scp.audio_path + ' -t wav -r ' + str(sr) + ' -c 1 -b 16 - |'
        elif ext == 'flac':
            # TODO(slg): check sr param for flac
            wav_scp['audio_path'] = 'flac -c -d -s ' + wav_scp.audio_path + ' |'
        try:
            wav_scp.to_csv(os.path.join(dir_path,'wav.scp'), sep=' ', index=False, header=None)
            TextNormalizer.remove_double_quote_from_file(os.path.join(dir_path,'wav.scp'))
        except IOError as e:
            logging.exception(str(e))
        utt2spk = self._df[['uuid','sid']]
        try:
            utt2spk.to_csv(os.path.join(dir_path, 'utt2spk'), sep=' ', index=False, header=None)
        except IOError as e:
            logging.exception(str(e))
        text = self._df[['uuid','text']]
        try:
            text.to_csv(os.path.join(dir_path, 'text'), sep=' ', index=False, header=None)
            TextNormalizer.remove_double_quote_from_file(os.path.join(dir_path,'text'))
        except IOError as e:
            logging.exception(str(e))

    @property
    def dataset(self):
        return(self._df)

class ASRDatasetCSV(ASRDataset):
    def __init__(self, path:str,
                map:dict={'sid':'client_id','country':'accent','audio_path':'path','text':'sentence'},
                sep:str='\t',
                lang:str='en',
                header:int=0,
                skipinitialspace:bool=True,
                name:str='common_voice',
                prepend_audio_path:str='',
                normalize:bool=True):
    
        self._csv_path = path

        df = pd.read_csv(self._csv_path, sep=sep, header=header, skipinitialspace=skipinitialspace, error_bad_lines=False)    
        names = {value : key for (key, value) in map.items()}
        df.rename(columns=names, inplace=True)

        df['uuid'] = [str(uuid.uuid4()) for x in range(df.shape[0])]
        df['audio_path'] = df['audio_path'].swifter.apply(lambda x: prepend_audio_path + '/' + x)

        if 'sid' not in df.columns:
            df['sid'] = df.audio_path.apply(lambda x: Path(x).parent.name)

        if normalize:
            normalizer = TextNormalizer(lang)
            df['text'] = df['text'].swifter.apply(normalizer.normalize)
        
        df = df.drop_duplicates(subset=['text','sid'])
        df.replace("",float("NaN"), inplace=True)
        df.dropna(subset=['text'],inplace=True)
        self._df = df
    
    @property
    def df(self)->pd.DataFrame:
        return(self._df)
