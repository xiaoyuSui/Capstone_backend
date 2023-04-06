from flask import Flask,jsonify, request
from Capstone import SpeechTranslation
import librosa
import soundfile as sf
import io
from urllib.request import urlopen
import json
from flask_cors import CORS
from pydub import AudioSegment
from os import path


app = Flask(__name__)
CORS(app)
#cors = CORS(app, resources={r"/api/*": {"origins": "*"}})


@app.route('/api/translate/url=<path:url>', methods=['GET'])
def get_translation_by_url(url):
    translator = SpeechTranslation()
    audio, sr = sf.read(io.BytesIO(urlopen(url).read()))
    audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    wav, rate, article_en, zh_trans = translator.translate(audio)
    #ipd.Audio(wav, rate=rate)
    data = {'article_en': article_en,
            'zh_trans': zh_trans}
    jstr=json.dumps(data,ensure_ascii=False)
    return jstr


@app.route('/api/translate', methods=['POST'])
def get_translation_by_file():
    fileStorage = request.files['file_from_react']
    bytes=fileStorage.read()
    sound = AudioSegment.from_file(io.BytesIO(bytes), format="mp3")
    print(type(sound))
    sound.export("test.wav", format="wav")

    audio, sr = librosa.load('./test.wav')
    audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    translator = SpeechTranslation()
    wav, rate, article_en, zh_trans = translator.translate(audio)
    print(zh_trans)
    #ipd.Audio(wav, rate=rate)
    data = {'article_en': article_en,
            'zh_trans': zh_trans,
            'status':1
        }

    '''
        with open(blobfile, 'r') as f: 

        audio, sr = sf.read(f.read())
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        translator = SpeechTranslation()
        wav, rate, article_en, zh_trans = translator.translate(audio)
        #ipd.Audio(wav, rate=rate)
        data = {'article_en': article_en,
                'zh_translate': zh_trans,
                'status':1
            }
      
        file = request.files['file_from_react']
        filename = file.filename
        print(f"Uploading file {filename}")
        file_bytes = file.read()
        file_content = io.BytesIO(file_bytes).readlines()

        translator = SpeechTranslation()
        audio, sr = sf.read(file_content)
        #audio, sr = librosa.load(response)
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        wav, rate, article_en, zh_trans = translator.translate(audio)
        #ipd.Audio(wav, rate=rate)
        data = {'article_en': article_en,
                'zh_translate': zh_trans,
                'status':1
            }
    '''
    jstr=json.dumps(data,ensure_ascii=False)
    return jstr

@app.route('/api/translatemc', methods=['POST'])
def get_translationmc_by_file():
    fileStorage = request.files['file_from_react']
    bytes=fileStorage.read()
    sound = AudioSegment.from_file(io.BytesIO(bytes), format="mp3")
    print(type(sound))
    sound.export("test.wav", format="wav")

    audio, sr = librosa.load('./test.wav')
    audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    translator = SpeechTranslation()
    #wav, rate, article_en, zh_trans, dict = translator.translate_mc(audio)s
    article_en, zh_trans, dict = translator.translate_mc(audio)
    print(zh_trans)
    #ipd.Audio(wav, rate=rate)
    data = {'article_en': article_en,
            'zh_trans': zh_trans,
            'dict': dict,
            'status':1
        }

    print(dict)
    jstr=json.dumps(data,ensure_ascii=False)
    return jstr
        

    


@app.route('/')
def test():
    return 'This is my first API call!'

#python -m flask --app main run
#ssh -L 8282:localhost:5000 -p 39122 suixiaoyu@129.128.209.207