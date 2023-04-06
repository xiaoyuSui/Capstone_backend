#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# In[2]:


from dropout_mc import *
from nltk.translate.bleu_score import sentence_bleu
from fastdtw import fastdtw


def get_blue_dist_func(weights=(1 / 3, 1 / 3, 1 / 3), lang='en'):
    weights = {1:(1, ), 2:(1/2, 1/2), 3:(1/3, 1/3, 1/3)}
    def blue_dist_func(a, b):
        if isinstance(a, str):
            if lang == 'en':
                a = a.split(' ')
                b = b.split(' ')
            elif lang == 'zh':
                a = [v for v in a]
                b = [v for v in b]
        if len(a) == 0 and len(b) == 0:
            return 0
        elif len(a) == 0 or len(b) == 0:
            return 1
        if isinstance(a, list) and (not isinstance(a[0], list)):
            a = [a]
        if len(a[0]) < 4:
            key = 1
        elif len(a[0]) < 6:
            key = 2
        else:
            key = 3
        #key = min(len(a[0]), 3)
        return 1 - sentence_bleu(a, b, weights=weights[key])
    return blue_dist_func

def dtw(a1, a2):
    distance, path = fastdtw(a1.cpu().numpy(), a2.cpu().numpy())
    return distance / max(len(a1), len(a2))


# In[3]:


import inspect
import torch.nn.functional as F

class Dropout_manager:
    """
    This class implement forward hook with dropout function
    to insert dropout computation inside a pytorch neural network.
    """
    def __init__(self, model, dropout_p=0, before_layer_type="activation"):
        #self.recorder = dict()
        self.layer_types = list()
        self.dropout_p = dropout_p
        if before_layer_type == "activation":
            # https://stats.stackexchange.com/questions/240305/where-should-i-place-dropout-layers-in-a-neural-network
            target_layers = inspect.getmembers(torch.nn.modules.activation, inspect.isclass)
            filter_out_layer = set(["Module", "MultiheadAttention", 
                                    "NonDynamicallyQuantizableLinear",
                                    "Parameter, Tensor"])
            for layer in target_layers:
                if layer[0] in filter_out_layer:
                    continue
                else:
                    self.layer_types.append(layer[1])
        else:
            raise NotImplementedError
        if isinstance(model, torch.nn.DataParallel):
            self.model = model.module
        else:
            self.model = model
        self.handlers = list()
    
    def _register_hooker(self):
        #self.recorder[name] = list()
        def named_hooker(module, input, output):
            assert len(input) == 1
            return F.dropout(input[0], p=self.dropout_p, training=True)
        return named_hooker
    
    def _whether_to_insert_dropout(self, layer):
        for layer_type in self.layer_types:
            if isinstance(layer, layer_type):
                return True
        return False
        
    def register_hookers(self):
        modules_to_insert = []
        for module in self.model.modules():
            insert_flag = self._whether_to_insert_dropout(module)
#             if insert_flag == True:
#                 handler = module.register_forward_hook(self._register_hooker())
#                 self.handlers.append(handler)
            if module.__class__.__name__ == 'MBartEncoderLayer' or module.__class__.__name__ == 'MBartDecoderLayer':
                module.train()
                module.activation_dropout = self.dropout_p
                module.dropout = self.dropout_p
        
    def remove_handlers(self):
        for i in self.handlers:
            i.remove()
        self.handlers.clear()
        
    def __del__(self):
        self.remove_handlers()


# In[4]:


from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from datasets import load_dataset
import torch
import torchaudio
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from fairseq.checkpoint_utils import load_model_ensemble_and_task_from_hf_hub
from fairseq.models.text_to_speech.hub_interface import TTSHubInterface
import IPython.display as ipd

class EnglishASR(object):
    def __init__(self, pretrained="facebook/wav2vec2-large-960h-lv60-self"):
        self.processor = Wav2Vec2Processor.from_pretrained(pretrained)
        self.model = Wav2Vec2ForCTC.from_pretrained(pretrained)
        self.model.eval()
    
    def predict(self, audio, sr=16000):
        input_values = self.processor(audio, sampling_rate=sr, return_tensors="pt", padding="longest").input_values
        logits = self.model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        en_transcription = self.processor.batch_decode(predicted_ids)
        return en_transcription
    
    def predict_mc(self, audio, sr=16000, mc_times=10, p=0.01):
        ori_prediction = self.predict(audio, sr)
        # Dropout Run
        bleu_dist = get_blue_dist_func(lang='en')
        activate_mc_dropout_seq(self.model, True, p)
        mc_prediction = []
        vro = 0
        vr = 0
        for _ in range(mc_times):
            mc_prediction += self.predict(audio)
#             print(str(len(mc_prediction)) + ': ' + mc_prediction[-1])
        for i in range(mc_times):
            vro += (1 - bleu_dist(mc_prediction[i], ori_prediction[0]))
            tmp = 0
            for j in range(mc_times):
                if j!= i:
                    tmp += (1 - bleu_dist(mc_prediction[i], mc_prediction[j]))
            vr += (tmp / (mc_times - 1))
        self.model.eval()
        vro = 1 - (vro / mc_times)
        vr = 1 - (vr / mc_times)
        return ori_prediction, vro, vr
        
    
class EnglishChineseTranslation(object):
    def __init__(self, pretrained="facebook/mbart-large-50-one-to-many-mmt"):
        self.model = MBartForConditionalGeneration.from_pretrained(pretrained)
        self.model.eval()
        self.tokenizer = MBart50TokenizerFast.from_pretrained(pretrained, src_lang="en_XX")
    
    def predict(self, article_en):
        model_inputs = self.tokenizer(article_en, return_tensors="pt")
        generated_tokens = self.model.generate(
            **model_inputs,
            forced_bos_token_id=self.tokenizer.lang_code_to_id["zh_CN"]
        )
        zh_trans = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        return zh_trans[0]
    
    def predict_mc(self, article_en, mc_times=10, p=0.01):
        ori_prediction = self.predict(article_en)
        bleu_dist = get_blue_dist_func(lang='zh')
        dropout_manager = Dropout_manager(self.model, p)
        dropout_manager.register_hookers()
        activate_mc_dropout_seq(self.model, True, p)
        mc_prediction = []
        vro = 0
        vr = 0
        for _ in range(mc_times):
            mc_prediction += [self.predict(article_en)]
#             print(str(len(mc_prediction)) + ': ' + mc_prediction[-1])
        for i in range(mc_times):
            vro += (1 - bleu_dist(mc_prediction[i], ori_prediction))
            tmp = 0
            for j in range(mc_times):
                if j!= i:
                    tmp += (1 - bleu_dist(mc_prediction[i], mc_prediction[j]))
            vr += (tmp / (mc_times - 1))
        self.model.eval()
        dropout_manager.remove_handlers()
        vro = 1 - (vro / mc_times)
        vr = 1 - (vr / mc_times)
        return ori_prediction, vro, vr
        

class ChineseTTS(object):
    def __init__(self, pretrained="facebook/tts_transformer-zh-cv7_css10"):
        self.models, self.cfg, self.task = load_model_ensemble_and_task_from_hf_hub(
            pretrained,
            arg_overrides={"vocoder": "hifigan", "fp16": False}
        )
        self.model = self.models[0].cuda()
        self.model.eval()
        TTSHubInterface.update_cfg_with_data_cfg(self.cfg, self.task.data_cfg)
        self.generator = self.task.build_generator([self.model], self.cfg)
        
    def predict(self, article_zh):
        sample = TTSHubInterface.get_model_input(self.task, article_zh)
        sample["net_input"]["src_tokens"] = sample["net_input"]["src_tokens"].cuda()
        sample["net_input"]["src_lengths"] = sample["net_input"]["src_lengths"].cuda()
        sample["speaker"] = sample["speaker"].cuda()

        wav, rate = TTSHubInterface.get_prediction(self.task, self.model, self.generator, sample)
        return wav, rate
    
    def predict_mc(self, article_zh, mc_times=10, p=0.01):
        ori_prediction, rate = self.predict(article_zh)
        activate_mc_dropout_seq(self.model, True, p)
        mc_prediction = []
        vro = 0
        vr = 0
        for _ in range(mc_times):
            mc_prediction += [self.predict(article_zh)[0]]
        for i in range(mc_times):
            vro += (1 - dtw(mc_prediction[i], ori_prediction))
            tmp = 0
            for j in range(mc_times):
                if j!= i:
                    tmp += (1 - dtw(mc_prediction[i], mc_prediction[j]))
            vr += (tmp / (mc_times - 1))
        self.model.eval()
        vro = 1 - (vro / mc_times)
        vr = 1 - (vr / mc_times)
        return ori_prediction, rate, vro, vr


class SpeechTranslation(object):
    def __init__(self, mode='en_zh', asr="facebook/wav2vec2-large-960h-lv60-self", 
                 translation="facebook/mbart-large-50-one-to-many-mmt", 
                 tts="facebook/tts_transformer-zh-cv7_css10"):
        self.mode = mode
        self.source_asr, self.translation, self.target_tts = None, None, None
        if self.mode == 'en_zh':
            self.source_asr = EnglishASR(asr)
            self.translation = EnglishChineseTranslation(translation)
            self.target_tts = ChineseTTS(tts)
    
    def translate(self, audio, sr=16000):
        article_en = self.source_asr.predict(audio, sr)
        print('===English Transcription===')
        print(article_en)
        print()
        zh_trans = self.translation.predict(article_en)
        print('===Chinese Transcription===')
        print(zh_trans)
        print()
        (wav, rate) = self.target_tts.predict(zh_trans)
        return wav.cpu(), rate, article_en, zh_trans
    
    def translate_mc(self, audio, sr=16000, mc_times=10, p=0.01):
        dict = {}
        article_en, vro, vr = self.source_asr.predict_mc(audio, sr, mc_times, p)
        dict['asr_vro'] = vro
        dict['asr_vr'] = vr
        print('===English Transcription===')
        print(article_en)
        print('VRO: %.4f, VR: %.4f' % (vro, vr))
        print()

        zh_trans, vro, vr = self.translation.predict_mc(article_en, mc_times, p)
        dict['trans_vro'] = vro
        dict['trans_vr'] = vr
        print('===Chinese Transcription===')
        print(zh_trans)
        print('VRO: %.4f, VR: %.4f' % (vro, vr))
        print()
        '''

        print('===Chinese Speech Generation===')
        wav, rate, vro, vr = self.target_tts.predict_mc(zh_trans, mc_times, p)
        print('VRO: %.4f, VR: %.4f' % (vro, vr))
        dict['tts_vro'] = vro
        dict['tts_vr'] = vr
        '''

        #return wav.cpu(), rate, article_en, zh_trans, dict
        return article_en, zh_trans, dict


# In[5]:
'''

import pandas as pd
#commonvoice = pd.read_csv('./data/CommonVoiceEN/v1.csv')
#commonvoice[:10]
# In[6]:
import librosa
# In[7]:
translator = SpeechTranslation()
# In[8]:

audio, sr = librosa.load('./84-121550-0000.flac')
audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
wav, rate = translator.translate(audio)
ipd.Audio(wav, rate=rate)

import soundfile as sf
import io

from urllib.request import Request, urlopen
url = 'https://suixiaoyu-capstone-data.s3.us-west-2.amazonaws.com/121123/84-121123-0000.flac'
#req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
audio, sr = sf.read(io.BytesIO(urlopen(url).read()))
#audio, sr = librosa.load(response)
audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
wav, rate = translator.translate(audio)
ipd.Audio(wav, rate=rate)
'''
# In[9]:

'''
wav, rate = translator.translate_mc(audio, p=0.01)
ipd.Audio(wav, rate=rate)


# In[10]:


audio, sr = librosa.load('./data' + commonvoice['path'][15][1:])
audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
wav, rate = translator.translate(audio)
ipd.Audio(wav, rate=rate)


# In[11]:


wav, rate = translator.translate_mc(audio, p=0.01)
ipd.Audio(wav, rate=rate)


# In[ ]:





# In[12]:


translator.translation.model


# In[13]:


translator.source_asr.model


# In[14]:


translator.target_tts.model


# In[ ]:



'''
