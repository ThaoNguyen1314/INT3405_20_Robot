#===================================Text Preprocessing ============================
#Ignoring the warnings
import warnings
warnings.filterwarnings('ignore')

# Importing the required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re, string, unicodedata
import nltk
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.pooling import GlobalMaxPooling1D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
from keras.layers import *
from keras import backend
from sklearn.metrics import f1_score, confusion_matrix
import tensorflow as tf

import re
import string
import codecs

#Function for Text Preprocessing
# stop_words = set(stopwords.words("english")) 
# lemmatizer = WordNetLemmatizer()

def normalize_text(text):

    #Remove các ký tự kéo dài: vd: đẹppppppp
    text = re.sub(r'([A-Z])\1+', lambda m: m.group(1).upper(), text, flags=re.IGNORECASE)

    # Chuyển thành chữ thường
    text = text.lower()

    #Chuẩn hóa tiếng Việt, xử lý emoj, chuẩn hóa tiếng Anh, thuật ngữ
    replace_list = {
        'òa': 'oà', 'óa': 'oá', 'ỏa': 'oả', 'õa': 'oã', 'ọa': 'oạ', 'òe': 'oè', 'óe': 'oé','ỏe': 'oẻ',
        'õe': 'oẽ', 'ọe': 'oẹ', 'ùy': 'uỳ', 'úy': 'uý', 'ủy': 'uỷ', 'ũy': 'uỹ','ụy': 'uỵ', 'uả': 'ủa',
        'ả': 'ả', 'ố': 'ố', 'u´': 'ố','ỗ': 'ỗ', 'ồ': 'ồ', 'ổ': 'ổ', 'ấ': 'ấ', 'ẫ': 'ẫ', 'ẩ': 'ẩ',
        'ầ': 'ầ', 'ỏ': 'ỏ', 'ề': 'ề','ễ': 'ễ', 'ắ': 'ắ', 'ủ': 'ủ', 'ế': 'ế', 'ở': 'ở', 'ỉ': 'ỉ',
        'ẻ': 'ẻ', 'àk': u' à ','aˋ': 'à', 'iˋ': 'ì', 'ă´': 'ắ','ử': 'ử', 'e˜': 'ẽ', 'y˜': 'ỹ', 'a´': 'á',
        #Quy các icon về 2 loại emoj: Tích cực hoặc tiêu cực
        "👹": " tệ  ", '👍': '  ngon  ',
        "💎": " ngon ", "💩": " tệ  ","😕": " tệ  ", "😱": " tệ  ",
        "🚫": " tệ  ",  "🤬": " tệ  ","🧡": " ngon ",
        '👎': '  tệ   ', '😣': '  tệ   ','✨': '  ngon  ', '❣': '  ngon  ','☀': '  ngon  ',
        '♥': '  ngon  ', '🤩': '  ngon  ', 'like': '  ngon  ',
        '🖤': '  ngon  ', '🤤': '  ngon  ', ':(': '  tệ   ', '😢': '  tệ   ',
        '❤': '  ngon  ', '😍': '  ngon  ', '😘': '  ngon  ', '😪': '  tệ   ', '😊': '  ngon  ',
        '?': ' ? ', '😁': '  ngon  ', '💖': '  ngon  ', '😟': '  tệ   ', '😭': '  tệ   ',
        '💯': '  ngon  ', '💗': '  ngon  ', '♡': '  ngon  ', '💜': '  ngon  ', '🤗': '  ngon  ',
        '^^': '  ngon  ', '😨': '  tệ   ', '☺': '  ngon  ', '💋': '  ngon  ', '👌': '  ngon  ',
        '😖': '  tệ   ', ':((': '  tệ   ', '😡': '  tệ   ', '😠': '  tệ   ',
        '😒': '  tệ   ', '😏': '  tệ   ', '😝': '  ngon  ', '😄': '  ngon  ',
        '😙': '  ngon  ', '😤': '  tệ   ', '😎': '  ngon  ', '😆': '  ngon  ', '💚': '  ngon  ',
        '✌': '  ngon  ', '💕': '  ngon  ', '😞': '  tệ   ', '😓': '  tệ   ', '️🆗️': '  ngon  ',
        '😉': '  ngon  ', '😂': '  ngon  ', '😋': '  ngon  ',
        '💓': '  ngon  ', '😐': '  tệ   ', ':3': '  ngon  ', '😫': '  tệ   ', '😥': '  tệ   ',
        '😬': ' 😬 ', '😌': '  tệ  ', '💛': '  ngon  ', '🤝': '  ngon  ',
        '😗': '  ngon  ', '🤔': '  tệ   ', '😑': '  tệ   ', '🙏': '  tệ   ',
        '😻': '  ngon  ', '💙': '  ngon  ', '💟': '  ngon  ', '🍹':' ngon ', '😀':' ngon ','😃': ' ngon ',
        '😚': '  ngon  ', '❌': '  tệ   ', '👏': '  ngon  ', ';)': '  ngon  ', '<3': '  ngon  ',
        '🌷': '  ngon  ', '🌸': '  ngon  ', '🌺': '  ngon  ',
        '🌼': '  ngon  ', '🍓': '  ngon  ', '🐅': '  ngon  ', '🐾': '  ngon  ', '👉': '  ngon  ',
        '💐': '  ngon  ', '💞': '  ngon  ', '💥': '  ngon  ', '💪': '  ngon  ',
        '😇': '  ngon  ', '😛': '  ngon  ', '😜': '  ngon  ',
        '☹': '  tệ   ',  '💀': '  tệ   ',
        '😔': '  tệ   ', '😧': '  tệ   ', '😩': '  tệ   ', '😰': '  tệ   ',
        '😵': '  tệ   ', '😶': '  tệ   ', '🙁': '  tệ   ',
        #Chuẩn hóa 1 số sentiment words/English words
        ':))': '   ngon  ', ':)': '  ngon  ', 'ô kêi': ' ok ', 'okie': ' ok ', ' o kê ': ' ok ',
        'okey': ' ok ', 'ôkê': ' ok ', 'oki': ' ok ', ' oke ':  ' ok ',' okay':' ok ','okê':' ok ',
        ' tks ': u' cám ơn ', 'thks': u' cám ơn ', 'thanks': u' cám ơn ', 'ths': u' cám ơn ', 'thank': u' cám ơn ',
        '⭐': 'star ', '*': 'star ', '🌟': 'star ', '🎉': u'  ngon  ',
        'kg ': u' không ','not': u' không ', u' kg ': u' không ', '"k ': u' không ',' kh ':u' không ','kô':u' không ','hok':u' không ',' kp ': u' không phải ',u' kô ': u' không ', '"ko ': u' không ', u' ko ': u' không ', u' k ': u' không ', 'khong': u' không ', u' hok ': u' không ',
        'he he': '  ngon  ','hehe': '  ngon  ','hihi': '  ngon  ', 'haha': '  ngon  ', 'hjhj': '  ngon  ',
        ' lol ': '  tệ   ',' cc ': '  tệ   ','cute': u' dễ thương ','huhu': '  tệ   ', ' vs ': u' với ', 'wa': ' quá ', 'wá': u' quá', 'j': u' gì ', '“': ' ',
        ' sz ': u' cỡ ', 'size': u' cỡ ', u' đx ': u' được ', 'dk': u' được ', 'dc': u' được ', 'đk': u' được ',
        'đc': u' được ','authentic': u' chuẩn chính hãng ',u' aut ': u' chuẩn chính hãng ', u' auth ': u' chuẩn chính hãng ', 'thick': u'  ngon  ', 'store': u' cửa hàng ',
        'shop': u' cửa hàng ', 'sp': u' sản phẩm ', 'gud': u' tốt ','god': u' tốt ','wel done':' tốt ', 'good': u' tốt ', 'gút': u' tốt ',
        'sấu': u' xấu ','gut': u' tốt ', u' tot ': u' tốt ', u' nice ': u' tốt ', 'perfect': 'rất tốt', 'bt': u' bình thường ',
        'time': u' thời gian ', 'qá': u' quá ', u' ship ': u' giao hàng ', u' m ': u' mình ', u' mik ': u' mình ',
        'ể': 'ể', 'product': 'sản phẩm', 'quality': 'chất lượng','chat':' chất ', 'excelent': 'hoàn hảo', 'bad': ' tệ ','fresh': ' tươi ','sad': '  tệ  ',
        'date': u' hạn sử dụng ', 'hsd': u' hạn sử dụng ','quickly': u' nhanh ', 'quick': u' nhanh ','fast': u' nhanh ','delivery': u' giao hàng ',u' síp ': u' giao hàng ',
        'beautiful': u' đẹp tuyệt vời ', u' tl ': u' trả lời ', u' r ': u' rồi ', u' shopE ': u' cửa hàng ',u' order ': u' đặt hàng ',
        'chất lg': u' chất lượng ',u' sd ': u' sử dụng ',u' dt ': u' điện thoại ',u' nt ': u' nhắn tin ',u' tl ': u' trả lời ',u' sài ': u' xài ',u'bjo':u' bao giờ ',
        'thik': u' thích ',u' sop ': u' cửa hàng ', ' fb ': ' facebook ', ' face ': ' facebook ', ' very ': u' rất ',u'quả ng ':u' quảng  ',
        'dep': u' đẹp ',u' xau ': u' xấu ','delicious': u'  ngon  ', u'hàg': u' hàng ', u'qủa': u' quả ',
        'iu': u' yêu ','fake': u' giả mạo ', 'trl': 'trả lời', '><': u'  ngon  ',
        ' por ': u'  tệ  ',' poor ': u'  tệ  ', 'ib':u' nhắn tin ', 'rep':u' trả lời ',u'fback':' feedback ','fedback':' feedback ',
        # #dưới 3* quy về 1*, trên 3* quy về 5*
        # '6 sao': ' 5star ','6 star': ' 5star ', '5star': ' 5star ','5 sao': ' 5star ','5sao': ' 5star ',
        # 'starstarstarstarstar': ' 5star ', '1 sao': ' 1star ', '1sao': ' 1star ','2 sao':' 1star ','2sao':' 1star ',
        # '2 starstar':' 1star ','1star': ' 1star ', '0 sao': ' 1star ', '0star': ' 1star ',
        }

    for k, v in replace_list.items():
        text = text.replace(k, v)

    # chuyen punctuation thành space
    translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    text = text.translate(translator)

    texts = text.split()
    len_text = len(texts)

    #texts = [t.replace('_', ' ') for t in texts]
    # for i in range(len_text):
    #     cp_text = texts[i]
    #     if cp_text in not_list: # Xử lý vấn đề phủ định (VD: áo này chẳng đẹp--> áo này notpos)
    #         numb_word = 2 if len_text - i - 1 >= 4 else len_text - i - 1

    #         for j in range(numb_word):
    #             if texts[i + j + 1] in pos_list:
    #                 texts[i] = 'notpos'
    #                 texts[i + j + 1] = ''

    #             if texts[i + j + 1] in nag_list:
    #                 texts[i] = 'notnag'
    #                 texts[i + j + 1] = ''
    #     else: #Thêm feature cho những sentiment words (áo này đẹp--> áo này đẹp  ngon )
    #         if cp_text in pos_list:
    #             texts.append(' ngon ')
    #         elif cp_text in nag_list:
    #             texts.append(' tệ  ')

    #text = u' '.join(texts)

    #remove nốt những ký tự thừa thãi
    text = text.replace(u'"', u' ')
    text = text.replace(u'️', u'')
    text = text.replace('🏻','')
    return text

def add_data(data):
    #Từ điển tích cực, tiêu cực, phủ định
    path_nag = '/kaggle/working/Sentiment_Phobert/sentiment_dicts/nag.csv'
    path_pos = '/kaggle/working/Sentiment_Phobert/sentiment_dicts/pos.csv'

    df = pd.read_csv(path_nag)
    nag = pd.DataFrame({'input':df['Comment'],'label':df['Rating']})
    nag = nag.dropna(how='all')
    nag = nag.reset_index(drop=True)

    df = pd.read_csv(path_pos)
    pos = pd.DataFrame({'input':df['Comment'],'label':df['Rating']})
    pos = pos.dropna(how='all')
    pos = pos.reset_index(drop=True)
    data = data.append(pos, ignore_index = True)
    data = data.append(nag, ignore_index = True)
    return data

def clean_text_test(X):
    idx = 0
    processed = []
    for text in X:
        text = str(text)
        text = normalize_text(text)
        processed.append(text)
        idx += 1
    return processed
