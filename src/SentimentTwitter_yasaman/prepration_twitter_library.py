import sys
import subprocess
import re
import pandas as pd
from hazm import Normalizer, WordTokenizer, Lemmatizer
from datetime import datetime

try:
    from persiantools.jdatetime import JalaliDate
except ImportError:
    print("Installing required package: persiantools")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "persiantools"])
    from persiantools.jdatetime import JalaliDate


def extract_links(text):
    return ' '.join(re.findall(r'http[s]?://\S+', str(text))) if pd.notna(text) else ""

def extract_hashtags(text):
    return ' '.join(re.findall(r'#\S+', str(text))) if pd.notna(text) else ""

def extract_mentions(text):
    return ' '.join(re.findall(r'@\S+', str(text))) if pd.notna(text) else ""


class Preprocess:
    def __init__(self):
        self.normalizer = Normalizer()
        self.tokenizer = WordTokenizer()
        self.lemmatizer = Lemmatizer()

    def handle_correct_alphabets(self, doc_string):
        multiple_subs = [
            (r"ٲ|ٱ|إ|ﺍ|أ", r"ا"),
            (r"ﺁ|آ", r"ا"),
            (r"ﺐ|ﺏ|ﺑ", r"ب"),
            (r"ﭖ|ﭗ|ﭙ|ﺒ|ﭘ", r"پ"),
            (r"ﭡ|ٺ|ٹ|ﭞ|ٿ|ټ|ﺕ|ﺗ|ﺖ|ﺘ", r"ت"),
            (r"ﺙ|ﺛ", r"ث"),
            (r"ﺝ|ڃ|ﺠ|ﺟ", r"ج"),
            (r"ڃ|ﭽ|ﭼ", r"چ"),
            (r"ﺢ|ﺤ|څ|ځ|ﺣ", r"ح"),
            (r"ﺥ|ﺦ|ﺨ|ﺧ", r"خ"),
            (r"ڏ|ډ|ﺪ|ﺩ", r"د"),
            (r"ڙ|ڗ|ڒ|ڑ|ڕ|ﺭ|ﺮ", r"ر"),
            (r"ﺮ|ﺯ", r"ز"),
            (r"ﮊ", r"ژ"),
            (r"ݭ|ݜ|ﺱ|ﺲ|ښ|ﺴ|ﺳ", r"س"),
            (r"ﺵ|ﺶ|ﺸ|ﺷ", r"ش"),
            (r"ﺺ|ﺼ|ﺻ", r"ص"),
            (r"ﺽ|ﺾ|ﺿ|ﻀ", r"ض"),
            (r"ﻁ|ﻂ|ﻃ|ﻄ", r"ط"),
            (r"ﻆ|ﻇ|ﻈ", r"ظ"),
            (r"ڠ|ﻉ|ﻊ|ﻋ", r"ع"),
            (r"ﻎ|ۼ|ﻍ|ﻐ|ﻏ", r"غ"),
            (r"ﻒ|ﻑ|ﻔ|ﻓ", r"ف"),
            (r"ﻕ|ڤ|ﻖ|ﻗ", r"ق"),
            (r"ڭ|ﻚ|ﮎ|ﻜ|ﮏ|ګ|ﻛ|ﮑ|ﮐ|ڪ|ك", r"ک"),
            (r"ﮚ|ﮒ|ﮓ|ﮕ|ﮔ", r"گ"),
            (r"ﻝ|ﻞ|ﻠ|ڵ", r"ل"),
            (r"ﻡ|ﻤ|ﻢ|ﻣ", r"م"),
            (r"ڼ|ﻦ|ﻥ|ﻨ", r"ن"),
            (r"ވ|ﯙ|ۈ|ۋ|ﺆ|ۊ|ۇ|ۏ|ۅ|ۉ|ﻭ|ﻮ|ؤ", r"و"),
            (r"ﺔ|ﻬ|ھ|ﻩ|ﻫ|ﻪ|ۀ|ە|ة|ہ", r"ه"),
            (r"ﭛ|ﻯ|ۍ|ﻰ|ﻱ|ﻲ|ں|ﻳ|ﻴ|ﯼ|ې|ﯽ|ﯾ|ﯿ|ێ|ے|ى|ي", r"ی"),
        ]
        for old, new in multiple_subs:
            doc_string = re.sub(old, new, doc_string)
        return doc_string

    def clean(self, text):
        if not isinstance(text, str):
            text = str(text) if pd.notna(text) else ""

        text = re.sub(r'[^آ-ی۰-۹\s]', '', text)
        text = re.sub(r'[0-9]', lambda x: chr(ord(x.group()) + 1728), text)
        text = self.handle_correct_alphabets(text)
        text = re.sub(r"(.)\1{3,}", r"\1", text)
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'(\n|\r)', ' ', text)
        text = ' '.join([self.normalizer.normalize(word) for word in text.split()])

        return text

def preprocess_data(df):

    df['description_links'] = df['description'].apply(extract_links)
    df['description_hashtags'] = df['description'].apply(extract_hashtags)
    df['description_mentions'] = df['description'].apply(extract_mentions)
    df['description'] = df['description'].apply(lambda x: re.sub(r'http[s]?://\S+|#\S+|@\S+', '', str(x)).strip())

    df['reply_links'] = df['in_reply_to_text'].apply(extract_links)
    df['reply_hashtags'] = df['in_reply_to_text'].apply(extract_hashtags)
    df['reply_mentions'] = df['in_reply_to_text'].apply(extract_mentions)
    df['in_reply_to_text'] = df['in_reply_to_text'].apply(lambda x: re.sub(r'http[s]?://\S+|#\S+|@\S+', '', str(x)).strip())

    preprocessor = Preprocess()
    df['description'] = df['description'].apply(preprocessor.clean)
    df['in_reply_to_text'] = df['in_reply_to_text'].apply(preprocessor.clean)

    return df

if __name__ == "__main__":
    print("Preprocessing module is ready!")
