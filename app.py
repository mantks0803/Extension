from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
import re
import unicodedata
import emoji
import os
import spacy

app = Flask(__name__)
CORS(app)


class MyVocab():
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.vocab_size = None

    def load_vocab(self, path="vocab.txt"):
        with open(path, "r", encoding="utf-8") as f:
            vocab = [line.strip() for line in f.readlines()]
        self.idx2word = vocab
        self.word2idx = {}
        for i, w in enumerate(vocab):
            if w not in self.word2idx:
                self.word2idx[w] = i
        self.vocab_size = len(vocab)

    def mapping(self, tokens):
        unk_idx = self.word2idx.get("<UNK>", 0)
        return [self.word2idx.get(t, unk_idx) for t in tokens]


class DataPreprocess():
    def __init__(self, stopwords, tokenizer, emoji_labels):
        self.stopwords = stopwords
        self.tokenizer = tokenizer          # tokenizer là spacy pipeline
        self.emoji_labels = emoji_labels
        self.__emoji_sorted = sorted(self.emoji_labels.keys(), key=len, reverse=True)
        self.__emoji_regex = "|".join(re.escape(k) for k in self.__emoji_sorted)
        self.__pattern = re.compile(self.__emoji_regex, flags=re.IGNORECASE)
        self.vietnamese_pattern = r"[^a-zA-Z0-9_\sàáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđĐ]"

    def process(self, text: str) -> list:
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)
        text = text.lower()

        text = self.replace_emoji(text)
        text = emoji.demojize(text, delimiters=(" emoji_", " "))

        placeholders = re.findall(r"<[^<>]+>", text)
        for i, ph in enumerate(placeholders):
            text = text.replace(ph, f"__PLACEHOLDER{i}__")

        text = re.sub(self.vietnamese_pattern, "", text, flags=re.UNICODE)

        for i, ph in enumerate(placeholders):
            text = text.replace(f"__PLACEHOLDER{i}__", ph)

        tokens = self.tokenize(text)
        return tokens

    def __repl(self, match: str):
        key = match.group().lower()
        return " " + self.emoji_labels[key] + " "

    def replace_emoji(self, text: str):
        return self.__pattern.sub(self.__repl, text)

    def tokenize(self, text: str) -> list:
        # Dùng spacy để tách từ giống như lúc train
        tokens = [token.text for token in self.tokenizer(text)]
        cleaned_token = []
        for token in tokens:
            if re.match(r'^\s*$', token):
                continue
            if token in self.emoji_labels.values():
                cleaned_token.append(f"<{token}>")
                continue
            if token.startswith("emoji_"):
                cleaned_token.append(f"<{token[6:]}>")
                continue
            if re.search(r'\d', token):
                if re.fullmatch(r'\d+', token):
                    cleaned_token.append("<NUM>")
                else:
                    cleaned_token.append("<UNK>")
            else:
                cleaned_token.append(token)
        return cleaned_token


class VSFCClassifier(nn.Module):
    def __init__(self, vocab_size, sentiment_classes, pad_idx,
                 embed_dim=128, hidden_dim=256, num_filters=64, kernel_size=3):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=pad_idx
        )

        self.conv = nn.Conv1d(in_channels=embed_dim, out_channels=num_filters,
                              kernel_size=kernel_size, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)

        self.encoder = nn.LSTM(
            input_size=64,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True
            # dropout=0.5
        )

        self.attn = nn.Linear(in_features=hidden_dim * 2, out_features=1)
        self.sentiment_head = nn.Linear(in_features=hidden_dim * 2, out_features=sentiment_classes)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x = x.permute(0, 2, 1)
        c = self.conv(x)
        c = self.relu(c)
        c = self.pool(c)
        c = c.permute(0, 2, 1)
        enc_out, _ = self.encoder(c)
        attn_weights = torch.softmax(self.attn(enc_out), dim=1)
        pooled = torch.sum(enc_out * attn_weights, dim=1)
        sentiment_logits = self.sentiment_head(pooled)
        return sentiment_logits


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[*] Server đang khởi động với thiết bị: {device}")

# Tải danh sách emoji
emoji_labels = {
    ':))': 'colonsmilesmile', ':)': 'colonsmile', ':(': 'colonsad', '@@': 'colonsurprise',
    '<3': 'colonlove', ':d': 'colonsmilesmile', ':3': 'coloncontemn', ':v': 'colonbigsmile',
    ':_': 'coloncc', ':p': 'colonsmallsmile', '>>': 'coloncolon', ':">': 'colonlovelove',
    '^^': 'colonhihi', ':': 'doubledot', ':(': 'colonsadcolon', ':’(': 'colonsadcolon',
    ':@': 'colondoublesurprise', 'v.v': 'vdotv', '...': 'dotdotdot', '/': 'fraction', 'c#': 'cshrap'
}

nlp = spacy.blank('vi')

preprocess = DataPreprocess(stopwords=[], tokenizer=nlp, emoji_labels=emoji_labels)

# Tải từ điển
vocab = MyVocab()
try:
    vocab.load_vocab('vocab.txt')
    print(f"[*] Đã tải thành công từ điển với {vocab.vocab_size} từ.")
except Exception as e:
    print("[!] Lỗi: Không tìm thấy file 'vocab.txt'. Hãy đảm bảo file nằm cùng thư mục.")
    exit(1)

# Khởi tạo mô hình
pad_idx = vocab.word2idx.get("<PAD>", 0)
model = VSFCClassifier(
    vocab_size=vocab.vocab_size,
    sentiment_classes=3,
    pad_idx=pad_idx
)

try:
    model.load_state_dict(torch.load('vsfc_model.pth', map_location=device))
    model.eval()
    model.to(device)
    print("[*] Não bộ AI 'vsfc_model.pth' đã được kích hoạt thành công!")
except Exception as e:
    print(f"[!] Lỗi nạp Model: {e}")
    exit(1)


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'Vui lòng cung cấp tham số "text"'}), 400

    text = data['text']

    # Tiền xử lý và mapping
    tokens = preprocess.process(text)
    token_ids = vocab.mapping(tokens)

    sos_idx = vocab.word2idx.get("<SOS>", 0)
    eos_idx = vocab.word2idx.get("<EOS>", 0)
    token_ids = [sos_idx] + token_ids + [eos_idx]

    input_tensor = torch.tensor([token_ids], dtype=torch.long).to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        pred = logits.argmax(dim=1).item()

    labels = {0: '🟢 TÍCH CỰC', 1: '🟡 TRUNG TÍNH', 2: '🔴 TIÊU CỰC'}
    print(f"\n---> Khách hàng: '{text}'")
    print(f"---> AI   : {labels[pred]}")

    return jsonify({
        'result': labels[pred],
        'tokens': tokens
    })


if __name__ == '__main__':
    print("\n" + "=" * 50)
    print(" Sẵn sàng, port : 5000")
    print("=" * 50 + "\n")
    app.run(port=5000, debug=False)