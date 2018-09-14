import os

import nltk
from bs4 import BeautifulSoup
from gensim import models
from keras.callbacks import ModelCheckpoint, LambdaCallback
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM, Embedding, Bidirectional, CuDNNLSTM
from keras.models import Sequential, load_model
from sklearn import feature_extraction
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-epochs", type=int, default=1)
parser.add_argument("-units1", type=int, default=30)
parser.add_argument("-units2", type=int, default=30)
parser.add_argument("-batchsize", type=int, default=64)
parser.add_argument("-gpu", type=bool, default=False)

args = vars(parser.parse_args())
print(args)

documentos = []
arquivos = os.listdir("dados/letras-musicas")
for arquivo in arquivos:
    caminho = "dados/letras-musicas/" + arquivo
    if os.path.isfile(caminho):
        with open(caminho, "r") as f:
            html = f.read()
        html = feature_extraction.text.strip_accents_ascii(html.lower())
        html = html.replace("<br/>", " NLINHA ")
        html = html.replace("<p>", " NLINHA ")
        soup = BeautifulSoup(html, "html.parser")
        documentos.append(soup.text.strip())

docs_tokenizados = []
tokenizador = nltk.TreebankWordTokenizer()
for doc in documentos:
    tokens = tokenizador.tokenize(doc)
    docs_tokenizados.append(tokens)

caminho_modelo_w2v = "modelos/w2v_sertanejo_1000musicas.model"
if os.path.isfile(caminho_modelo_w2v):
    print("Carregando modelo w2v previo...")
    w2v_model = models.Word2Vec.load(caminho_modelo_w2v)
else:
    print("Treinando modelo w2v...")
    w2v_model = models.Word2Vec(docs_tokenizados, size=350, window=15, min_count=0, workers=os.cpu_count(), iter=300)
    w2v_model.save(caminho_modelo_w2v)

vocab = list(w2v_model.wv.vocab)


# Funcoes que retornam o indice da palavra no vocabulario criado pelo word2vec e vice-versa
def word2idx(word):
    return w2v_model.wv.vocab[word].index


def idx2word(idx):
    return w2v_model.wv.index2word[idx]


maxlen = 15
sentences = []
next_tokens = []

for doc in docs_tokenizados:  # Cria as "sentencas" para cada documento do corpus
    for i in range(0, len(doc) - maxlen):
        sentences.append(doc[i:i + maxlen])
        next_tokens.append(doc[i + maxlen])

print('Number of sequences:', len(sentences), "\n")

# Criando os inputs(x) e targets(y)
x = np.zeros((len(sentences), maxlen), dtype=np.int32)
y = np.zeros((len(sentences)), dtype=np.int32)
for i, sentence in enumerate(sentences):
    for t, token in enumerate(sentence):
        x[i, t] = word2idx(token)
    y[i] = word2idx(next_tokens[i])

pretrained_weights = w2v_model.wv.vectors
tamanho_vocab = pretrained_weights.shape[0]
tamanho_vetor_w2v = pretrained_weights.shape[1]  # 350
print("Tamanho vocab e w2v vector: ", (tamanho_vocab, tamanho_vetor_w2v))

units1 = args["units1"]
units2 = args["units2"]
caminho_modelo_lstm = "modelos/lstm-w2v-wordlevel-{}len-{}-{}-sertanejo.model".format(maxlen, units1, units2)
if os.path.isfile(caminho_modelo_lstm):
    print("Carregando modelo lstm previo...")
    model = load_model(caminho_modelo_lstm)
else:
    print("Criando novo modelo LSTM")
    model = Sequential()
    model.add(Embedding(input_dim=tamanho_vocab, output_dim=tamanho_vetor_w2v, weights=[pretrained_weights]))

    if args["gpu"]:
        model.add(CuDNNLSTM(units1, return_sequences=True))
    else:
        model.add(LSTM(units1, return_sequences=True))

    model.add(Dropout(0.1))

    if args["gpu"]:
        model.add(CuDNNLSTM(units=units2))
    else:
        model.add(LSTM(units=units2))
    model.add(Dropout(0.1))

    model.add(Dense(tamanho_vocab, activation='softmax'))  # Quantidade de 'respostas' possiveis. Tokens neste caso.
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=["accuracy"])


def gerar_texto(epoch, logs):
    print('----- Generating text after Epoch: %d' % epoch)

    # pick a random seed
    start = np.random.randint(0, len(sentences))
    seed_tokens = list(sentences[start])
    print("Seed:")
    print("\"", " ".join(seed_tokens), "\"\n")

    # generate characters
    for i in range(500):

        xt = np.zeros((1, maxlen), dtype=np.int32)

        for t, token in enumerate(seed_tokens):
            xt[0, t] = word2idx(token)

        prediction = model.predict(xt, verbose=0)

        # ordered = (-prediction[0]).argsort()[:5]
        # indice = np.random.choice(ordered)
        indice = (np.argmax(prediction))
        result = idx2word(indice)
        print("\n" if result == "NLINHAasda" else result, end=" ")
        seed_tokens.append(result)
        seed_tokens = seed_tokens[1:len(seed_tokens)]

    print("\n\nFIM\n\n", )


checkpoint = ModelCheckpoint(caminho_modelo_lstm, monitor='loss', verbose=1, save_best_only=True, mode='min')
print_callback = LambdaCallback(on_epoch_end=gerar_texto)
callbacks_list = [checkpoint, print_callback]

print(model.summary())

model.fit(x, y, epochs=args["epochs"], batch_size=args["batchsize"], callbacks=callbacks_list)
