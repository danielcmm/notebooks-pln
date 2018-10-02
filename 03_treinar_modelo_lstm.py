import argparse
import os
import numpy as np
from gensim import models
from keras.callbacks import ModelCheckpoint, LambdaCallback, TensorBoard
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM, Embedding, CuDNNLSTM
from keras.models import Sequential, load_model
from keras.optimizers import Adam
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("-epochs", type=int, default=1)
parser.add_argument("-units1", type=int, default=30)
parser.add_argument("-units2", type=int, default=30)
parser.add_argument("-batchsize", type=int, default=64)
parser.add_argument("-gpu", type=bool, default=False)
parser.add_argument("-t", type=int, default=5)
parser.add_argument("-lr", type=float, default=0.001)

args = vars(parser.parse_args())
print(args)

with open('modelos/docs_tokenizados_vagalume.pkl', 'rb') as fp:
    docs_tokenizados = pickle.load(fp)

caminho_modelo_w2v = "modelos/w2v_sertanejo_28kmusicas_tweet_tknzr_window_15_mincount_0.model"
if not os.path.isfile(caminho_modelo_w2v):
    raise FileExistsError("Modelo w2v nao encontrado.")

w2v_model = models.Word2Vec.load(caminho_modelo_w2v)
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
learning_rate = args["lr"]
nome_modelo = "lstm-w2v-wordlevel-{}len-{}-{}-lr-{}sertanejo-vagalume.model".format(maxlen, units1, units2, learning_rate)
caminho_modelo_lstm = "modelos/{}".format(nome_modelo)
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
    optimizer = Adam(lr=learning_rate)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=["accuracy"] )


def gerar_texto(epoch, logs):
    print('----- Generating text after Epoch: %d' % epoch)

    start = np.random.randint(0, len(sentences))
    seed_tokens = list(sentences[start])
    print("Seed:")
    print("\"", " ".join(seed_tokens), "\"\n")
    temperatura = args["t"]

    aux = 1
    for i in range(500):

        xt = np.zeros((1, maxlen), dtype=np.int32)

        for t, token in enumerate(seed_tokens):
            xt[0, t] = word2idx(token)

        prediction = model.predict(xt, verbose=0)

        if temperatura > 0 and aux % temperatura == 0:
            ordered = (-prediction[0]).argsort()[:3]
            indice = np.random.choice(ordered)
        else:
            indice = (np.argmax(prediction))

        result = idx2word(indice)
        print("\n" if result == "nlinha" else result, end=" ")
        seed_tokens.append(result)
        seed_tokens = seed_tokens[1:len(seed_tokens)]
        aux = aux + 1

    print("\n\nFIM\n\n", )


checkpoint = ModelCheckpoint(caminho_modelo_lstm, monitor='loss', verbose=1, save_best_only=True, mode='min')
print_callback = LambdaCallback(on_epoch_end=gerar_texto)
tensorboard = TensorBoard(log_dir="logs/{}".format(nome_modelo))
callbacks_list = [checkpoint, print_callback, tensorboard]

print(model.summary())

model.fit(x, y, epochs=args["epochs"], batch_size=args["batchsize"], callbacks=callbacks_list)
