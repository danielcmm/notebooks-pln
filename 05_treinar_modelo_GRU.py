import argparse
import os
import numpy as np
from gensim import models
from keras.callbacks import ModelCheckpoint, LambdaCallback, TensorBoard
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Embedding, CuDNNGRU, GRU
from keras.models import Sequential, load_model
from keras import optimizers
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

caminho_modelo_w2v = "modelos/w2v_sertanejo_28kmusicas_tweet_tknzr_window_15_mincount_10.model"
if not os.path.isfile(caminho_modelo_w2v):
    raise FileExistsError("Modelo w2v nao encontrado.")

w2v_model = models.Word2Vec.load(caminho_modelo_w2v)
vocab = list(w2v_model.wv.vocab)


def word2idx(word):
    return w2v_model.wv.vocab[word].index


def idx2word(idx):
    return w2v_model.wv.index2word[idx]


with open('modelos/dados_treino_mincount_10_vagalume.pkl', 'rb') as fp:
    dados_treino = pickle.load(fp)

x = dados_treino[0]
y = dados_treino[1]

tamanho_sentencas = 15
pretrained_weights = w2v_model.wv.vectors
tamanho_vocab = pretrained_weights.shape[0]
tamanho_vetor_w2v = pretrained_weights.shape[1]  # 350
print("Tamanho vocab e w2v vector: ", (tamanho_vocab, tamanho_vetor_w2v))
print("Quantidade de sentencas para treino: {}".format(len(x)))

units1 = args["units1"]
units2 = args["units2"]
learning_rate = args["lr"]
nome_modelo = "gru-w2v-wordlevel-mincount10-{}-len-{}-{}-lr-{}-sertanejo-vagalume.model".format(tamanho_sentencas, units1, units2, learning_rate)
caminho_modelo_lstm = "modelos/{}".format(nome_modelo)


if os.path.isfile(caminho_modelo_lstm):
    print("Carregando modelo lstm previo...")
    model = load_model(caminho_modelo_lstm)
else:
    print("Criando novo modelo GRU")
    model = Sequential()
    model.add(Embedding(input_dim=tamanho_vocab, output_dim=tamanho_vetor_w2v, weights=[pretrained_weights]))

    if args["gpu"]:
        model.add(CuDNNGRU(units1, return_sequences=True))
    else:
        model.add(GRU(units1, return_sequences=True))

    model.add(Dropout(0.1))

    if args["gpu"]:
        model.add(CuDNNGRU(units=units2))
    else:
        model.add(GRU(units=units2))
    model.add(Dropout(0.1))

    model.add(Dense(tamanho_vocab, activation='softmax'))  # Quantidade de 'respostas' possiveis. Tokens neste caso.
    optimizer = optimizers.RMSprop(lr=learning_rate)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=["sparse_categorical_accuracy"])


def gerar_texto(epoch, logs):
    print('----- Gerando texto apos a Epoch: %d' % epoch)

    idx_sentenca = np.random.randint(0, len(x))
    idx_tokens = list(x[idx_sentenca])
    seed_tokens = [idx2word(i) for i in idx_tokens]
    print("Seed:")
    print("\"", " ".join(seed_tokens), "\"\n")
    temperatura = args["t"]

    aux = 1
    for i in range(500):

        xt = np.zeros((1, tamanho_sentencas), dtype=np.int32)

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
