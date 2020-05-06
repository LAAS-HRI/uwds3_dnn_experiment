from word_embeddings import create_embedding_layer
from siamese import siamese_model
from keras.models import Model
from keras.layers import Input, LSTM, Dropout, Dense, TimeDistributed


def base_model(pre_trained_embedding_file, max_length, hidden_dim, vector_dim):
    """Create base model of the siamese network
    """
    input = Input([max_length])
    embedding = create_embedding_layer(pre_trained_embedding_file)(input)
    x = LSTM(hidden_dim, dropout=0.5)(embedding)
    preds = Dense(vector_dim, activation="sigmoid")(x)
    model = Model(input, preds)
    print "Sentence features extractor:"
    model.summary()
    return model


def siamese_lstm(pre_trained_embedding_file, max_length, lstm_hidden_dim, vector_dim):
    """Create siamese network
    """
    base = base_model(pre_trained_embedding_file, max_length, lstm_hidden_dim, vector_dim)
    return siamese_model([max_length], base, metric="cosine")


if __name__ == '__main__':
    glove_embedding_file = "./models/word_embeddings/glove.6B.300d.txt"
    model = siamese_lstm(glove_embedding_file, 20, 256, 100)
