import keras
from keras.layers import Input
from keras.models import Model, load_model
from keras.losses import binary_crossentropy
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD


REGUL = 1e-5
DROPOUT = 0.5


def eye_contact_cnn(input_shape):
    """ Model based on Mitsuzumi, Y. et al.: Deep eyecontact detector: Robust
    eye contact bid detection using convolutional neural network. (BMVC).(2017)
    modified to process only one eye at the time
    """
    input = Input(input_shape)

    conv1 = Conv2D(30, (3, 3), padding="same",
                   kernel_regularizer=l2(REGUL),
                   activation="relu")(input)

    dropout1 = Dropout(DROPOUT)(conv1)

    conv2 = Conv2D(30, (3, 3), padding="same",
                   kernel_regularizer=l2(REGUL),
                   activation="relu")(dropout1)

    dropout2 = Dropout(DROPOUT)(conv2)

    pooling1 = MaxPooling2D(pool_size=(2, 2))(dropout2)

    conv3 = Conv2D(30, (3, 3), padding="same",
                   kernel_regularizer=l2(REGUL),
                   activation="relu")(pooling1)

    dropout3 = Dropout(DROPOUT)(conv3)

    conv4 = Conv2D(30, (3, 3), padding="same",
                   kernel_regularizer=l2(REGUL),
                   activation="relu")(dropout3)

    dropout4 = Dropout(DROPOUT)(conv4)

    pooling2 = MaxPooling2D(pool_size=(2, 2))(dropout4)

    flatten = Flatten()(pooling2)

    dense1 = Dense(512, activation="relu", kernel_regularizer=l2(REGUL))(flatten)

    dropout5 = Dropout(DROPOUT)(dense1)

    dense2 = Dense(1024, activation="relu", kernel_regularizer=l2(REGUL))(dropout5)

    dropout6 = Dropout(DROPOUT)(dense2)

    dense3 = Dense(1024, activation="relu", kernel_regularizer=l2(REGUL))(dropout6)

    dropout7 = Dropout(DROPOUT)(dense3)

    output = Dense(1, activation="sigmoid")(dropout7)

    model = Model(input, output)
    model.summary()
    model.compile(optimizer=SGD(lr=0.01, nesterov=True), loss=binary_crossentropy, metrics=["accuracy"])
    return model


if __name__ == '__main__':
    batch_size = 20
    epochs = 200
    nb_train_samples = 1200
    nb_validation_samples = 600

    input_shape = (36, 60, 1)
    target_size = (36, 60)

    eye_contact_train_data_path = "./data/eye_contact/train/"
    eye_contact_val_data_path = "./data/eye_contact/validation/"

    train_data_generator = ImageDataGenerator(rotation_range=5,
                                              height_shift_range=0.0,
                                              width_shift_range=0.0,
                                              shear_range=0.0,
                                              zoom_range=0.2,
                                              horizontal_flip=False,
                                              vertical_flip=True,
                                              rescale=1./255,
                                              fill_mode="nearest")

    test_data_generator = ImageDataGenerator(rescale=1./255)

    train_generator = train_data_generator.flow_from_directory(eye_contact_train_data_path,
                                                               target_size=target_size,
                                                               batch_size=batch_size,
                                                               color_mode="grayscale",
                                                               class_mode="binary")

    validation_generator = test_data_generator.flow_from_directory(eye_contact_val_data_path,
                                                                   target_size=target_size,
                                                                   batch_size=batch_size,
                                                                   color_mode="grayscale",
                                                                   class_mode="binary")

    model = eye_contact_cnn(input_shape)

    board = TensorBoard(log_dir="./tensorboard", histogram_freq=0, write_graph=True, write_images=True)
    early_stop = EarlyStopping(monitor='val_loss', patience=7, verbose=1, mode='min')
    reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.98, min_lr=0.001, patience=5, verbose=1)
    mcp_save = ModelCheckpoint("./model.h5", save_best_only=True, monitor='val_loss', verbose=1, mode='min')

    model.fit_generator(train_generator,
                        steps_per_epoch=nb_train_samples // batch_size,
                        epochs=epochs,
                        validation_data=validation_generator,
                        validation_steps=nb_validation_samples // batch_size,
                        callbacks=[early_stop, mcp_save, board, reduce])

    best_model = load_model("./model.h5")
    model_json = best_model.to_json()
    with open("./model.json", "w") as json_file:
        json_file.write(model_json)
