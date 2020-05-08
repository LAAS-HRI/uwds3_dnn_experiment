from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.models import Model, load_model
from keras.losses import categorical_crossentropy
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.preprocessing.image import ImageDataGenerator


def eye_contact_cnn(input_shape):
    """ Model based on Mitsuzumi, Y. et al.: Deep eyecontact detector: Robust
    eye contact bid detection using convolutional neural network. (BMVC).(2017)
    modified to process only one eye at the time
    """
    input = Input(input_shape)

    conv1 = Conv2D(20, (3, 3), padding="same",
                   activation="relu")(input)

    conv2 = Conv2D(20, (3, 3), padding="same",
                   activation="relu")(conv1)

    pooling1 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(20, (3, 3), padding="same",
                   activation="relu")(pooling1)

    conv4 = Conv2D(20, (3, 3), padding="same",
                   activation="relu")(conv3)

    pooling2 = MaxPooling2D(pool_size=(2, 2))(conv4)

    flatten = Flatten()(pooling2)

    dense1 = Dense(512, activation="relu")(flatten)

    dropout = Dropout(0.5)(dense1)

    output = Dense(2, activation="softmax")(dropout)

    model = Model(input, output)
    model.summary()
    model.compile(optimizer="adam", loss=categorical_crossentropy, metrics=["accuracy"])
    return model


if __name__ == '__main__':
    batch_size = 300
    epochs = 400
    nb_train_samples = 3959
    nb_validation_samples = 2000

    input_shape = (36, 60, 3)
    target_size = (36, 60)

    eye_contact_train_data_path = "./data/eye_contact/train/"
    eye_contact_val_data_path = "./data/eye_contact/validation/"

    train_data_generator = ImageDataGenerator(rotation_range=0,
                                              height_shift_range=0.0,
                                              width_shift_range=0.0,
                                              shear_range=0.05,
                                              zoom_range=0.05,
                                              horizontal_flip=True,
                                              vertical_flip=True,
                                              rescale=1./255,
                                              fill_mode="nearest")

    test_data_generator = ImageDataGenerator(rescale=1./255)

    train_generator = train_data_generator.flow_from_directory(eye_contact_train_data_path,
                                                               target_size=target_size,
                                                               batch_size=batch_size,
                                                               class_mode="categorical")

    validation_generator = test_data_generator.flow_from_directory(eye_contact_val_data_path,
                                                                   target_size=target_size,
                                                                   batch_size=batch_size,
                                                                   class_mode="categorical")

    model = eye_contact_cnn(input_shape)

    board = TensorBoard(log_dir="./tensorboard", histogram_freq=0, write_graph=True, write_images=True)
    early_stop = EarlyStopping(monitor='val_acc', patience=7, verbose=1, mode='max')
    mcp_save = ModelCheckpoint("./model.h5", save_best_only=True, monitor='val_acc', verbose=1, mode='max')

    model.fit_generator(train_generator,
                        steps_per_epoch=nb_train_samples // batch_size,
                        epochs=epochs,
                        validation_data=validation_generator,
                        validation_steps=nb_validation_samples // batch_size,
                        callbacks=[early_stop, mcp_save, board])

    best_model = load_model("./model.h5")
    model_json = best_model.to_json()
    with open("./model.json", "w") as json_file:
        json_file.write(model_json)
