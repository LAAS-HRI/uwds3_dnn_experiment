from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.models import Model
from keras.regularizers import l2
from keras.losses import binary_crossentropy
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.preprocessing.image import ImageDataGenerator


def eye_contact_cnn(input_shape):
    """ Model based on Mitsuzumi, Y., Nakazawa et al.: Deep eyecontact detector:
    Robust eye contact bid detection using convolutional neural network. (BMVC).(2017)
    """
    input = Input(input_shape)

    conv1 = Conv2D(20, (3, 3), padding="same", activation="relu",
                   kernel_regularizer=l2(0.0001))(input)

    conv2 = Conv2D(20, (3, 3), padding="same", activation="relu",
                   kernel_regularizer=l2(0.0001))(conv1)

    pooling1 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(20, (3, 3), padding="same", activation="relu",
                   kernel_regularizer=l2(0.0001))(pooling1)

    conv4 = Conv2D(20, (3, 3), padding="same", activation="relu",
                   kernel_regularizer=l2(0.0001))(conv3)

    pooling2 = MaxPooling2D(pool_size=(2, 2))(conv4)

    flatten = Flatten()(pooling2)

    dense1 = Dense(512, activation="relu")(flatten)

    dropout1 = Dropout(0.1)(dense1)

    dense2 = Dense(1024, activation="relu")(dropout1)

    dropout2 = Dropout(0.1)(dense2)

    dense3 = Dense(1024, activation="relu")(dropout2)

    dropout3 = Dropout(0.2)(dense3)

    preds = Dense(1, activation="sigmoid")(dropout3)

    model = Model(input, preds)
    model.summary()
    model.compile(optimizer="adam", loss=binary_crossentropy, metrics=['accuracy'])
    return model


if __name__ == '__main__':
    batch_size = 50
    epochs = 400
    nb_train_samples = 6200
    nb_validation_samples = 4000

    input_shape = (32, 16, 3)
    target_size = (32, 16)

    eye_contact_train_data_path = "./data/eye_contact/train/"
    eye_contact_val_data_path = "./data/eye_contact/validation/"

    train_data_generator = ImageDataGenerator(rotation_range=20,
                                              height_shift_range=0.0,
                                              width_shift_range=0.0,
                                              shear_range=0.02,
                                              zoom_range=0.05,
                                              horizontal_flip=True,
                                              vertical_flip=True,
                                              rescale=1./255,
                                              fill_mode="nearest")

    test_data_generator = ImageDataGenerator(rescale=1./255)

    train_generator = train_data_generator.flow_from_directory(eye_contact_train_data_path,
                                                               target_size=target_size,
                                                               batch_size=batch_size,
                                                               class_mode="binary")

    validation_generator = test_data_generator.flow_from_directory(eye_contact_val_data_path,
                                                                   target_size=target_size,
                                                                   batch_size=batch_size,
                                                                   class_mode="binary")

    model = eye_contact_cnn(input_shape)

    board = TensorBoard(log_dir='./tensorboard', histogram_freq=0, write_graph=True, write_images=True)
    early_stop = EarlyStopping(monitor='val_acc', patience=20, verbose=1, mode='max')
    mcp_save = ModelCheckpoint('./data/eye_contact/eye_contact_cnn.hdf5', save_best_only=True, monitor='val_acc', verbose=1, mode='max')

    model.fit_generator(train_generator,
                        steps_per_epoch=nb_train_samples // batch_size,
                        epochs=epochs,
                        validation_data=validation_generator,
                        validation_steps=nb_validation_samples // batch_size,
                        callbacks=[early_stop, mcp_save, board])
