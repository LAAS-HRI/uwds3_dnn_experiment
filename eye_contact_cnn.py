from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.models import Model
from keras.regularizers import l2
from keras.losses import binary_crossentropy
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator


def eye_contact_cnn(input_shape):
    input = Input(input_shape)

    conv1 = Conv2D(16, (1, 1), padding="same", activation="relu",
                   kernel_regularizer=l2(0.0001))(input)
    pooling1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    dropout1 = Dropout(0.1)(pooling1)

    conv2 = Conv2D(32, (3, 3), padding="same", activation="relu",
                   kernel_regularizer=l2(0.0001))(dropout1)
    pooling2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    dropout2 = Dropout(0.1)(pooling2)

    conv3 = Conv2D(64, (3, 3), padding="same", activation="relu",
                   kernel_regularizer=l2(0.0001))(dropout2)
    pooling3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    dropout3 = Dropout(0.1)(pooling3)

    features = Flatten()(dropout3)
    features = Dense(512, activation="relu")(features)
    dropout = Dropout(0.5)(features)
    preds = Dense(1, activation="sigmoid")(dropout)

    model = Model(input, preds)
    model.summary()
    model.compile(optimizer="rmsprop", loss=binary_crossentropy, metrics=['accuracy'])
    return model


if __name__ == '__main__':
    batch_size = 4
    epochs = 30
    nb_train_samples = 1600
    nb_validation_samples = 600

    input_shape = (32, 16, 3)
    target_size = (32, 16)

    eye_contact_train_data_path = "./data/eye_contact/train/"
    eye_contact_val_data_path = "./data/eye_contact/validation/"

    train_data_generator = ImageDataGenerator(rotation_range=20,
                                              shear_range=0.02,
                                              zoom_range=0.05,
                                              horizontal_flip=True,
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

    early_stop = EarlyStopping(monitor='val_loss', patience=6, verbose=1, mode='min')
    mcp_save = ModelCheckpoint('./data/eye_contact/eye_contact_cnn.hdf5', save_best_only=True, monitor='val_loss', verbose=1, mode='min')

    model.fit_generator(train_generator,
                        steps_per_epoch=nb_train_samples // batch_size,
                        epochs=epochs,
                        validation_data=validation_generator,
                        validation_steps=nb_validation_samples // batch_size,
                        callbacks=[early_stop, mcp_save])
