import keras
import tensorflow as tf

from tf.keras.applications import ResNet50


if __name__ == '__main__':
    model = ResNet50(weights='imagenet', include_top=False, pooling='avg', trainable=False)
    model.save("resnet50_features_model.h5")
    model_json = model.to_json()
    with open("./resnet50_features_model.json", "w") as json_file:
        json_file.write(model_json)
