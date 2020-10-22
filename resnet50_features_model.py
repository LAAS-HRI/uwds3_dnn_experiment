from keras.applications.resnet50 import ResNet50
from keras.layers import Input
from keras.models import Model


if __name__ == '__main__':
    model = ResNet50(weights='imagenet', input_shape=(224,224,3), include_top=False, pooling='avg')
    model.trainable = False
    model.summary()
    model.save("resnet50_features_model.h5")
    model_json = model.to_json()
    with open("./resnet50_features_model.json", "w") as json_file:
        json_file.write(model_json)
