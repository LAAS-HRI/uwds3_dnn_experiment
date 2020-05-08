import cv2
import argparse


def parse_args():
    """Parses command line arguments."""
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")

    parser.add_argument(
        "--weigths",
        type=str,
        required=False,
        default="exported_model/optimized_model.pb",
        help="Tensorflow weights")
    parser.add_argument(
        "--model",
        type=str,
        required=False,
        default="exported_model/model.pbtxt",
        help="Tensorflow pbtxt file")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    model = cv2.dnn.readNetFromTensorflow(args.weigths, args.model)

    img = cv2.imread('test.png')

    print("Input shape: {}".format(img.shape))

    normalized_img = img/255

    blob = cv2.dnn.blobFromImage(normalized_img, size=img.shape[:2], swapRB=True, crop=False)

    print blob

    model.setInput(blob)

    preds = model.forward()

    print ("Prediction: {}".format(preds))
