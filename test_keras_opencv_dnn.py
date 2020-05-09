import cv2
import numpy as np
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
    parser.add_argument(
        "--image",
        type=str,
        required=False,
        default="test.png",
        help="Test image")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print("Loading model...")
    model = cv2.dnn.readNetFromTensorflow(args.weigths, args.model)
    print("Model loaded, trying basic inference:")
    bgr_img = cv2.imread(args.image)

    print("Input shape: {}".format(bgr_img.shape))

    blob = cv2.dnn.blobFromImage(bgr_img,
                                 1.0/255,
                                 bgr_img.shape[:2],
                                 (0, 0, 0),
                                 swapRB=True,
                                 crop=False)

    print("Input tensor:")

    print(blob)

    model.setInput(blob)

    output = model.forward()
    print("Output tensor: {}".format(output))

    print("Test batch inference:")

    imgs = []

    imgs.append(bgr_img)
    imgs.append(bgr_img)

    print("Batch length {}".format(len(imgs)))

    print("Input tensor:")

    blob2 = cv2.dnn.blobFromImages(imgs,
                                   1.0/255,
                                   bgr_img.shape[:2],
                                   (0, 0, 0),
                                   swapRB=True,
                                   crop=False)
    print blob2
    model.setInput(blob2)
    output = model.forward()
    print("Output tensor: {}".format(output))
