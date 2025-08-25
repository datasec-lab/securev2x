"""
Evaluate keras model accuracy on generated test images
"""
import argparse
import numpy as np
import os
import random
import sys
# next line is not necessary for anything important
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
import tensorflow as tf
from tensorflow.keras.datasets import cifar100
from importlib.util import spec_from_file_location, module_from_spec
from os import path

def generate_images(image_path, num_images, folder_path):
    """Sample and save a random set of num_images images"""
    # Normalize dataset
    (_, _), (x_test, y_test) = cifar100.load_data()
    x_test = x_test.astype("float32")
    x_test /= 255
    # Sample and save images 
    sample_set = random.sample(list(range(len(x_test))), num_images)
    images = [x_test[i] for i in sample_set]
    classes = [y_test[i] for i in sample_set]
    for i, (img, cls) in enumerate(zip(images, classes)):
        # Reshape image for rust
        _, _, chans = img.shape
        rust_tensor = np.array([[img[:, :, c] for c in range(chans)]])
        np.save(os.path.join(image_path, f"image_{i}.npy"), rust_tensor.flatten().astype(np.float64))
    np.save(path.join(folder_path, f"classes.npy"), np.array(classes).flatten().astype(np.int64))


def test_network(model, image_path, folder_path):
    """Gets inference results from given network"""
    # Load image classes
    classes = np.load(path.join(folder_path, "classes.npy"))
    # Run inference on all images and track plaintext predictions
    results_path = path.join(folder_path, "py_predict.txt")
    correct = []
    with open(results_path, "w") as f:
        for i in range(len(classes)):
            # Load image and reshape to proper shape
            image = np.load(path.join(image_path, f"image_{i}.npy")).reshape(3, 32, 32)
            image = np.array([[image[:, i, j] for j in range(32)] for i in range(32)]).reshape(32, 32, 3)
            prediction = np.argmax(model.predict(np.expand_dims(image, axis=0))) 
            f.write("Sample = {} | Prediction = {} | Class = {}\n".format(i, prediction, classes[i]))
            correct += [1] if prediction == classes[i] else [0]
    # Save prediction results    
    np.save(path.join(folder_path, "plaintext.npy"), np.array(correct))
    return 100 * (sum(correct) / len(classes))

def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument('model', type=int, help='<REQUIRED> Use Minionn (0) or Resnet32 (1)')
    # parser.add_argument('-w', '--weights_path', required=True, type=str,
    #                     help='<REQUIRED> Path to model weights')
    # parser.add_argument('-a', '--approx', nargs='+', type=int, required=False,
    #                     help='Set approx layesrs')
    # parser.add_argument('-i', '--image_path', required=False, type=str,
    #                     help='Path to place images')
    # parser.add_argument('-g', '--generate', required=False, type=int,
    #                     help='How many images to generate (default 0)')
    # args = parser.parse_args()

    folder_path = "predictions"
    # Load the correct model and dataset
    dataset = cifar100
    model_path = path.abspath("../../../../../python/resnet/resnet32_model.py")
    num_images = 10
    
    spec = spec_from_file_location(path.basename(model_path), model_path)
    model_builder = module_from_spec(spec)
    sys.modules[path.basename(model_path)] = model_builder
    spec.loader.exec_module(model_builder)

    # Resolve paths 
    weights_path = "../../../../../python/resnet/pretrained/model" # path.abspath(args.weights_path)
    image_path = "test_images" # path.abspath(args.image_path) if args.image_path else os.path.curdir
    os.makedirs(image_path, exist_ok=True)

    # Build model
    model = model_builder.build()
    model.load_weights(weights_path)

    # Sample images
    generate_images(image_path, num_images, folder_path)
    
    print(f"Accuracy: {test_network(model, image_path, folder_path)}%")

if __name__ == "__main__":
    main()
