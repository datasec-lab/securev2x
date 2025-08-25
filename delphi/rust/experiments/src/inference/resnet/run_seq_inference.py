import os

def main():
    weights_path = "../../../../../python/resnet/pretrained/model.npy"
    num_layers = 0
    data_path = "test_images"
    num_samples = 10
    results_file = "predictions/del_predict.txt"
    os.system("cargo +nightly run --bin resnet-sequential-inference -- --weights {} --layers {} --images {} --num_samples {} --results_file {} > debug_output.txt".format(weights_path, num_layers, data_path, num_samples, results_file))

if __name__ == "__main__":
    main()