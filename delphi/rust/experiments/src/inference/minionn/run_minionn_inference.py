import os

def main():
    weights = "../../../../../python/minionn/pretrained/relu/model.npy"
    layers = 0
    os.system("cargo +nightly run --bin minionn-inference -- --weights {} --layers {}".format(weights, layers))

if __name__ == "__main__":
    main()