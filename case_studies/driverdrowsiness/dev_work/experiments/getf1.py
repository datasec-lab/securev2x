def main():
    with open("output9-crypten.txt", 'r') as f:
        x = f.readlines()
        for i in range(len(x)):
            x[i] = x[i].strip("\n").split(" ")
            for j in range(len(x[i])):
                x[i][j] = int(x[i][j])
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for out in x: 
        if out[1] == out[2] and out[1]==1:
            tp += 1
        elif out[1] != out[2] and out[1]==1:
            fn += 1
        elif out[1] == out[2] and out[1]==0:
            tn += 1
        elif out[1] != out[2] and out[1]==0:
            fp += 1
    precision = tp/(tp + fp)
    recall = tp/(tp + fn)
    f1 = (2*precision*recall)/(precision+recall)
    print("[INFO]: F1 = {}".format(f1))

if __name__ == "__main__":
    main()
