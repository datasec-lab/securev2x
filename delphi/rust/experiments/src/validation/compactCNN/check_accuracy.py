def main():
    run = "7"
    file = "Eeg_Samples_and_Validation/Classification_Results{}.txt".format(run)
    x = []
    with open(file, 'r') as f:
        x = f.readlines()
        for i in range(len(x)):
            x[i] = x[i].split()
            for j in range(len(x[i])):
                x[i][j] = int(x[i][j])
    unique = dict()
    for i in range(len(x)):
        if not x[i][0] in unique:
            unique[x[i][0]]=(x[i])
    correct = 0
    total = 0
    for key in unique:
        if unique[key][1] == unique[key][2]:
            correct += 1
        total += 1
    
    print(f"correct  = {correct}")
    print(f"total    = {total}")
    print(f"accuracy = {correct / total}")

if __name__ == "__main__":
    main()

