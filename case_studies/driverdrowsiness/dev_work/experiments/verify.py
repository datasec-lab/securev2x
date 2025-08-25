def main():
    run = 7
    file2_name = "/home/jjl20011/snap/snapd-desktop-integration/current/Lab/Projects/Project1-V2X-Secure2PC/v2x-delphi-2pc/case_studies/driverdrowsiness/dev_work/experiments/Classification_Results{}.txt".format(run)
    file3_name = "/home/jjl20011/snap/snapd-desktop-integration/current/Lab/Projects/Project1-V2X-Secure2PC/v2x-delphi-2pc/case_studies/driverdrowsiness/dev_work/experiments/output9-crypten.txt"
    f1_name = "/home/jjl20011/snap/snapd-desktop-integration/current/Lab/Projects/Project1-V2X-Secure2PC/v2x-delphi-2pc/case_studies/driverdrowsiness/dev_work/experiments/output9.txt"
    f2_name = file2_name
    file_1 = []
    file_2 = []

    with open(f1_name, "r") as f:
        x = f.readlines()
        x = [item.strip("\n") for item in x]
        x = [item.split(" ") for item in x]
        for i in range(len(x)):
            x[i][0] = int(x[i][0])
            x[i][1] = int(x[i][1])
            x[i][2] = int(x[i][2])
        file_1 = x.copy()
    
    with open(f2_name, "r") as f:
        x = f.readlines()
        x = [item.strip("\n") for item in x]
        x = [item.split(" ") for item in x]
        for i in range(len(x)):
            x[i][0] = int(x[i][0])
            x[i][1] = int(x[i][1])
            x[i][2] = int(x[i][2])
        file_2 = x.copy()

    n = 0
    if len(file_1) > len(file_2):
        n = len(file_2)
    else: n = len(file_1)

    file_1_dict = dict()
    for i in range(len(file_1)):
        file_1_dict[file_1[i][0]] = file_1[i][1:]
    file_2_dict = dict()
    for i in range(len(file_2)):
        file_2_dict[file_2[i][0]] = file_2[i][1:]

    count = 0
    ones_count = 0
    zeros_count = 0
    
    for key in file_1_dict:
        if not key in file_2_dict:
            continue
        if not file_1_dict[key][1] == file_2_dict[key][1]:
            print("inference {}: py = {} de = {}".format(key,
                                                         file_1_dict[key][1], 
                                                         file_2_dict[key][1]))
            count += 1
            if key <= 156: zeros_count += 1
            elif key > 156: ones_count += 1

    # formulas 
    # sensitivity = tp/(tp+fn)
    # specificity = tn/(tn+fp)
    # ppp = tp/(tp+fp)
    # npp = tn/(tn+fn)
    # hitrate = (tp+tn)/(tp+tn+fp+fn)

    # get classification rates for the delphi model
    fp = 0
    tp = 0
    fn = 0
    tn = 0
    for key in file_2_dict:
        if file_2_dict[key][1] == 0:
            if file_2_dict[key][0] == 0:
                tn += 1
            else: fn += 1
        if file_2_dict[key][1] == 1:
            if file_2_dict[key][0] == 1:
                tp += 1
            else: fp += 1
    
    # get classification rates for python model
    fp1 = 0
    tp1 = 0
    fn1 = 0
    tn1 = 0
    for key in file_1_dict:
        if file_1_dict[key][1] == 0:
            if file_1_dict[key][0] == 0:
                tn1 += 1
            else: fn1 += 1
        if file_1_dict[key][1] == 1:
            if file_1_dict[key][0] == 1:
                tp1 += 1
            else: fp1 += 1
    
    print("delphi model ppp: {}".format(tp/(fp+tp)))
    print("python model ppp: {}".format(tp1/(fp1+tp1)))
    print("delphi model npp: {}".format(tn/(tn + fn)))
    print("python model npp: {}".format(tn1/(tn1 + fn1)))
    print("delphi acc: {}".format((tn + tp)/(tn + tp + fp + fn)))
    print("python acc: {}".format((tn1 + tp1)/(tn1 + tp1 + fp1 + fn1)))

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = (2 * precision * recall)/(precision + recall)

    print("delphi - f1 score: {}".format(f1))

    pyprec = tp1 / (tp1 + fp1)
    pyrecall = tp1 / (tp1 + fn1)
    pyf1 = (2 * pyprec * pyrecall) / (pyprec + pyrecall)

    print("plaintext f1 score: {}".format(pyf1))

    percent = count/n
    print("ones count: {}".format(ones_count))
    print("zeros count: {}".format(zeros_count))
    print("total count: {}".format(count))
    print("percent difference between inferences: {}".format(percent))

    # file_1 = sorted(file_1, key=lambda item: item[0])
    # file_2 = sorted(file_2, key=lambda item: item[0])

    # ones_count = 0
    # zeros_count = 0
    # count = 0
    # for i in range(n):
    #     assert file_1[i][0] == file_2[i][0]
    #     if not file_1[i][2] == file_2[i][2]:
    #         print("inference {}: py = {} de = {}".format(file_1[i][0],
    #                                                      file_1[i][2], 
    #                                                      file_2[i][2]))
    #         count += 1
    #         if file_1[i][0] <= 156: zeros_count += 1
    #         elif file_1[i][0] > 156: ones_count += 1
    # percent = count / n
    # print("ones count: {}".format(ones_count))
    # print("zeros count: {}".format(zeros_count))
    # print("total count: {}".format(count))
    # print("percent difference between inferences: {}".format(percent))

if __name__ == "__main__":
    main()
