if __name__ == "__main__": 
    test_eeg_parameters_txt = []
    with open("gen_test_eeg_py_parameters.txt", 'r') as f:
        test_eeg_parameters_txt = f.readlines()

    eval_model_parameters_txt = []
    with open("eval_model_parameters.txt", 'r') as f:
        eval_model_parameters_txt = f.readlines()

    # if len(eval_model_parameters_txt) != len(test_eeg_parameters_txt): 
    #     print("these files are not the same length")
    #     print(len(eval_model_parameters_txt),"eval_model")
    #     print(len(test_eeg_parameters_txt),"eeg_gen")
    for i in range(len(eval_model_parameters_txt)):
        if eval_model_parameters_txt[i] != test_eeg_parameters_txt[i]:
            print(f"There was a difference on line {i}")
            print(f"eeg_gen: {test_eeg_parameters_txt[i]}")
            print(f"eval_model: {eval_model_parameters_txt[i]}")

    # no difference found at all (anywhere in the file)