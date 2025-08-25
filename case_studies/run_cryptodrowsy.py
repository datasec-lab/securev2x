from driverdrowsiness.cryptodrowsy import run

if __name__ == '__main__':
    run(
        device='cpu',
        w_path='driverdrowsiness/pretrained/sub1/model.pth',
        l_path='driverdrowsiness/dev_work/test_features_labels/1-drowsy_labels.pth',
        f_path='driverdrowsiness/dev_work/test_features_labels/1-drowsy_features.pth'
    )