use clap::{App, Arg, ArgMatches};
use experiments::compact_cnn::construct_compact_cnn;
use neural_network::{ndarray::Array4, npy::NpyData};
use rand::SeedableRng;
use rand_chacha::ChaChaRng;
use std::{io::Read, path::Path};

const RANDOMNESS: [u8; 32] = [
    0x11, 0xe0, 0x8f, 0xbc, 0x89, 0xa7, 0x34, 0x01, 0x45, 0x86, 0x82, 0xb6, 0x51, 0xda, 0xf4, 0x76,
    0x5d, 0xc9, 0x8d, 0xea, 0x23, 0xf2, 0x90, 0x8f, 0x9d, 0x03, 0xf2, 0x77, 0xd3, 0x4a, 0x52, 0xd2,
];

/// `get_args` handles the fetching of arguments from command line
/// input from the user
/// 
/// Arguments: 
/// - None
/// 
/// Returns:
/// - `App::new("compact-cnn")`: `ArgMatches<'static>`
///     - contains the user arguments passed to the program at
///     run time
fn get_args() -> ArgMatches<'static> {
    App::new("compact-cnn-accuracy")
        .arg(
            Arg::with_name("weights")
                .short("w")
                .long("weights")
                .takes_value(true)
                .help("path to weights")
                .required(true),
        )
        .arg(
            Arg::with_name("eeg_data")
                .short("i")
                .long("eeg_data")
                .takes_value(true)
                .help("Path to test eeg_data")
                .required(true),
        )
        .arg(
            Arg::with_name("layers")
                .short("l")
                .long("layers")
                .takes_value(true)
                .help("Number of approximation layers to use")
                .required(true)
        )
        .get_matches()
}

fn main() {
    let mut rng = ChaChaRng::from_seed(RANDOMNESS);
    let args = get_args();
    let weights = args.value_of("weights").unwrap();
    let eeg_data = args.value_of("eeg_data").unwrap();
    let layers = clap::value_t!(args.value_of("layers"), usize).unwrap();

    // Build network
    // there are 314 samples in the test batch for compact_cnn
    let mut network = construct_compact_cnn(None, 1, layers, &mut rng);
    let architecture = (&network).into();

    // load network weights
    network.from_numpy(&weights).unwrap();

    // open all eeg_data, classes, and classification results
    let data_dir = Path::new(&eeg_data);

    // the required files are generated when you run `generate_images.py`
    let mut buf = vec![];
    std::fs::File::open(data_dir.join(Path::new("classes.npy")))
        .unwrap()
        .read_to_end(&mut buf)
        .unwrap();
    let classes: Vec<i64> = NpyData::from_bytes(&buf).unwrap().to_vec();

    buf = vec![];
    std::fs::File::open(data_dir.join(Path::new("plaintext.npy")))
        .unwrap()
        .read_to_end(&mut buf)
        .unwrap();
    let plaintext: Vec<i64> = NpyData::from_bytes(&buf).unwrap().to_vec();

    let mut eeg_data: Vec<Array4<f64>> = Vec::new();
    for i in 0..classes.len() {
        buf = vec![];
        std::fs::File::open(data_dir.join(Path::new(&format!("eeg_sample_{}.npy", i))))
            .unwrap()
            .read_to_end(&mut buf)
            .unwrap();
        let eeg_vec: Vec<f64> = NpyData::from_bytes(&buf).unwrap().to_vec();
        let input = Array4::from_shape_vec((1, 1, 1, 384), eeg_vec).unwrap();
        eeg_data.push(input);
    }
    experiments::validation::validate::run(network, architecture, eeg_data, classes, plaintext);
}

// Run the following command to run this file (from `/delphi/rust/experiments`)
// cargo +nightly run --bin compact-cnn-accuracy -- --weights /home/jjl20011/snap/snapd-desktop-integration/current/Lab/V2V-Delphi-Applications/python/pretrained_model_weights/modified_numpy_models/model_subj_9.npy --eeg_data /home/jjl20011/snap/snapd-desktop-integration/current/Lab/V2V-Delphi-Applications/delphi/rust/experiments/src/validation/Eeg_Samples_and_Validation --layers 0


// RUN MODEL WITH NO APPROX RELU
// cargo +nightly run --bin compact-cnn-accuracy -- --weights /home/jjl20011/snap/snapd-desktop-integration/current/Lab/V2V-Delphi-Applications/python/pretrained_model_weights/modified_numpy_models/modified_numpy_models_no_approx/model_subj_9.npy --layers 0 --eeg_data /home/jjl20011/snap/snapd-desktop-integration/current/Lab/V2V-Delphi-Applications/delphi/rust/experiments/src/validation/Eeg_Samples_and_Validation

// RUN MODEL WITH APPROX RELU
// 