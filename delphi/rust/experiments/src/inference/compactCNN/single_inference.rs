// clap is a rust framework which allows you to easily
// implement argument parsing for your program

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

/// Parses the arguments from the command line in delphi's style
/// 
/// `cargo +nightly run --bin compact-cnn-inference -- --weights folder/numpy_model.npy --layers num_approx_relus`
fn get_args() -> ArgMatches<'static> {
    App::new("compact-cnn-inference")
        .arg(
            Arg::with_name("weights")
                .short("w")
                .long("weights")
                .takes_value(true)
                .help("Path to weights")
                .required(true),
        )
        .arg(
            Arg::with_name("layers")
                .short("l")
                .long("layers")
                .takes_value(true)
                .help("Number of polynomial layers (0-7)")
                .required(true),
        )
        .get_matches()
}

/// `main()` runs the basic inference for CompactCNN with inputs taken from `get_args()`
/// 
/// - rng --> random value used for masking in security protocols
/// - args --> gets arguments from cl
/// - weights --> saved weights generated from the .npy model passed to inference
/// - layers --> **DON'T KNOW** just specifies num relu approximations
fn main() {
    let mut rng = ChaChaRng::from_seed(RANDOMNESS);
    let args = get_args();
    let weights = args.value_of("weights").unwrap();
    let layers = clap::value_t!(args.value_of("layers"), usize).unwrap();

    // adjust the second parameter of the construct_compact_cnn() method to change the batch_size input
    let mut network = construct_compact_cnn(None, 1, layers, &mut rng);
    let architecture = (&network).into();

    // load network weights
    network.from_numpy(&weights).unwrap();

    // Open EEG data and class
    let mut buf = vec![];
    std::fs::File::open(Path::new("eeg_class.npy"))
        .unwrap()
        .read_to_end(&mut buf)
        .unwrap();
    let class: i64 = NpyData::from_bytes(&buf).unwrap().to_vec()[0];

    buf = vec![];
    std::fs::File::open(Path::new("eeg_data.npy"))
        .unwrap()
        .read_to_end(&mut buf)
        .unwrap();
    let eeg_vec: Vec<f64> = NpyData::from_bytes(&buf).unwrap().to_vec();
    let eeg_data = Array4::from_shape_vec((1, 1, 1, 384), eeg_vec).unwrap();

    println!("Finished setup, running inference function");

    experiments::inference::inference::run(&network, &architecture, &eeg_data, class, "None", -1)
}

// run the following command to run the compact cnn inference on the 
// eeg data
// 0 relu approximation layers are used

// file for the modified conv_fold and bias_fold modified parameter sets: '/home/jjl20011/snap/snapd-desktop-integration/current/Lab/V2V-Delphi-Applications/python/pretrained_model_weights/modified_numpy_models/model_subj_9.npy'
// cargo +nightly run --bin compact-cnn-inference -- --weights /home/jjl20011/snap/snapd-desktop-integration/current/Lab/V2V-Delphi-Applications/python/no_batch-norm_experiments/compact_cnn_no_batch_norm_seed0_subj9.npy --layers 0