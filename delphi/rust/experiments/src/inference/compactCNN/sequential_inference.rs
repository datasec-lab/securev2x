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
/// `cargo +nightly run --bin compact-cnn-sequential-inference -- --weights folder/numpy_model.npy --layers num_approx_relus --eeg_data eeg_dat_folder --num_samples 314 --results_file abs_file_path`
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
        .arg(
            Arg::with_name("eeg_data")
                .short("eeg")
                .long("eeg_data")
                .takes_value(true)
                .help("Path to eeg test data")
                .required(true),
        )
        .arg(
            Arg::with_name("num_samples")
                .short("n")
                .long("num_samples")
                .takes_value(true)
                .help("Number of samples to run")
                .required(true),
        )
        .arg(
            Arg::with_name("results_file")
                .short("r")
                .long("results_file")
                .takes_value(true)
                .help("Where the results of sequential inference will be stored")
                .required(true),
        )
        .get_matches()
}

/// Runs a single inference of CompactCNN on test data using Delphi's system
/// 
/// Arguments
/// - class --- i64 value, indicating the class of the corresponding test sample
/// - eeg_data --- Vector of `f64` values containing the test sample data
//fn single_inference(class:i64, eeg_vec:Vec<f64>, layers:usize, weights:&str) {
    // let args = get_args();
    // let weights = args.value_of("weights").unwrap();
    // let layers = clap::value_t!(args.value_of("layers"), usize).unwrap();

    // adjust the second parameter of the construct_compact_cnn() method to change the batch_size input
    // let mut network = construct_compact_cnn(None, 1, layers, &mut rng);
    // let architecture = (&network).into();

    // // load network weights
    // network.from_numpy(&weights).unwrap();

    // Open EEG data and class
    // let mut buf = vec![];
    // std::fs::File::open(Path::new("eeg_class.npy"))
    //     .unwrap()
    //     .read_to_end(&mut buf)
    //     .unwrap();
    // let class: i64 = NpyData::from_bytes(&buf).unwrap().to_vec()[0];

    // buf = vec![];
    // std::fs::File::open(Path::new("eeg_data.npy"))
    //     .unwrap()
    //     .read_to_end(&mut buf)
    //     .unwrap();
    // let eeg_vec: Vec<f64> = NpyData::from_bytes(&buf).unwrap().to_vec();
//     let eeg_data = Array4::from_shape_vec((1, 1, 1, 384), eeg_vec).unwrap();

//     println!("Finished setup, running inference function");

//     experiments::inference::inference::run(network, architecture, eeg_data, class)
// }

/// Run the validation / accuracy steps for Delphi iteratively instead of
/// as a single batch of inferences. For updates to the code, run as a single
/// batch - significantly improve efficiency of the validation run
/// 
/// Arguments:
/// - None
/// 
/// Returns: 
/// - None
/// 
/// **Functionality**
fn main() {
    // get the input arguments 
    let args = get_args();
    let weights_path = args.value_of("weights").unwrap();
    let num_layers = clap::value_t!(args.value_of("layers"), usize).unwrap();
    let data_path = args.value_of("eeg_data").unwrap();
    let num_samples = clap::value_t!(args.value_of("num_samples"), usize).unwrap();
    let mut rng = ChaChaRng::from_seed(RANDOMNESS);
    let results_file = args.value_of("results_file").unwrap();

    let data_dir = Path::new(&data_path);

    let mut network = construct_compact_cnn(None, 1, num_layers, &mut rng);
    let architecture = (&network).into();

    // load network weights
    network.from_numpy(&weights_path).unwrap();

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
    // let plaintext: Vec<i64> = NpyData::from_bytes(&buf).unwrap().to_vec();

    let mut eeg_data: Vec<Array4<f64>> = Vec::new();
    for i in 0..num_samples {
        buf = vec![];
        std::fs::File::open(data_dir.join(Path::new(&format!("eeg_sample_{}.npy", i))))
            .unwrap()
            .read_to_end(&mut buf)
            .unwrap();
        let eeg_vec: Vec<f64> = NpyData::from_bytes(&buf).unwrap().to_vec();
        let input = Array4::from_shape_vec((1,1,1,384), eeg_vec).unwrap();
        eeg_data.push(input);
    }       
    
    for i in 77..157 {
        experiments::inference::inference::run(&network, &architecture, &eeg_data[i], classes[i], results_file, i as i64);
    }
}

// start a new screen to run the inference using the following commands
// `screen -S session_name`
// To kill a session use the command
// `screen -X -S session_name quit` 
// if you have multiple screens running with the same name, you may need to specify the session
// process ID in front of the session_name
// use `screen -r session_name` to switch into the desired session

// `cargo +nightly run --bin compact-cnn-sequential-inference -- 
//     --weights /home/jjl20011/snap/snapd-desktop-integration/current/Lab/Projects/Project1-V2X-Secure2PC/v2x-delphi-2pc/case_studies/driverdrowsiness/pretrained/sub9/model.npy
//     --layers 0
//     --eeg_data /home/jjl20011/snap/snapd-desktop-integration/current/Lab/Projects/Project1-V2X-Secure2PC/v2x-delphi-2pc/delphi/rust/experiments/src/validation/Eeg_Samples_and_Validation
//     --num_samples (1 to 314) 
//     --results_file /home/jjl20011/snap/snapd-desktop-integration/current/Lab/Projects/Project1-V2X-Secure2PC/v2x-delphi-2pc/delphi/rust/experiments/src/validation/Eeg_Samples_and_Validation/Classification_Results.txt

// This is written in a single line as follows
//  cargo +nightly run --bin compact-cnn-sequential-inference -- --weights /home/jjl20011/snap/snapd-desktop-integration/current/Lab/Projects/Project1-V2X-Secure2PC/v2x-delphi-2pc/case_studies/driverdrowsiness/pretrained/sub9/model.npy --layers 0 --eeg_data /home/jjl20011/snap/snapd-desktop-integration/current/Lab/Projects/Project1-V2X-Secure2PC/v2x-delphi-2pc/delphi/rust/experiments/src/validation/Eeg_Samples_and_Validation --num_samples 3 --results_file /home/jjl20011/snap/snapd-desktop-integration/current/Lab/Projects/Project1-V2X-Secure2PC/v2x-delphi-2pc/delphi/rust/experiments/src/validation/Eeg_Samples_and_Validation/Classification_Results.txt
