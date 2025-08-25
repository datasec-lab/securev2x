use clap::{App, Arg, ArgMatches};
use std::env;
use experiments::resnet32::construct_resnet_32;
use neural_network::{ndarray::Array4, npy::NpyData};
use rand::SeedableRng;
use rand_chacha::ChaChaRng;
use std::{io::Read, path::Path};

const RANDOMNESS: [u8; 32] = [
    0x11, 0xe0, 0x8f, 0xbc, 0x89, 0xa7, 0x34, 0x01, 0x45, 0x86, 0x82, 0xb6, 0x51, 0xda, 0xf4, 0x76,
    0x5d, 0xc9, 0x8d, 0xea, 0x23, 0xf2, 0x90, 0x8f, 0x9d, 0x03, 0xf2, 0x77, 0xd3, 0x4a, 0x52, 0xd2,
];

fn get_args() -> ArgMatches<'static> {
    App::new("resnet-sequential-inference")
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
            Arg::with_name("images")
                .short("images")
                .long("images")
                .takes_value(true)
                .help("Path to image test data")
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

fn main() {
    env::set_var("RUST_BACKTRACE", "1");
    let args = get_args();
    let weights_path = args.value_of("weights").unwrap();
    let num_layers = clap::value_t!(args.value_of("layers"), usize).unwrap();
    let data_path = args.value_of("images").unwrap();
    let num_samples = clap::value_t!(args.value_of("num_samples"), usize).unwrap();
    let mut rng = ChaChaRng::from_seed(RANDOMNESS);
    let results_file = args.value_of("results_file").unwrap();

    let data_dir = Path::new(&data_path);
    let info_path = Path::new("predictions");

    let mut network = construct_resnet_32(None, 1, num_layers, &mut rng);
    let architecture = (&network).into();

    // load network weights
    network.from_numpy(&weights_path).unwrap();
    println!("{:?}", network);

    let mut buf = vec![];
    std::fs::File::open(info_path.join(Path::new("classes.npy")))
        .unwrap()
        .read_to_end(&mut buf)
        .unwrap();
    let classes: Vec<i64> = NpyData::from_bytes(&buf).unwrap().to_vec();

    buf = vec![];
    std::fs::File::open(info_path.join(Path::new("plaintext.npy")))
        .unwrap()
        .read_to_end(&mut buf)
        .unwrap();
    // let plaintext: Vec<i64> = NpyData::from_bytes(&buf).unwrap().to_vec();

    let mut image_data: Vec<Array4<f64>> = Vec::new();
    for i in 0..classes.len() {
        buf = vec![];
        std::fs::File::open(data_dir.join(Path::new(&format!("image_{}.npy", i))))
            .unwrap()
            .read_to_end(&mut buf)
            .unwrap();
        let image_vec: Vec<f64> = NpyData::from_bytes(&buf).unwrap().to_vec();
        let input = Array4::from_shape_vec((1,3,32,32), image_vec).unwrap();
        image_data.push(input);
    }       
    
    for i in 0..10 {
        experiments::inference::inference::run(&network, &architecture, &image_data[i], classes[i], results_file, i as i64);
    }
}