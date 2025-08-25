use crate::*;
use neural_network::{ndarray::Array4, tensors::Input, NeuralArchitecture};
use rand::SeedableRng;
use rand_chacha::ChaChaRng;
use std::{cmp, fs, io::Write};

const RANDOMNESS: [u8; 32] = [
    0x11, 0xe0, 0x8f, 0xbc, 0x89, 0xa7, 0x34, 0x01, 0x45, 0x86, 0x82, 0xb6, 0x51, 0xda, 0xf4, 0x76,
    0x5d, 0xc9, 0x8d, 0xea, 0x23, 0xf2, 0x90, 0x8f, 0x9d, 0x03, 0xf2, 0x77, 0xd3, 0x4a, 0x52, 0xd2,
];

pub fn softmax(x: &Input<TenBitExpFP>) -> Input<TenBitExpFP> {
    let mut max: TenBitExpFP = x[[0, 0, 0, 0]];
    x.iter().for_each(|e| {
        max = match max.cmp(e) {
            // if max is less than e, return e is the new max
            cmp::Ordering::Less => *e,
            // if max is greater than e, return max as the new max
            _ => max,
        };
    });
    let mut e_x: Input<TenBitExpFP> = x.clone();
    e_x.iter_mut().for_each(|e| {
        *e = f64::from(*e - max).exp().into();
    });
    let e_x_sum = 1.0 / f64::from(e_x.iter().fold(TenBitExpFP::zero(), |sum, val| sum + *val));
    e_x.iter_mut().for_each(|e| *e *= e_x_sum.into());
    return e_x;
}

// we will implement the batch normalization pre-processing step here
pub fn run(
    network: &NeuralNetwork<TenBitAS, TenBitExpFP>,
    architecture: &NeuralArchitecture<TenBitAS, TenBitExpFP>,
    image: &Array4<f64>,
    class: i64,
    results_file: &str,
    sample: i64,
) {
    let mut server_rng = ChaChaRng::from_seed(RANDOMNESS);
    let mut client_rng = ChaChaRng::from_seed(RANDOMNESS);
    let server_addr = "127.0.0.1:8001";
    // if we modify the dimensions of the client_output, can we perform the test on 314 samples simultaneously?
    let mut client_output = Output::zeros((1, 10, 0, 0));
    crossbeam::thread::scope(|s| {
        let server_output = s
            .spawn(|_| {
                nn_server(
                    &server_addr, 
                    &network, 
                    &mut server_rng,
                )
            });
        client_output = s
            .spawn(|_| {
                nn_client(
                    &server_addr,
                    &architecture,
                    (image.clone()).into(),
                    &mut client_rng,
                )
            })
            .join()
            .unwrap();
        // run .join() waits for thread execution to finish
        server_output.join().unwrap();
    })
    .unwrap();
    let sm = softmax(&client_output);
    let max = sm.iter().map(|e| f64::from(*e)).fold(0. / 0., f64::max);
    let index = sm.iter().position(|e| f64::from(*e) == max).unwrap() as i64;
    
    // write the output values to a separate ".txt" file that can be analyzed with python
    // later to obtain accuracy information, etc.
    if results_file != "None" && sample != -1 {
        let mut file_ref = fs::OpenOptions::new().append(true).open(results_file).expect("Unable to open file");
        
        //write results to text file in format [{class} {inference}\n]
        file_ref.write_all(&format!("{} {} {}\n" ,sample, class, index).as_bytes()).expect("write failed");
    }
    println!("Correct class is {}, inference result is {}", class, index);
}
