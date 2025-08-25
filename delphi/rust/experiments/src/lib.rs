use ::neural_network as nn;
extern crate num_cpus;
extern crate rayon;
use algebra::{fields::near_mersenne_64::F, FixedPoint, FixedPointParameters, Polynomial};
use bench_utils::*;
use io_utils::{counting::CountingIO, imux::IMuxSync};
use nn::{
    layers::{
        average_pooling::AvgPoolParams,
        convolution::{Conv2dParams, Padding},
        fully_connected::FullyConnectedParams,
        Layer, LayerDims, LinearLayer, NonLinearLayer,
        //batch_norm::BatchNormParams,
    },
    tensors::*,
    NeuralArchitecture, NeuralNetwork,
};
use protocols::{neural_network::NNProtocol, AdditiveShare};
use rand::{CryptoRng, Rng, RngCore};
use std::{
    io::{BufReader, BufWriter},
    net::{TcpListener, TcpStream},
    //ops::Add,
};

pub mod inference;
pub mod latency;
pub mod linear_only;
pub mod minionn;
pub mod compact_cnn;
pub mod mnist;
pub mod resnet32;
pub mod throughput;
pub mod validation;

/// Documentation for TenBitExpParams struct
pub struct TenBitExpParams {}

/// Does this work
impl FixedPointParameters for TenBitExpParams {
    type Field = F;
    const MANTISSA_CAPACITY: u8 = 3;
    const EXPONENT_CAPACITY: u8 = 8;
}

/// Alias for `FixedPoint<TenBitExpParams>`
type TenBitExpFP = FixedPoint<TenBitExpParams>;
/// Alias for `AdditiveShare<TenBitExpParams>`
type TenBitAS = AdditiveShare<TenBitExpParams>;

/// client_connect function
pub fn client_connect(
    addr: &str,
) -> (
    IMuxSync<CountingIO<BufReader<TcpStream>>>,
    IMuxSync<CountingIO<BufWriter<TcpStream>>>,
) {
    // TODO: Maybe change to rayon_num_threads
    let mut readers = Vec::with_capacity(16);
    let mut writers = Vec::with_capacity(16);
    for _ in 0..16 {
        let stream = TcpStream::connect(addr).unwrap();
        readers.push(CountingIO::new(BufReader::new(stream.try_clone().unwrap())));
        writers.push(CountingIO::new(BufWriter::new(stream)));
    }
    (IMuxSync::new(readers), IMuxSync::new(writers))
}

/// server_connect function
/// 
/// Arguments: 
/// - addr --- server address (string slice)
/// 
/// Returns:
/// - tuple --- `(ImuxSync reader, ImuxSync writer)`
/// 
/// **Functionality**
/// 
/// Listens for a signal on the input server address. Then, it generates
/// `readers` and `writers` vectors 
/// 
/// **More Information**
/// 
/// Readers and writers are initialized via the `IMuxSync::new()` function which 
/// takes as input a `Vec<CountingIO<BufReader<TcpStream>>>` object.
/// So, it takes a vector of `CountingIO<BufReader<TcpStream>>` objects and outputs a 
/// new `ImuxSync` object. 
/// The `CountingIO<BufReader<TcpStream>>` object is an object which keeps track of the 
/// number of bytes written. So (at this point) the ImuxSync object is a `vector` of `channels`
/// which contain byte counters for `BufReader<TcpStream>` objects
/// `BufReader` objects are more efficient for making small and repeated calls to the same
/// network stream or file, just as is needed in the case of Delphi. 
/// `TcpStream` is the data stream between a local and remote socket which follows the TCP protocol
pub fn server_connect(
    addr: &str,
) -> (
    IMuxSync<CountingIO<BufReader<TcpStream>>>,
    IMuxSync<CountingIO<BufWriter<TcpStream>>>,
) {
    let listener = TcpListener::bind(addr).unwrap();
    let mut incoming = listener.incoming();
    let mut readers = Vec::with_capacity(16);
    let mut writers = Vec::with_capacity(16);
    for _ in 0..16 {
        let stream = incoming.next().unwrap().unwrap();
        readers.push(CountingIO::new(BufReader::new(stream.try_clone().unwrap())));
        writers.push(CountingIO::new(BufWriter::new(stream)));
    }
    (IMuxSync::new(readers), IMuxSync::new(writers))
}

/// neural network client abstraction - called directly for inference
/// 
/// Arguments:
/// - `server_addr` --- a string slice `&str` representing the server addres to communicate with
/// - `architecture` --- a reference to a `NeuralArchitecture<T, U>` object
/// - `rng` --- a cryptographically secure random number generation core
/// 
/// Returns:
/// - `client_output: Input<TenBitExpFP>` --- an `Input<I>` object where
/// `I: FixedPoint<TenBitExpParams>`
/// 
/// **Functionality**
/// 
/// 1. Set up client `reader` and `writer` by running `client_connect(server_addr)`
/// 2. Run `NNProtocol::offline_client_protocol()` which returns an output value to 
/// the `client_output` variable (returned to caller) 
/// 
/// **More Information:**
/// 
/// `Input<I>` structs have several functions including the following which
/// are necessary for performing Delphi calculations
/// - Implement `Input<I>` where `I` must implement the `Share` trait
///     - `Input.share()` <= `I: Share` 
///     - `Input.share_with_randomness()` <= `I: Share`
/// - Do not require `I: Share`
///     - `Input.to_tensor()`
///     - `Input.to_repr()`
///     - `Input.from_tensor()`
///     - `Input.randomize_local_share()`
pub fn nn_client<R: RngCore + CryptoRng>(
    server_addr: &str,
    architecture: &NeuralArchitecture<TenBitAS, TenBitExpFP>,
    input: Input<TenBitExpFP>,
    rng: &mut R,
) -> Input<TenBitExpFP> {
    let (client_state, offline_read, offline_write) = {
        let (mut reader, mut writer) = client_connect(server_addr);
        (
            NNProtocol::offline_client_protocol(&mut reader, &mut writer, &architecture, rng)
                .unwrap(),
            reader.count(),
            writer.count(),
        )
    };

    let (client_output, online_read, online_write) = {
        let (mut reader, mut writer) = client_connect(server_addr);
        (
            NNProtocol::online_client_protocol(
                &mut reader,
                &mut writer,
                &input,
                &architecture,
                &client_state,
            )
            .unwrap(),
            reader.count(),
            writer.count(),
        )
    };
    add_to_trace!(|| "Offline Communication", || format!(
        "Read {} bytes\nWrote {} bytes",
        offline_read, offline_write
    ));
    add_to_trace!(|| "Online Communication", || format!(
        "Read {} bytes\nWrote {} bytes",
        online_read, online_write
    ));
    client_output
}

/// neural network server abstraction - called directly for inference
/// 
/// Arguments: 
/// - server_addr --- a string slice indicating the address of the server (party)
/// - nn --- (reference) an abstraction of the NeuralNetwork object with layers to be evaluated sequentially
/// - rng --- (reference) the core of a random number generator which is supposed to be cryptographically secure
/// 
/// Returns:
/// - None
/// 
/// **Functionality**
/// - sets up client-server connection by calling `server_connect()` (returns reader
/// and writer `ImuxSync` objects)
/// - runs the NNProtocol:offline_server_protocol
/// - runs the NNProtocol:online_server_protocol
pub fn nn_server<R: RngCore + CryptoRng>(
    server_addr: &str,
    nn: &NeuralNetwork<TenBitAS, TenBitExpFP>,
    rng: &mut R,
) {
    let (server_state, offline_read, offline_write) = {
        // define tuple (reader, writer) set the values as the expression below
        let (mut reader, mut writer) = server_connect(server_addr);
        (
            NNProtocol::offline_server_protocol(&mut reader, &mut writer, &nn, rng).unwrap(),
            reader.count(),
            writer.count(),
        )
    };

    let (_, online_read, online_write) = {
        let (mut reader, mut writer) = server_connect(server_addr);
        (
            NNProtocol::online_server_protocol(&mut reader, &mut writer, &nn, &server_state)
                .unwrap(),
            reader.count(),
            writer.count(),
        )
    };
    add_to_trace!(|| "Offline Communication", || format!(
        "Read {} bytes\nWrote {} bytes",
        offline_read, offline_write
    ));
    add_to_trace!(|| "Online Communication", || format!(
        "Read {} bytes\nWrote {} bytes",
        online_read, online_write
    ));
}

/// generates a random number which is either positive or negative
/// 
/// Arguments: 
/// - rng --- random number core which only needs to implement the `Rng` trait
/// but will likely also implement the `CryptoRng` trait. Is a mutable reference.
/// 
/// Returns:
/// - tuple --- `(f, n)` 
/// `f` is a `f64` type which is the product of some random value (-1.0 or 1.0)
/// `n` is the value `f` converted into a `tenBitExpFP` or 
/// `FixedPoint<TenBitExpParams>` object (they are the same
/// For the purposes of Delphi, we almost exclusively use the `n` value returned,
/// or `generate_random_number().1`
///  
pub fn generate_random_number<R: Rng>(rng: &mut R) -> (f64, TenBitExpFP) {
    let is_neg: bool = rng.gen();
    let mul = if is_neg { -1.0 } else { 1.0 };
    let float: f64 = rng.gen();
    let f = TenBitExpFP::truncate_float(float * mul);
    let n = TenBitExpFP::from(f);
    (f, n)
}

fn sample_conv_layer<R: RngCore + CryptoRng>(
    vs: Option<&tch::nn::Path>,
    input_dims: (usize, usize, usize, usize),
    kernel_dims: (usize, usize, usize, usize),
    stride: usize,
    padding: Padding,
    rng: &mut R,
) -> (
    LinearLayer<TenBitAS, TenBitExpFP>,
    LinearLayer<TenBitExpFP, TenBitExpFP>,
) {
    let mut kernel = Kernel::zeros(kernel_dims);
    let mut bias = Kernel::zeros((kernel_dims.0, 1, 1, 1));
    kernel
        .iter_mut()
        .for_each(|ker_i| *ker_i = generate_random_number(rng).1);
    bias.iter_mut()
        .for_each(|bias_i| *bias_i = generate_random_number(rng).1);
    let layer_params = match vs {
        Some(vs) => Conv2dParams::<TenBitAS, _>::new_with_gpu(
            vs,
            padding,
            stride,
            kernel.clone(),
            bias.clone(),
        ),
        None => Conv2dParams::<TenBitAS, _>::new(padding, stride, kernel.clone(), bias.clone()),
    };
    let output_dims = layer_params.calculate_output_size(input_dims);
    let layer_dims = LayerDims {
        input_dims,
        output_dims,
    };
    let layer = LinearLayer::Conv2d {
        dims: layer_dims,
        params: layer_params,
    };

    let pt_layer_params =
        Conv2dParams::<TenBitExpFP, _>::new(padding, stride, kernel.clone(), bias.clone());
    let pt_layer = LinearLayer::Conv2d {
        dims: layer_dims,
        params: pt_layer_params,
    };
    (layer, pt_layer)
}

fn sample_fc_layer<R: RngCore + CryptoRng>(
    vs: Option<&tch::nn::Path>,
    input_dims: (usize, usize, usize, usize),
    out_chn: usize,
    rng: &mut R,
) -> (
    LinearLayer<TenBitAS, TenBitExpFP>,
    LinearLayer<TenBitExpFP, TenBitExpFP>,
) {
    let weight_dims = (out_chn, input_dims.1, input_dims.2, input_dims.3);
    let mut weights = Kernel::zeros(weight_dims);
    weights
        .iter_mut()
        .for_each(|w_i| *w_i = generate_random_number(rng).1);

    let bias_dims = (out_chn, 1, 1, 1);
    let mut bias = Kernel::zeros(bias_dims);
    bias.iter_mut()
        .for_each(|w_i| *w_i = generate_random_number(rng).1);

    // This is for the pytorch model implementation
    let pt_weights = weights.clone();
    let pt_bias = bias.clone();
    let params = match vs {
        Some(vs) => FullyConnectedParams::new_with_gpu(vs, weights, bias),
        None => FullyConnectedParams::new(weights, bias),
    };
    let output_dims = params.calculate_output_size(input_dims);
    let dims = LayerDims {
        input_dims,
        output_dims,
    };
    let pt_params = FullyConnectedParams::new(pt_weights, pt_bias);
    let layer = LinearLayer::FullyConnected { dims, params };
    let pt_layer = LinearLayer::FullyConnected {
        dims,
        params: pt_params,
    };
    (layer, pt_layer)
}

/// Batch normalization layer function for use with Delphi system
/// 
/// Arguments 
/// - input_dims --- a `tuple` of size 4, each of type `usize`
/// - rng --- a cryptographically secure random core
/// 
/// Returns
/// - tuple --- `(layer, ptlayer)`
/// `layer` is the layer output computed using naive algorithms
/// `ptlayer` is the layer output computed using GPU optimzed 
/// `tch` algorithms
/// 
// fn sample_batch_layer<R: RngCore + CryptoRng>(
//     input_dims: (usize, usize, usize, usize),
//     // normalizer: f64, // need some type of float to compute the 
//     // training set mu/sigma for use with inference
//     rng: &mut R,
// ) -> (LinearLayer<TenBitAS, TenBitExpFP>, Option<LinearLayer<TenBitExpFP, TenBitExpFP>>)
// {
//     // this version of the code randomly initializes the weights for the gamma matrix
//     let gamma_dims = input_dims;
//     let mut gammas = Kernel::zeros(gamma_dims);
//     gammas
//         .iter_mut()
//         .for_each(|w_i| *w_i = generate_random_number(rng).1);
//     let bias_dims = input_dims;
//     let mut bias = Kernel::zeros(bias_dims);
//     bias.iter_mut()
//         .for_each(|w_i| *w_i = generate_random_number(rng).1);
//     let params = BatchNormParams::new(gammas, bias);
//     // output_dims are the exact same as the input dimensions of the layer
//     let output_dims = (input_dims.0, input_dims.1, input_dims.2, input_dims.3);
//     let dims = LayerDims {
//         input_dims,
//         output_dims,
//     };
//     let layer = LinearLayer::BatchNorm { dims, params };
//     // right now, I'm only returning the normal layer for batch normalization
//     (layer, None)
// }

#[allow(dead_code)]
fn sample_iden_layer(
    input_dims: (usize, usize, usize, usize),
) -> (
    LinearLayer<TenBitAS, TenBitExpFP>,
    LinearLayer<TenBitExpFP, TenBitExpFP>,
) {
    let output_dims = input_dims;
    let layer_dims = LayerDims {
        input_dims,
        output_dims,
    };
    let layer = LinearLayer::Identity { dims: layer_dims };
    let pt_layer = LinearLayer::Identity { dims: layer_dims };
    (layer, pt_layer)
}

#[allow(dead_code)]
fn sample_avg_pool_layer(
    input_dims: (usize, usize, usize, usize),
    (pool_h, pool_w): (usize, usize),
    stride: usize,
) -> LinearLayer<TenBitAS, TenBitExpFP> {
    let size = (pool_h * pool_w) as f64;
    let avg_pool_params = AvgPoolParams::new(pool_h, pool_w, stride, TenBitExpFP::from(1.0 / size));
    let pool_dims = LayerDims {
        input_dims,
        output_dims: avg_pool_params.calculate_output_size(input_dims),
    };

    LinearLayer::AvgPool {
        dims: pool_dims,
        params: avg_pool_params,
    }
}



/// non-linear activation function for relu
fn add_activation_layer(nn: &mut NeuralNetwork<TenBitAS, TenBitExpFP>, relu_layers: &[usize]) {
    let cur_input_dims = nn.layers.last().as_ref().unwrap().output_dimensions();
    let layer_dims = LayerDims {
        input_dims: cur_input_dims,
        output_dims: cur_input_dims,
    };
    let num_layers_so_far = nn.layers.len();
    let is_relu = relu_layers.contains(&num_layers_so_far);
    let layer = if is_relu {
        Layer::NLL(NonLinearLayer::ReLU(layer_dims))
    } else {
        let activation_poly_coefficients = vec![
            TenBitExpFP::from(0.2),
            TenBitExpFP::from(0.5),
            TenBitExpFP::from(0.2),
        ];
        let poly = Polynomial::new(activation_poly_coefficients);
        let poly_layer = NonLinearLayer::PolyApprox {
            dims: layer_dims,
            poly,
            _v: std::marker::PhantomData,
        };
        Layer::NLL(poly_layer)
    };
    nn.layers.push(layer);
}
