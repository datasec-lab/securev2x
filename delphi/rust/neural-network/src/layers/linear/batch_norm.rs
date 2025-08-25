// Batch normalization (linear) layer for use with Delphi - CompactCNN

use crate::tensors::{Input, Kernel, Output};
use algebra::{fp_64::Fp64Parameters, FixedPoint, FixedPointParameters, FpParameters, PrimeField};
use num_traits::Zero;
use std::{
    marker::PhantomData,
    ops::{Add, Mul},
};
use tch::nn;

// Implementation only requires that we use the (trained) Gamma and Beta matrix
// of the model

// CompactCNN implements the normalization step while running the forward pass
// it doesn't save the values and use them again later.

// I will need to reimplement the code in the python model and have it so that the
// means are saved throughout the mini-batches which are run over the model
#[derive(Debug)]
pub struct BatchNormParams<F, C> {
    pub gammas: Kernel<C>,
    pub betas: Kernel<C>,
    // maybe I don't need the tch_config part
    pub tch_config: Option<nn::BatchNorm>,
    pub eval_method: crate::EvalMethod,
    _variable: PhantomData<F>,
}

unsafe impl<F, C> Send for BatchNormParams<F, C> {}
unsafe impl<F, C> Sync for BatchNormParams<F, C> {}

impl<F, C> BatchNormParams<F, C>
where
    F: Zero + Copy + Add<C, Output = F> + Mul<C, Output = F>,
    C: Copy + Into<F>,
{
    pub fn new(gamma: Kernel<C>, beta: Kernel<C>) -> Self {
        Self {
            gammas: gamma,
            betas: beta,
            tch_config: None,
            eval_method: crate::EvalMethod::Naive,
            _variable: PhantomData,
        }
    }

    pub fn batch_norm_naive(&self, input: &Input<F>, out: &mut Output<F>) {
        let (batch_size, in_channels, in_height, in_width) = input.dim();
        let out_dim = (batch_size, in_channels, in_height, in_width);
        assert_eq!(out.dim(), out_dim);

        for b in 0..batch_size {
            for c in 0..in_channels {
                for h in 0..in_height {
                    for w in 0..in_width {
                        let val = input[(b, c, h, w)];
                        let gamma = self.gammas[(c, 0, 0, 0)];
                        let beta = self.betas[(c, 0, 0, 0)];
                        // gamma and beta are just Kernels
                        out[(b, c, h, w)] = val * gamma + beta
                    }
                }
            }
        }
    }
}
