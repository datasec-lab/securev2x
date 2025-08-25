use crate::tensors::{Input, Kernel, Output};
use algebra::{fp_64::Fp64Parameters, FixedPoint, FixedPointParameters, FpParameters, PrimeField};
use num_traits::Zero;
use std::{
    marker::PhantomData,
    ops::{Mul, Add},
};
use tch::nn;

#[derive(Debug)]
pub struct BatchNormParams<F, C> {
    pub gammas: Kernel<C>,
    pub betas: Kernel<C>,
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
                        out[(b, c, h, w)] = val * gamma + beta;
                    }
                }
            }
        }
    }
}

impl<P: FixedPointParameters, I> BatchNormParams<I, FixedPoint<P>>
where
    I: Zero + Copy + Into<FixedPoint<P>> + Add<FixedPoint<P>, Output = I> + Mul<FixedPoint<P>, Output = I>,
    FixedPoint<P>: Into<I>,
    <P::Field as PrimeField>::Params: Fp64Parameters,
    P::Field: PrimeField<BigInt = <<P::Field as PrimeField>::Params as FpParameters>::BigInt>,
{
    pub fn new_with_gpu(
        vs: &nn::Path,
        gamma: Kernel<FixedPoint<P>>,
        beta: Kernel<FixedPoint<P>>,
    ) -> Self {
        let (out_channels, ..) = gamma.dim();
        let device = vs.device();
        let gamma_tensor = gamma.to_tensor().to_device(device);
        let beta_tensor = beta.to_tensor().to_device(device);
        let mut out = Self::new(gamma, beta);
        out.eval_method = crate::EvalMethod::TorchDevice(device);

        let batchnorm2d_cfg = nn::BatchNormConfig {
            ..Default::default()
        };
        let mut tch_config = nn::batch_norm2d(
            vs,
            out_channels as i64,
            batchnorm2d_cfg
        );

        tch_config.ws = gamma_tensor;
        tch_config.bs = beta_tensor;
        out.tch_config = Some(tch_config);
        out
    }
}
Reply