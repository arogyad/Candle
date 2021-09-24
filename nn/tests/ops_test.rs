#![allow(unused)]
use arrayfire::{
    constant, convolve2_gradient_nn, convolve2_nn, dim4, print, Array, ConvGradientType,
};
use nn::conv2d::Conv2d;
use nn::ops::MatMul;
use nn::tensor::Tensor;
#[test]
fn matmul() {
    let a = Tensor::new(constant(1., dim4!(3, 3, 1, 1)), None);
    let b = Tensor::new(constant(2., dim4!(3, 3, 1, 1)), None);
    let c = MatMul::apply(&a, &b);
    c.backward();
}
