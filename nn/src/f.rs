#![allow(dead_code)]
use super::tensor::Tensor;
use arrayfire::{convolve2_nn, matmul, Array, Dim4, MatProp};

/// Calculate y = xW.t() + b from the given inputs
pub fn linear(x: &Array<f64>, weight: &Array<f64>, bias: &Option<Tensor>) -> Array<f64> {
    if let Some(b) = bias {
        matmul(x, weight, MatProp::NONE, MatProp::TRANS) + &b.data
    } else {
        matmul(x, weight, MatProp::NONE, MatProp::TRANS)
    }
}

pub fn conv2(
    input: &Array<f64>,
    filter: &Array<f64>,
    strides: Dim4,
    padding: Dim4,
    dilation: Dim4,
) -> Array<f64> {
    convolve2_nn(input, filter, strides, padding, dilation)
}
