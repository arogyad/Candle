#![allow(dead_code)]
use arrayfire::{matmul, Array, MatProp};

pub fn linear(x: &Array<f64>, weight: &Array<f64>, bias: Option<&Array<f64>>) -> Array<f64> {
    if let Some(b) = bias {
        matmul(x, weight, MatProp::NONE, MatProp::TRANS) + b
    } else {
        matmul(x, weight, MatProp::NONE, MatProp::TRANS)
    }
}
