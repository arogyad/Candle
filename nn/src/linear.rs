#![allow(dead_code)]
#![allow(unused)]
use super::tensor::Tensor;
use arrayfire::{Array, Dim4};

pub struct Linear<'a> {
    pub weight: Tensor<'a>,
    pub bias: Option<Tensor<'a>>,
}

impl<'a> Linear<'a> {
    pub fn new(in_feat: u64, out_feat: u64, bias: bool) -> Self {
        if bias {
            Linear {
                weight: Tensor::randn(Dim4::new(&[1, 1, out_feat, in_feat])),
                bias: Some(Tensor::randn(Dim4::new(&[1, 1, 1, out_feat]))),
            }
        } else {
            Linear {
                weight: Tensor::randn(Dim4::new(&[1, 1, out_feat, in_feat])),
                bias: None,
            }
        }
    }
    // TODO
    pub fn forward(&self, data: &Tensor) {}
}
