#![allow(unused)]
use crate::model::Model;
use crate::{ops::MatMul, tensor::Tensor};
use arrayfire::{dim4, matmul, print, randu, Array};

pub struct Linear {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
}

impl Linear {
    pub fn new(in_channels: u64, out_channels: u64, bias: bool) -> Self {
        if bias {
            // Dimension in In Channels * Out Channels * 1 * 1 (Batch Size not yet implemented)
            Self {
                weight: Tensor::new(randu(dim4!(out_channels, in_channels, 1, 1)), None),
                bias: Some(Tensor::new(randu(dim4!(out_channels, 1, 1, 1)), None)),
            }
        } else {
            Self {
                weight: Tensor::new(randu(dim4!(out_channels, in_channels, 1, 1)), None),
                bias: None,
            }
        }
    }
}

impl Model for Linear {
    fn forward(&self, data: &Tensor) -> Tensor {
        if let Some(n) = &self.bias {
            // Doesn't work rn!
            let t1 = MatMul::apply(data, &self.weight);
            let t2 = &t1 + n;
            t1
        } else {
            let t1 = MatMul::apply(data, &self.weight);
            t1
        }
    }

    fn backward(&self) {
        self.weight.backward();
        if let Some(n) = &self.bias {
            n.backward();
        }
    }
}
