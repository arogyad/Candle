#![allow(unused)]
use crate::model::Model;
use crate::{ops::MatMul, tensor::Tensor};
use arrayfire::{constant, dim4, matmul, print, randu, transpose, Array};

pub struct Linear {
    pub weight: Tensor,
    pub saved: Tensor,
    pub bias: Option<Tensor>,
}

impl Linear {
    pub fn new(in_channels: u64, out_channels: u64, bias: bool) -> Self {
        if bias {
            // Dimension in In Channels * Out Channels * 1 * 1 (Batch Size not yet implemented)
            Self {
                weight: Tensor::new(randu(dim4!(out_channels, in_channels, 1, 1)), None),
                bias: Some(Tensor::new(randu(dim4!(1, out_channels, 1, 1)), None)),
                saved: Tensor::new(constant(1., dim4!(1, 1, 1, 1)), None),
            }
        } else {
            Self {
                weight: Tensor::new(randu(dim4!(out_channels, in_channels, 1, 1)), None),
                bias: None,
                saved: Tensor::new(constant(1., dim4!(1, 1, 1, 1)), None),
            }
        }
    }
}

impl Model for Linear {
    fn forward(&mut self, data: &Tensor) -> &Tensor {
        let t1 = MatMul::apply(data, &self.weight);
        if let Some(n) = &self.bias {
            let t2 = &t1 + n;
            self.saved = t2;
        } else {
            self.saved = t1;
        }
        &self.saved
    }

    fn backward(&self) {
        self.saved.backward();
    }
}
