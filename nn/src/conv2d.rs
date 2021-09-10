#![allow(unused)]
use arrayfire::{dim4, randu};

use crate::{model::Model, tensor::Tensor};

// The shape of the weight is [H x W x in_channels x out_channels]
pub struct Conv2d {
    weight: Tensor,
    kernel_size: [u64; 2],
    out_channels: u64,
    in_channels: u64,
}

impl Conv2d {
    pub fn new(in_channels: u64, out_channels: u64, kernel_size: [u64; 2]) -> Self {
        Self {
            weight: Tensor::new(
                randu(dim4!(
                    kernel_size[0],
                    kernel_size[1],
                    in_channels,
                    out_channels
                )),
                None,
            ),
            kernel_size,
            out_channels,
            in_channels,
        }
    }
}

// impl Model for Conv2d {
//     fn forward(&mut self, data: &Tensor) -> &Tensor {
//          for each out dim  {
//              for each in dim {
//                  result += conv2d(data, weight[j][c])
//              }
//          }
//     }
// }
