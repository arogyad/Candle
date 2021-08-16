#![allow(unused)]
use super::tensor::Tensor;
use arrayfire::Array;
use std::cell::RefCell;

// Function Trait Defns
pub trait Function {
    fn apply(&self) -> Tensor;
    fn backward<'b>(&self, grad: &'b RefCell<Array<f64>>) -> [&'b RefCell<Array<f64>>; 2];
    fn forward(&self) -> Array<f64>;
    fn parents(&self) -> [&Tensor; 2];
}

// Binary Operations
pub struct Add<'a> {
    parents: [&'a Tensor<'a>; 2],
}

impl<'a> Add<'a> {
    pub fn new(parents: [&'a Tensor; 2]) -> Self {
        Self { parents }
    }
}

impl<'a> Function for Add<'a> {
    fn apply(&self) -> Tensor {
        Tensor::new(self.forward(), Some(self), false)
    }

    fn forward(&self) -> Array<f64> {
        &self.parents[0].data + &self.parents[1].data
    }

    fn backward<'b>(&self, grad: &'b RefCell<Array<f64>>) -> [&'b RefCell<Array<f64>>; 2] {
        [grad, grad]
    }

    fn parents(&self) -> [&Tensor; 2] {
        self.parents
    }
}
