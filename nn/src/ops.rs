#![allow(unused)]
use super::tensor::Tensor;
use arrayfire::{matmul, mul, Array};
use std::{
    borrow::Borrow,
    cell::{Ref, RefCell},
};

// Function Trait Defns
pub trait Function {
    fn apply(&self) -> Tensor;
    fn backward(&self, grad: Ref<Array<f64>>) -> [Array<f64>; 2];
    fn forward(&self) -> Array<f64>;
    fn parents(&self) -> [&Tensor; 2];
}

// Basic Operations
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
        Tensor::new(self.forward(), Some(self))
    }

    fn forward(&self) -> Array<f64> {
        &self.parents[0].data + &self.parents[1].data
    }

    fn backward(&self, grad: Ref<Array<f64>>) -> [Array<f64>; 2] {
        [grad.copy(), grad.copy()]
    }

    fn parents(&self) -> [&Tensor; 2] {
        self.parents
    }
}

pub struct Sub<'a> {
    parents: [&'a Tensor<'a>; 2],
}

impl<'a> Sub<'a> {
    pub fn new(parents: [&'a Tensor; 2]) -> Self {
        Self { parents }
    }
}

impl<'a> Function for Sub<'a> {
    fn apply(&self) -> Tensor {
        Tensor::new(self.forward(), Some(self))
    }

    fn forward(&self) -> Array<f64> {
        &self.parents[0].data - &self.parents[1].data
    }

    fn backward(&self, grad: Ref<Array<f64>>) -> [Array<f64>; 2] {
        [grad.copy(), -grad.copy()]
    }

    fn parents(&self) -> [&Tensor; 2] {
        self.parents
    }
}

pub struct Mul<'a> {
    parents: [&'a Tensor<'a>; 2],
}

impl<'a> Mul<'a> {
    pub fn new(parents: [&'a Tensor; 2]) -> Self {
        Self { parents }
    }
}

impl<'a> Function for Mul<'a> {
    fn apply(&self) -> Tensor {
        Tensor::new(self.forward(), Some(self))
    }

    fn forward(&self) -> Array<f64> {
        &self.parents[0].data * &self.parents[1].data
    }

    fn backward(&self, grad: Ref<Array<f64>>) -> [Array<f64>; 2] {
        [
            &self.parents[1].data * grad.copy(),
            &self.parents[0].data * grad.copy(),
        ]
    }

    fn parents(&self) -> [&Tensor; 2] {
        self.parents
    }
}

// TODO: Implement Power
pub struct Pow<'a> {
    parents: [&'a Tensor<'a>; 2],
}

impl<'a> Pow<'a> {
    pub fn new(parents: [&'a Tensor; 2]) -> Self {
        Self { parents }
    }
}

// Other Operations
pub struct MatMul<'a> {
    parents: [&'a Tensor<'a>; 2],
}

impl<'a> MatMul<'a> {
    pub fn new(parents: [&'a Tensor; 2]) -> Self {
        Self { parents }
    }
}

impl<'a> Function for MatMul<'a> {
    fn apply(&self) -> Tensor {
        Tensor::new(self.forward(), Some(self))
    }

    fn forward(&self) -> Array<f64> {
        matmul(
            &self.parents[0].data,
            &self.parents[1].data,
            arrayfire::MatProp::NONE,
            arrayfire::MatProp::TRANS,
        )
    }

    fn backward(&self, grad: Ref<Array<f64>>) -> [Array<f64>; 2] {
        [
            matmul(
                grad.borrow(),
                &self.parents[1].data,
                arrayfire::MatProp::NONE,
                arrayfire::MatProp::TRANS,
            ),
            matmul(
                &self.parents[0].data,
                grad.borrow(),
                arrayfire::MatProp::TRANS,
                arrayfire::MatProp::NONE,
            ),
        ]
    }

    fn parents(&self) -> [&Tensor; 2] {
        self.parents
    }
}
// TODO: Add conv2d operation
