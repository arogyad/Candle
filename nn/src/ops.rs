#![allow(dead_code)]
#![allow(unused)]
use super::tensor::Tensor;
use arrayfire::{matmul, Array};
use std::{borrow::Borrow, cell::RefCell};
pub trait Function {
    fn parents(&self) -> &[Tensor; 2];
    fn backward(&self, grad: &RefCell<Array<f64>>) -> [Array<f64>; 2];
}

pub struct Add {
    parents: [Tensor; 2],
}

impl Add {
    pub fn apply(p1: Tensor, p2: Tensor) -> Tensor {
        Tensor::new(
            &p1.data + &p2.data,
            Some(Box::new(Self { parents: [p1, p2] })),
        )
    }
}

impl Function for Add {
    fn parents(&self) -> &[Tensor; 2] {
        &self.parents
    }

    fn backward(&self, grad: &RefCell<Array<f64>>) -> [Array<f64>; 2] {
        // TODO: Remove Various Clones
        [grad.borrow().clone(), grad.borrow().clone()]
    }
}

pub struct Mul {
    parents: [Tensor; 2],
}

impl Mul {
    pub fn apply(p1: Tensor, p2: Tensor) -> Tensor {
        Tensor::new(
            &p1.data * &p2.data,
            Some(Box::new(Self { parents: [p1, p2] })),
        )
    }
}

impl Function for Mul {
    fn parents(&self) -> &[Tensor; 2] {
        &self.parents
    }

    fn backward(&self, grad: &RefCell<Array<f64>>) -> [Array<f64>; 2] {
        unsafe {
            [
                arrayfire::mul(&*grad.as_ptr(), &self.parents[1].data, false),
                arrayfire::mul(&*grad.as_ptr(), &self.parents[0].data, false),
            ]
        }
    }
}
pub struct MatMul {
    parents: [Tensor; 2],
}

impl MatMul {
    pub fn apply(p1: &Tensor, p2: &Tensor) -> Tensor {
        Tensor::new(
            MatMul::forward(p1, p2),
            Some(Box::new(Self {
                parents: [p1.get(), p2.get()],
            })),
        )
    }

    fn forward(p1: &Tensor, p2: &Tensor) -> Array<f64> {
        matmul(
            &p1.data,
            &p2.data,
            arrayfire::MatProp::NONE,
            arrayfire::MatProp::TRANS,
        )
    }
}

impl Function for MatMul {
    fn parents(&self) -> &[Tensor; 2] {
        &self.parents
    }

    fn backward(&self, grad: &RefCell<Array<f64>>) -> [Array<f64>; 2] {
        unsafe {
            [
                matmul(
                    &*grad.as_ptr(),
                    &self.parents[1].data,
                    arrayfire::MatProp::NONE,
                    arrayfire::MatProp::TRANS,
                ),
                matmul(
                    &*grad.as_ptr(),
                    &self.parents[0].data,
                    arrayfire::MatProp::TRANS,
                    arrayfire::MatProp::NONE,
                ),
            ]
        }
    }
}
