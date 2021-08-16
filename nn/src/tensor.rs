#![allow(dead_code)]
#![allow(unused)]
use super::ops::Function;
use arrayfire::{add, constant, div, exp, identity, matmul, mean, print, randn, Array, Dim4};
use core::cmp::{Eq, PartialEq};
use std::cell::RefCell;

pub struct Tensor<'a> {
    pub data: Array<f64>,
    pub grad: Option<RefCell<Array<f64>>>,
    pub _ctx: Option<&'a dyn Function>,
}

impl<'a> Tensor<'a> {
    // Main Creation Function
    pub fn new(data: Array<f64>, _ctx: Option<&'a dyn Function>, req_grad: bool) -> Self {
        if req_grad {
            Self {
                grad: None,
                data,
                _ctx,
            }
        } else {
            Self {
                data,
                grad: None,
                _ctx,
            }
        }
    }

    // Internal Representation Function
    pub fn assign(&mut self, x: Array<f64>) {
        self.data = x;
    }

    pub fn shape(&self) -> Dim4 {
        self.data.dims()
    }

    // Constructors
    pub fn zeros(dims: Dim4, req_grad: bool) -> Self {
        Tensor::new(constant(0.0f64, dims), None, req_grad)
    }

    pub fn randn(dims: Dim4, req_grad: bool) -> Self {
        Tensor::new(randn(dims), None, req_grad)
    }

    pub fn eye(dims: Dim4, req_grad: bool) -> Self {
        Tensor::new(identity(dims), None, req_grad)
    }

    pub fn single(value: f64, dim: Dim4, req_grad: bool) -> Self {
        Tensor::new(constant(value, dim), None, req_grad)
    }

    // Other Functions
    pub fn mean(&self, axis: i64) -> Array<f64> {
        mean(&self.data, axis)
    }

    pub fn dot(&self, w: Array<f64>) -> Array<f64> {
        matmul(
            &self.data,
            &w,
            arrayfire::MatProp::NONE,
            arrayfire::MatProp::NONE,
        )
    }

    pub fn sigmoid(&self) -> Array<f64> {
        let y = exp(&self.data);
        div(&y, &(&y + 1), false)
    }

    // Backward pass
    fn _deepwalk(
        node: &'a Tensor<'a>,
        nodes: &'_ mut Vec<&'a Tensor<'a>>,
        visited: &mut Vec<&'a Tensor<'a>>,
    ) {
        if let Some(n) = &node._ctx {
            visited.push(node);
            for i in n.parents() {
                if !visited.contains(&i) {
                    Self::_deepwalk(i, nodes, visited);
                }
            }
        }
        nodes.push(node);
    }

    fn walk(&'a self) -> Vec<&Tensor> {
        let mut nodes = Vec::new();
        let mut visited = Vec::new();
        Self::_deepwalk(self, &mut nodes, &mut visited);
        nodes.reverse();
        nodes
    }

    pub fn backward(&mut self) {
        self.grad = Some(RefCell::new(constant(1., self.data.dims())));
        for t0 in self.walk() {
            let grads = t0
                ._ctx
                .as_ref()
                .unwrap()
                .backward(t0.grad.as_ref().unwrap());
            for (t, g) in t0._ctx.as_ref().unwrap().parents().iter().zip(grads) {
                unsafe {
                    t.grad
                        .as_ref()
                        .unwrap()
                        .replace_with(|old| add(old, g.as_ptr().as_ref().unwrap(), false));
                }
            }
        }
    }
}

// Helper Function Definitions
impl<'a> std::fmt::Display for Tensor<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{:?}", print(&self.data))
    }
}

impl<'a> PartialEq for Tensor<'a> {
    fn eq(&self, other: &Self) -> bool {
        std::ptr::eq(self, other)
    }
}
impl<'a> Eq for Tensor<'a> {}
