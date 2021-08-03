#![allow(dead_code)]
use arrayfire::{constant, div, exp, identity, matmul, mean, print, randn, Array, Dim4};

pub struct Tensor {
    data: Array<f64>,
    grad: Option<Array<f64>>,
}

impl Tensor {
    // Main Creation Function
    pub fn new(data: Array<f64>, req_grad: bool) -> Self {
        if req_grad {
            Self {
                grad: Some(constant(1.0f64, data.dims())),
                data,
            }
        } else {
            Self { data, grad: None }
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
        Tensor::new(constant(0.0f64, dims), req_grad)
    }

    pub fn randn(dims: Dim4, req_grad: bool) -> Self {
        Tensor::new(randn(dims), req_grad)
    }

    pub fn eye(dims: Dim4, req_grad: bool) -> Self {
        Tensor::new(identity(dims), req_grad)
    }

    pub fn single(value: f64, dim: Dim4, req_grad: bool) -> Self {
        Tensor::new(constant(value, dim), req_grad)
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
}

// Helper Function Definitions
impl std::fmt::Display for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{:?}", print(&self.data))
    }
}
