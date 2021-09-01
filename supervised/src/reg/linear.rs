#![allow(unused)]
use super::traits::Model;

// Imports
use arrayfire::{dim4, matmul, randn, transpose, Array, MatProp};

pub struct Linear<'a> {
    data: Array<f64>,
    label: &'a Array<f64>,
    theta: Array<f64>,
}

impl<'a> Linear<'a> {
    pub fn new(data: Array<f64>, label: &'a Array<f64>) -> Self {
        let theta = randn(dim4!(data.dims()[4], 1, 1, 1));
        Linear { data, label, theta }
    }
}

impl<'a> Model for Linear<'a> {
    fn normalize(&mut self) {
        let mean = arrayfire::mean(&self.data, 0);
        let std = arrayfire::stdev_v2(&self.data, arrayfire::VarianceBias::DEFAULT, 0);
        self.data = (&self.data - mean) / std;
    }

    fn hypo(&self) -> Array<f64> {
        matmul(&self.data, &self.theta, MatProp::NONE, MatProp::NONE)
    }

    fn train(&mut self, alpha: f64, iter: i32) {
        self.gradient(alpha, iter);
    }

    fn gradient(&mut self, alpha: f64, iter: i32) {
        for _i in 1..iter {
            self.step(alpha);
        }
    }

    fn predict(&self, input: &Array<f64>) -> Array<f64> {
        matmul(input, &self.theta, MatProp::NONE, MatProp::NONE)
    }

    fn step(&mut self, alpha: f64) {
        let delta = matmul(
            &transpose(&(self.hypo() - self.label), false),
            &self.data,
            MatProp::NONE,
            MatProp::TRANS,
        ) * alpha;
        self.theta = &self.theta - delta;
    }
}
