#![allow(dead_code)]
use crate::reg::traits::Model;
use ndarray::{Array2, ArrayView2, Axis};

struct Logistic<'a> {
    data: Array2<f64>,
    label: ArrayView2<'a, f64>,
    theta: Array2<f64>,
}

impl<'a> Logistic<'a> {
    fn new(data: Array2<f64>, label: ArrayView2<'a, f64>) -> Self {
        let theta = Array2::from_shape_fn((data.ncols(), 1), |(_, _)| rand::random());
        Logistic { data, label, theta }
    }
}

impl<'a> Model for Logistic<'a> {
    fn normalize(&mut self) {
        let mean = self.data.mean_axis(Axis(0)).unwrap();
        let std = self.data.std_axis(Axis(0), 0.0);
        if std[[0]] != 0.0 {
            self.data = (&self.data - mean) / std;
        } else {
            eprintln!("The std is 0. Not Normalized!");
        }
    }

    fn hypo(&self) -> ndarray::Array2<f64> {
        self.data.dot(&self.theta).map(|&a| a.exp())
    }

    fn train(&mut self, alpha: f64, iter: i32) {
        self.gradient(alpha, iter);
    }

    fn gradient(&mut self, alpha: f64, iter: i32) {
        for _i in 1..iter {
            self.step(alpha);
        }
    }

    fn predict(&self, input: &Array2<f64>) -> Array2<f64> {
        self.theta.dot(input)
    }

    fn step(&mut self, alpha: f64) {
        self.theta =
            &self.theta - (alpha * (self.hypo() - self.label).reversed_axes().dot(&self.data));
    }
}
