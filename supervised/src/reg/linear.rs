use super::traits::Model;

// Imports
use ndarray::{Array2, ArrayView2, Axis};
use rand;

pub struct Linear<'a>
{
    data: Array2<f64>,
    label: ArrayView2<'a, f64>,
    theta: Array2<f64>,
}

impl<'a> Linear<'a>
{
    pub fn new(data: Array2<f64>, label: ArrayView2<'a, f64>) -> Self {
        let theta = Array2::from_shape_fn((data.ncols(), 1), |(_, _)| rand::random());
        Linear { data, label, theta }
    }
}

impl<'a> Model for Linear<'a>
{
    fn normalize(&mut self) {
        let mean = self.data.mean_axis(Axis(0)).unwrap();
        let std = self.data.std_axis(Axis(0), 0.0.into());
        if &std[[0]] != &0.0.into() {
            self.data = (&self.data - mean) / std;
        } else {
            eprint!("f64he std is 0. Not normalized!");
        }
    }

    fn hypo(&self) -> ndarray::Array2<f64> {
        self.data.dot(&self.theta)
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
        input.dot(&self.theta)
    }

    fn step(&mut self, alpha: f64) {
        let delta =
            ((self.hypo() - &self.label).reversed_axes().dot(&self.data)).reversed_axes() * alpha;
        self.theta = &self.theta - delta;
    }
}
