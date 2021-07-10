use super::traits::Model;

// Imports
use ndarray::{prelude::*, ScalarOperand};
use ndarray_rand::rand_distr::{Distribution, Standard};
use num_traits;
use rand;

pub struct Linear<'a, T>
where
    T: num_traits::Num + num_traits::cast::FromPrimitive,
{
    data: Array2<T>,
    label: ArrayView2<'a, T>,
    theta: Array2<T>,
}

impl<'a, T> Linear<'a, T>
where
    T: num_traits::Num + num_traits::cast::FromPrimitive,
    Standard: Distribution<T>,
{
    pub fn new(data: Array2<T>, label: ArrayView2<'a, T>) -> Self {
        let theta = Array2::from_shape_fn((data.ncols(), 1), |(_, _)| rand::random());
        Linear { data, label, theta }
    }
}

impl<'a, T> Model for Linear<'a, T>
where
    T: num_traits::float::Float + num_traits::cast::FromPrimitive + ScalarOperand,
    f64: Into<T>,
{
    type Item = T;

    fn normalize(&mut self) {
        let mean = self.data.mean_axis(Axis(0)).unwrap();
        let std = self.data.std_axis(Axis(0), 0.0.into());
        if &std[[0]] != &0.0.into() {
            self.data = (&self.data - mean) / std;
        } else {
            eprint!("The std is 0. Not normalized!");
        }
    }

    fn hypo(&self) -> ndarray::Array2<Self::Item> {
        self.data.dot(&self.theta)
    }

    fn train(&mut self, alpha: T, iter: i32) {
        self.gradient(alpha, iter);
    }

    fn gradient(&mut self, alpha: T, iter: i32) {
        for _i in 1..iter {
            self.step(alpha);
        }
    }

    fn predict(&self, input: &Array2<T>) -> Array2<T> {
        input.dot(&self.theta)
    }

    fn step(&mut self, alpha: T) {
        let delta =
            ((self.hypo() - &self.label).reversed_axes().dot(&self.data)).reversed_axes() * alpha;
        self.theta = &self.theta - delta;
    }
}
