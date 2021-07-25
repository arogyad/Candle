use ndarray::{self, Array2};

pub trait Model {
    fn normalize(&mut self);

    /// Inner product of the data and the theta
    fn hypo(&self) -> ndarray::Array2<f64>;

    /// Train the model
    fn train(&mut self, alpha: f64, iter: i32);

    fn gradient(&mut self, alpha: f64, iter: i32);

    /// Predict based of an input using the trained theta
    fn predict(&self, input: &Array2<f64>) -> Array2<f64>;

    /// One step of gradient descent
    fn step(&mut self, alpha: f64);
}
