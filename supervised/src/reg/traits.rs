use ndarray::{self, Array2};

pub trait Model {
    type Item;

    /// Normalize the given data
    /// data = (data - mean) / std;
    fn normalize(&mut self);

    /// Inner product of the data and the theta
    fn hypo(&self) -> ndarray::Array2<Self::Item>;

    /// Train the model
    fn train(&mut self, alpha: Self::Item, iter: i32);

    fn gradient(&mut self, alpha: Self::Item, iter: i32);

    /// Predict based of an input using the trained theta
    fn predict(&self, input: &Array2<Self::Item>) -> Array2<Self::Item>;

    /// One step of gradient descent
    fn step(&mut self, alpha: Self::Item);
}
