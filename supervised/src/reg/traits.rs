use arrayfire::Array;

pub trait Model {
    fn normalize(&mut self);

    /// Inner product of the data and the theta
    fn hypo(&self) -> Array<f64>;

    /// Train the model
    fn train(&mut self, alpha: f64, iter: i32);

    /// Entire gradient calculation step is done here
    fn gradient(&mut self, alpha: f64, iter: i32);

    /// Predict based of an input using the trained theta
    fn predict(&self, input: &Array<f64>) -> Array<f64>;

    /// One step of gradient descent
    fn step(&mut self, alpha: f64);
}
