use crate::tensor::Tensor;

pub trait Model {
    fn forward(&mut self, data: &Tensor) -> &Tensor;
    fn backward(&self);
}
