use crate::tensor::Tensor;

pub trait Model {
    fn forward(&self, data: &Tensor) -> Tensor;
    fn backward(&self);
}
