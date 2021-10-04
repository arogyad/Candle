#![allow(unused)]
use crate::tensor::Tensor;

fn sgd(tensor: &mut Tensor) {
    let _temp = arrayfire::sub(&tensor.data, tensor.grad.borrow().as_ref().unwrap(), false);
    tensor.data = _temp;
}
