#![allow(unused)]
use arrayfire::{constant, dim4, print, randn, Array};
use nn::linear::Linear;
use nn::model::Model;
use nn::tensor::Tensor;

#[test]
fn linear() {
    let a = Tensor::new(constant(2., dim4!(200, 200, 1, 1)), None);
    let b = Tensor::new(constant(3., dim4!(200, 200, 1, 1)), None);
    let c = &a * &b;
    let d = Tensor::new(constant(4., dim4!(200, 200, 1, 1)), None);
    let e = &c + &d;
    e.backward();
}

#[test]
fn linear_class() {
    let data = Tensor::new(randn(dim4!(256, 256, 1, 1)), None);
    let mut l = Linear::new(256, 128, false);
    l.forward(&data);
    l.backward();
    unsafe { print(&(*data.grad.as_ptr())) }
}
