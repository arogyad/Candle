mod ops;
mod tensor;
#[cfg(test)]
mod tests {
    use super::tensor::Tensor;
    use arrayfire::{constant, dim4};
    #[test]
    fn test1() {
        let a = Tensor::new(constant(2., dim4!(1, 1, 200, 200)), None);
        let b = Tensor::new(constant(3., dim4!(1, 1, 200, 200)), None);
        let c = &a * &b;
        let d = Tensor::new(constant(4., dim4!(1, 1, 200, 200)), None);
        let e = &c + &d;
        e.backward();
    }
}

pub mod benches {
    use super::tensor::Tensor;
    use arrayfire::{constant, dim4};
    pub fn linear(size: u64) {
        let a = Tensor::new(constant(2., dim4!(1, 1, size, size)), None);
        let b = Tensor::new(constant(3., dim4!(1, 1, size, size)), None);
        let c = &a * &b;
        let d = Tensor::new(constant(4., dim4!(1, 1, size, size)), None);
        let e = &c + &d;
        e.backward();
    }
}
