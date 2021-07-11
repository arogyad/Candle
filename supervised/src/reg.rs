#![allow(dead_code)]
pub mod linear;
pub mod poly;

// Imports for the Data Creation
use ndarray::{Array2, ArrayView2};
use ndarray_rand::rand_distr::{Distribution, Standard};

use self::{linear::Linear, poly::Poly};
mod traits;

enum Reg {
    Linear,
    Poly(i32),
}

struct Regression<'a, T>
where
    T: num_traits::Num + num_traits::cast::FromPrimitive,
{
    model: Linear<'a, T>,
}

impl<'a, T> Regression<'a, T>
where
    T: num_traits::Float + num_traits::cast::FromPrimitive,
    Standard: Distribution<T>,
{
    fn new(model: Reg, data: Array2<T>, label: ArrayView2<'a, T>) -> Self {
        Regression {
            model: match model {
                Reg::Linear => Linear::new(data, label),
                Reg::Poly(v) => Poly::new(data, label, v).get(),
            },
        }
    }
}
