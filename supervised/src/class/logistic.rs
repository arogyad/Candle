#![allow(unused)]
use ndarray::{Array2, ArrayView2};
use ndarray_rand::rand_distr::{Distribution, Standard};

struct Logistic<'a, T>
where
    T:num_traits::Num + num_traits::cast::FromPrimitive,
{
  data: Array2<T>,
  label: ArrayView2<'a, T>,
  theta: Array2<T>
}

impl<'a, T> Logistic<'a, T> 
where
  T: num_traits::float::Float + num_traits::cast::FromPrimitive,
  Standard: Distribution<T>
{
  fn new(data: Array2<T>, label: ArrayView2<'a, T>) -> Self {
    let theta = Array2::from_shape_fn((data.ncols(), 1), |(_,_)| rand::random());
    Logistic {
      data,
      label,
      theta,
    }
  }
}
