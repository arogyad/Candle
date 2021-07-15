#![allow(dead_code)]
pub mod linear;
pub mod poly;
pub mod traits;

// Imports for the Data Creation
use ndarray::{Array2, ArrayView2};
use ndarray_rand::rand_distr::{Distribution, Standard};
use std::ops::{Deref, DerefMut};

use self::{linear::Linear, poly::Poly};

pub
enum Reg
{
  Linear,
  Poly(i32),
}

pub struct Regression<'a, T>
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
    pub fn new(mode: Reg, data: Array2<T>, label: ArrayView2<'a, T>) -> Self {
        Regression {
            model: match mode {
                Reg::Linear => Linear::new(data, label),
                Reg::Poly(v) => Poly::new(data, label, v).get(),
            },
        }
    }
}

impl<'a, T> DerefMut for Regression<'a, T>
where
    T: num_traits::Num + num_traits::cast::FromPrimitive,
{
  fn deref_mut(&mut self) -> &mut Self::Target {
     &mut self.model 
  }
}

impl<'a, T> Deref for Regression<'a, T> 
where
    T: num_traits::Num + num_traits::cast::FromPrimitive,
{
  type Target = Linear<'a, T>;
 
  fn deref(&self) -> &Self::Target {
    &self.model 
  } 
}
