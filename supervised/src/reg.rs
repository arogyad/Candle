#![allow(dead_code)]
pub mod linear;
pub mod poly;
pub mod traits;

// Imports for the Data Creation
use ndarray::{Array2, ArrayView2};
use std::ops::{Deref, DerefMut};

use self::{linear::Linear, poly::Poly};

pub
enum Reg
{
  Linear,
  Poly(i32),
}

pub struct Regression<'a>
{
    model: Linear<'a>,
}

impl<'a> Regression<'a>
{
    pub fn new(mode: Reg, data: Array2<f64>, label: ArrayView2<'a, f64>) -> Self {
        Regression {
            model: match mode {
                Reg::Linear => Linear::new(data, label),
                Reg::Poly(v) => Poly::new(data, label, v).get(),
            },
        }
    }
}

impl<'a> DerefMut for Regression<'a>
{
  fn deref_mut(&mut self) -> &mut Self::Target {
     &mut self.model 
  }
}

impl<'a> Deref for Regression<'a> 
{
  type Target = Linear<'a>;
 
  fn deref(&self) -> &Self::Target {
    &self.model 
  } 
}
