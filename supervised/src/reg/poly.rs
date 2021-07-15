use super::linear::Linear;
use ndarray::{concatenate, prelude::*, Array2};
use ndarray_rand::rand_distr::{Distribution, Standard};
use num_traits;

pub struct Poly<'a, T>
where
    T: num_traits::Num + num_traits::FromPrimitive,
{
  lin: Linear<'a, T>,
}

impl<'a, T> Poly<'a, T>
where
    T: num_traits::Float + num_traits::cast::FromPrimitive,
    Standard: Distribution<T>,
{
    pub fn new(mut data: Array2<T>, label: ArrayView2<'a, T>, poly: i32) -> Self {
        Poly::make_poly(&mut data, poly);
        Poly {
            lin: Linear::new(data, label),
        }
    }

    fn make_poly(data: &mut Array2<T>, poly: i32) {
        let split = (data.ncols() / 2) as usize;
        let data_1 = data.slice(s![.., 0..split]);
        let data_2 = data.slice(s![.., split..split * 2]);
        for i in 1..poly + 1 {
            for j in 0..i + 1 {
                concatenate![
                    Axis(1),
                    *data,
                    (data_1.mapv(|a| a.powi(i - j)) * data_2.mapv(|a| a.powi(j)))
                ];
            }
        }
        if split % 2 != 0 {
            concatenate![Axis(1), *data, data.slice(s![.., split * 2..])];
        }
    }
    pub(super)fn get(self) -> Linear<'a, T> {
      self.lin
    } 
}

/*
impl<'a, T> DerefMut for Poly<'a, T>
where
    T: num_traits::float::Float + num_traits::cast::FromPrimitive,
{
  fn deref_mut(&mut self) -> &mut Self::Target {
     &mut self.lin
  }
}

impl<'a, T> Deref for Poly<'a, T>
where
    T: num_traits::float::Float + num_traits::cast::FromPrimitive,
{
  type Target = Linear<'a, T>;

  fn deref(&self) -> &Self::Target {
    &self.lin
  }
}
*/
