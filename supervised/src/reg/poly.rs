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
    pub fn new(data: Array2<T>, label: ArrayView2<'a, T>, poly: i32) -> Self {
        let data = Poly::make_poly(data, poly);
        Poly {
            lin: Linear::new(data, label),
        }
    }

    fn make_poly(data: Array2<T>, poly: i32) -> Array2<T> {
        let split = (data.ncols() / 2) as usize;
        let data_1 = data.slice(s![.., 0..split]);
        let data_2 = data.slice(s![.., split..split * 2]);
        let mut _temp = data.clone();
        for i in 1..poly + 1 {
            for j in 0..i + 1 {
                _temp = concatenate![
                    Axis(1),
                    _temp,
                    (data_1.mapv(|a| a.powi(i - j)) * data_2.mapv(|a| a.powi(j)))
                ];
            }
        }
        if split % 2 == 0 {
            _temp
        } else {
            concatenate![Axis(1), _temp, data.slice(s![.., split * 2..])]
        }
    }

    pub fn get(self) -> Linear<'a, T> {
        self.lin
    }
}
