use super::linear::Linear;
use ndarray::{concatenate, prelude::*, Array2};

pub struct Poly<'a>
{
  lin: Linear<'a>,
}

impl<'a> Poly<'a>
{
    pub fn new(mut data: Array2<f64>, label: ArrayView2<'a, f64>, poly: i32) -> Self {
        data = Poly::make_poly(&mut data, poly);
        Poly {
            lin: Linear::new(data, label),
        }
    }

    fn make_poly(data: &Array2<f64>, poly: i32) -> Array2<f64>{
        let split = (data.ncols() / 2) as usize;
        let mut _temp = data.clone();
        let data_1 = data.slice(s![.., 0..split]);
        let data_2 = data.slice(s![.., split..split * 2]);
        for i in 1..poly + 1 {
            for j in 0..i + 1 {
                _temp = concatenate![
                    Axis(1),
                    *data,
                    (data_1.mapv(|a| a.powi(i - j)) * data_2.mapv(|a| a.powi(j)))
                ];
            }
        }
        _temp
        /* if split % 2 != 0 {
            concatenate![Axis(1), *data, data.slice(s![.., split * 2..])];
        }*/
    }

    pub(super)fn get(self) -> Linear<'a> {
      self.lin
    } 
}

