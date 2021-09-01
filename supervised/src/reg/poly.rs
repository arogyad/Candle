use super::linear::Linear;
use arrayfire::{index, join, pow, Array, Seq};
pub struct Poly<'a> {
    lin: Linear<'a>,
}

impl<'a> Poly<'a> {
    pub fn new(mut data: Array<f64>, label: &'a Array<f64>, poly: i32) -> Self {
        Poly::make_poly(&mut data, poly);
        Poly {
            lin: Linear::new(data, label),
        }
    }

    fn make_poly(data: &mut Array<f64>, poly: i32) {
        let dims = data.dims();
        let data_1 = index(
            data,
            &[
                Seq::new(0.0, dims[0] as f32, 1.0),
                Seq::new(0.0, (dims[1] / 2) as f32, 1.0),
            ],
        );
        let data_2 = index(
            data,
            &[
                Seq::new(0.0, dims[0] as f32, 1.0),
                Seq::new((dims[1] / 2) as f32, dims[1] as f32, 1.0),
            ],
        );
        for i in 1..poly + 1 {
            for j in 0..i + 1 {
                *data = join(1, data, &pow(&data_1, &(i - j), false));
                *data = join(1, data, &pow(&data_2, &j, false));
            }
        }
    }

    pub fn get(self) -> Linear<'a> {
        self.lin
    }
}
