#![allow(dead_code)]
use ndarray::{concatenate, prelude::*, Array2};
use rand::{self, prelude::SliceRandom};

pub struct Knn {
    data: Array2<f64>,
    cluster: usize,
}

impl Knn {
    fn new(data: Array2<f64>, cluster: usize) -> Self {
        Self { data, cluster }
    }

    fn create_centroid(data: &Array2<f64>, cluster: usize) -> Array2<f64> {
        let index = Knn::create_permute(data.dim().0);
        let mut _arr = Array2::from_elem((0, data.dim().1), 0.);
        for i in 0..cluster {
            _arr = concatenate![
                Axis(0),
                _arr,
                data.index_axis(Axis(0), index[i])
                    .to_owned()
                    .into_shape((1, data.dim().1))
                    .unwrap()
            ];
        }
        _arr
    }

    fn create_permute(number: usize) -> Vec<usize> {
        let mut _arr: Vec<usize> = (0..number).collect();
        _arr.shuffle(&mut rand::thread_rng());
        _arr
    }
}
