#![allow(dead_code)]
use ndarray::{concatenate, prelude::*, Array2};
use rand::{self, prelude::SliceRandom};

pub struct KMeans {
    data: Array2<f64>,
    cluster: usize,
}

impl KMeans {
    pub fn new(data: Array2<f64>, cluster: usize) -> Self {
        Self { data, cluster }
    }

    pub fn train(&self, iter: usize) -> (Array1<usize>, Array2<f64>) {
        let mut cent = self.create_centroid();
        let mut maybe_centroids = Array1::from_vec(Vec::new());
        for _ in 0..iter {
            maybe_centroids = self.find_closest(cent.view());
            cent = self.compute(maybe_centroids.view());
        }
        (maybe_centroids, cent)
    }

    fn create_centroid(&self) -> Array2<f64> {
        let index = KMeans::create_permute(self.data.dim().0);
        let mut _arr = Array2::from_elem((0, self.data.dim().1), 0.);
        for i in 0..self.cluster {
            _arr = concatenate![
                Axis(0),
                _arr,
                self.data
                    .index_axis(Axis(0), index[i])
                    .to_owned()
                    .into_shape((1, self.data.dim().1))
                    .unwrap()
            ];
        }
        _arr
    }

    fn find_closest(&self, centroid: ArrayView2<f64>) -> Array1<usize> {
        let shape = self.data.dim();
        let cen_shape = self.data.dim();
        let mut maybe_centroids = Array1::zeros(shape.0);
        for i in 0..shape.0 {
            let mut distance = Array::zeros(cen_shape.0);
            for cent in 0..cen_shape.0 {
                let diff =
                    self.data.index_axis(Axis(0), i).to_owned() - centroid.index_axis(Axis(0), i);
                distance[cent] = diff.sum();
            }
            maybe_centroids[i] = KMeans::argmin(distance.view());
        }
        maybe_centroids
    }

    fn compute(&self, maybe_centroids: ArrayView1<usize>) -> Array2<f64> {
        let mut _arr = Array2::from_elem((0, self.data.dim().1), 0.);
        for index in 0..self.cluster {
            let close = KMeans::equal(maybe_centroids, index);
            let mut _temp = Array2::from_elem((0, self.data.dim().1), 0.);
            for val in close {
                _temp = concatenate![
                    Axis(0),
                    _temp,
                    self.data
                        .index_axis(Axis(0), val)
                        .to_owned()
                        .into_shape((1, self.data.dim().1))
                        .unwrap()
                ];
            }
            _arr = concatenate![
                Axis(0),
                _arr,
                _temp
                    .mean_axis(Axis(0))
                    .unwrap()
                    .into_shape((1, self.data.dim().1))
                    .unwrap()
            ];
        }
        _arr
    }

    // Helper Functions
    // create_permute -> Create Random Permutation Vec of usize
    // argmin -> Find the position of minimum value in the given array (slow?!)
    // equal -> seperate where the index and the value in the array are same!
    fn create_permute(number: usize) -> Vec<usize> {
        let mut _arr: Vec<usize> = (0..number).collect();
        _arr.shuffle(&mut rand::thread_rng());
        _arr
    }

    fn argmin(data: ArrayView1<f64>) -> usize {
        let mut index = 0;
        for i in 0..data.dim() {
            if data[i] > data[index] {
                index = i;
            }
        }
        index
    }

    fn equal(check: ArrayView1<usize>, index: usize) -> Array1<usize> {
        let mut _ids = Vec::new();
        for i in 0..check.dim() {
            if check[i] == index {
                _ids.push(i);
            }
        }
        Array1::from_iter(_ids.into_iter())
    }
}
