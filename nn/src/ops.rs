use super::tensor::Tensor;
use arrayfire::{dim4, matmul, pow, tile, transpose, Array};

enum Dim {
    Zero,
    One,
}

pub trait Function {
    fn parents(&self) -> &[Tensor];
    fn backward(&self, grad: Option<&Array<f64>>) -> [Option<Array<f64>>; 2];
}

pub struct Add {
    parents: [Tensor; 2],
}

impl Add {
    pub fn apply(p1: Tensor, p2: Tensor) -> Tensor {
        Tensor::new(
            Add::broadcast_sum(&p1.data, &p2.data),
            Some(Box::new(Self { parents: [p1, p2] })),
        )
    }
    // Simple broadcasting addition, can do the same to create broadcasting sub, mul etc..
    fn broadcast_sum(p1: &Array<f64>, p2: &Array<f64>) -> Array<f64> {
        // dim -> which dimension needs broadcasting, here we only look at the first and the second
        // as this makes the process simpler
        match (&p1.dims(), &p2.dims()) {
            // Check if the axis for broadcasting
            (x, y) if x[0] != y[0] => {
                if x[0] > y[0] {
                    p1 + Add::tile_sum(p2, x[0] - y[0] + 1, Dim::Zero)
                } else {
                    p2 + Add::tile_sum(p1, y[0] - x[0] + 1, Dim::Zero)
                }
            }
            (x, y) if x[1] != y[1] => {
                if x[1] > y[1] {
                    p1 + Add::tile_sum(p2, x[1] - y[1] + 1, Dim::One)
                } else {
                    p2 + Add::tile_sum(p1, y[1] - x[1] + 1, Dim::One)
                }
            }
            _ => p1 + p2,
        }
    }

    // tiles the smaller array to the size of larger array on the given dimension
    // input -> The smaller among the two arrays
    // dim: Dimension to be broadcasted
    // num_tiles ->  no. of times to be tiled i.e, dim of larger - dim of smaller + 1 along dim
    #[inline]
    fn tile_sum(input: &Array<f64>, num_tile: u64, dim: Dim) -> Array<f64> {
        match dim {
            Dim::Zero => tile(input, dim4!(num_tile, 1, 1, 1)),
            Dim::One => tile(input, dim4!(1, num_tile, 1, 1)),
        }
    }
}

impl Function for Add {
    fn parents(&self) -> &[Tensor] {
        &self.parents
    }

    fn backward(&self, grad: Option<&Array<f64>>) -> [Option<Array<f64>>; 2] {
        [Some(grad.unwrap().copy()), Some(grad.unwrap().copy())]
    }
}

pub struct Mul {
    parents: [Tensor; 2],
}

impl Mul {
    pub fn apply(p1: Tensor, p2: Tensor) -> Tensor {
        Tensor::new(
            &p1.data * &p2.data,
            Some(Box::new(Self { parents: [p1, p2] })),
        )
    }
}

impl Function for Mul {
    fn parents(&self) -> &[Tensor] {
        &self.parents
    }

    fn backward(&self, grad: Option<&Array<f64>>) -> [Option<Array<f64>>; 2] {
        // Grads are implicitly defined as const(1., ..) so this dereference shouldn't cause error
        [
            Some(arrayfire::mul(grad.unwrap(), &self.parents[1].data, false)),
            Some(arrayfire::mul(grad.unwrap(), &self.parents[0].data, false)),
        ]
    }
}

pub struct MatMul {
    parents: [Tensor; 2],
}

impl MatMul {
    pub fn apply(p1: &Tensor, p2: &Tensor) -> Tensor {
        Tensor::new(
            MatMul::forward(p1, p2),
            Some(Box::new(Self {
                parents: [p1.get(), p2.get()],
            })),
        )
    }

    fn forward(p1: &Tensor, p2: &Tensor) -> Array<f64> {
        matmul(
            &p1.data,
            &p2.data,
            arrayfire::MatProp::NONE,
            arrayfire::MatProp::TRANS,
        )
    }
}

impl Function for MatMul {
    fn parents(&self) -> &[Tensor] {
        &self.parents
    }

    fn backward(&self, grad: Option<&Array<f64>>) -> [Option<Array<f64>>; 2] {
        [
            Some(matmul(
                grad.unwrap(),
                &self.parents[1].data,
                arrayfire::MatProp::NONE,
                arrayfire::MatProp::NONE,
            )),
            Some(transpose(
                &matmul(
                    &self.parents[0].data,
                    grad.unwrap(),
                    arrayfire::MatProp::TRANS,
                    arrayfire::MatProp::NONE,
                ),
                false,
            )),
        ]
    }
}

pub struct Pow {
    parents: [Tensor; 1],
    n: i32,
}

impl Pow {
    pub fn apply(p1: &Tensor, n: i32) -> Tensor {
        Tensor::new(
            Pow::forward(p1, &n),
            Some(Box::new(Self {
                parents: [p1.get()],
                n,
            })),
        )
    }

    fn forward(p1: &Tensor, n: &i32) -> Array<f64> {
        pow(&p1.data, n, false)
    }
}

impl Function for Pow {
    fn parents(&self) -> &[Tensor] {
        &self.parents
    }

    // The grad is always "None"
    fn backward(&self, _grad: Option<&Array<f64>>) -> [Option<Array<f64>>; 2] {
        [
            Some(-self.n * pow(&self.parents[0].data, &(self.n - 1), false)),
            None,
        ]
    }
}
