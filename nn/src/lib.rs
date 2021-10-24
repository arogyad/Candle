#![allow(unused)]
pub mod conv2d;
pub mod linear;
pub mod model;
pub mod ops;
pub mod optim;
pub mod tensor;

use model::Model;
use tensor::Tensor;

// This is just the basic skeleton. Everything is yet to be done
struct Sequential {
    models: Vec<Box<dyn Model>>,
}

impl Sequential {
    fn forward(&mut self, data: &Tensor) {
        let mut data = data;
        for model in &mut self.models {
            data = model.forward(data);
        }
    }

    fn backward(&self) {
        for model in self.models.iter().rev() {
            model.backward();
        }
    }
}
