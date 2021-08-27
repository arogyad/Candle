#![allow(unused)]
use super::ops::Function;
use arrayfire::add;
use arrayfire::{constant, Array};
use std::cell::RefCell;
use std::ops::{Add, Deref, Mul};
use std::rc::Rc;

// The make do tensor class which is a wrapper around Rc<WTen>.
pub struct Tensor(pub Rc<WTen>);

impl Tensor {
    pub fn new(data: Array<f64>, _ctx: Option<Box<dyn Function>>) -> Self {
        Self(Rc::new(WTen::new(data, _ctx)))
    }

    pub fn get(&self) -> Tensor {
        Self(Rc::clone(&self.0))
    }
}

// Basic Operations
impl Deref for Tensor {
    type Target = WTen;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Add for &Tensor {
    type Output = Tensor;
    fn add(self, rhs: Self) -> Self::Output {
        super::ops::Add::apply(self.get(), rhs.get())
    }
}

impl Mul for &Tensor {
    type Output = Tensor;
    fn mul(self, rhs: Self) -> Self::Output {
        super::ops::Mul::apply(self.get(), rhs.get())
    }
}

// The actual "Tensor", everything inside the Tensor gets derefed to this tensor. This is named
// "WTen"; however, the above should have been named WTen and this should have been named Tensor,
// but this sacrifice had to be made for better readability.
pub struct WTen {
    pub data: Array<f64>,
    pub grad: RefCell<Array<f64>>,
    pub _ctx: Option<Box<dyn Function>>,
}

impl WTen {
    pub fn new(data: Array<f64>, _ctx: Option<Box<dyn Function>>) -> Self {
        Self {
            grad: RefCell::new(constant(0., data.dims())),
            data,
            _ctx,
        }
    }

    fn _deepwalk<'a>(node: &'a WTen, nodes: &mut Vec<&'a WTen>, visited: &mut Vec<&'a WTen>) {
        if let Some(n) = &node._ctx {
            visited.push(node);
            for i in n.parents() {
                if !visited.contains(&i.0.as_ref()) {
                    Self::_deepwalk(i, nodes, visited)
                }
            }
        }
    }

    fn walk(&self) -> Vec<&WTen> {
        let mut nodes = Vec::new();
        let mut visited = Vec::new();
        Self::_deepwalk(self, &mut nodes, &mut visited);
        nodes.reverse();
        nodes
    }

    pub fn backward(&self) {
        self.grad.replace(constant(1., self.data.dims()));
        for t0 in self.walk() {
            let grads = t0._ctx.as_ref().unwrap().backward(&t0.grad);
            for (t, g) in t0._ctx.as_ref().unwrap().parents().iter().zip(grads) {
                t.grad.replace_with(|old| add(old, &g, false));
            }
        }
    }
}

impl PartialEq for WTen {
    fn eq(&self, other: &Self) -> bool {
        std::ptr::eq(self, other)
    }
}

impl Eq for WTen {}
