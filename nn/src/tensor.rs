#![allow(unused)]
use super::ops::Function;
use arrayfire::{add, dim4};
use arrayfire::{constant, Array};
use std::cell::RefCell;
use std::ops::{Add, Deref, DerefMut, Mul};
use std::rc::Rc;

// The make do tensor class which is a wrapper around Rc<WTen>.
pub struct Tensor(Rc<WTen>);
impl Tensor {
    pub fn new(data: Array<f64>, _ctx: Option<Box<dyn Function>>) -> Self {
        Self(Rc::new(WTen::new(data, _ctx)))
    }

    pub fn get(&self) -> Tensor {
        Self(Rc::clone(&self.0))
    }

    fn pow(&self, n: i32) -> Tensor {
        crate::ops::Pow::apply(self, n)
    }
}

// Basic Operations
impl Deref for Tensor {
    type Target = WTen;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for Tensor {
    fn deref_mut(&mut self) -> &mut Self::Target {
        Rc::get_mut(&mut self.0).unwrap()
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
    pub grad: RefCell<Option<Array<f64>>>,
    pub _ctx: Option<Box<dyn Function>>,
}

impl WTen {
    pub fn new(data: Array<f64>, _ctx: Option<Box<dyn Function>>) -> Self {
        Self {
            grad: RefCell::new(None),
            data,
            _ctx,
        }
    }

    // Create a new tensor with the raised to that power

    fn _deepwalk<'a>(node: &'a WTen, nodes: &mut Vec<&'a WTen>, visited: &mut Vec<&'a WTen>) {
        if let Some(n) = &node._ctx {
            // Might cause error in future
            visited.push(node);
            for i in n.parents() {
                if !visited.contains(&i.0.as_ref()) {
                    Self::_deepwalk(i, nodes, visited)
                }
            }
            nodes.push(node);
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
        self.grad.replace(Some(constant(1., self.data.dims())));
        for t0 in self.walk() {
            if let Some(n) = t0.grad.borrow().as_ref() {
                // TODO: FIX this by adding leaf or not!!
                let grads = t0._ctx.as_ref().unwrap().backward(Some(n));
                for (t, g) in t0._ctx.as_ref().unwrap().parents().iter().zip(grads) {
                    if let Some(n) = g {
                        if let Some(tg) = t.grad.borrow().as_ref() {
                            t.grad
                                .replace_with(|old| Some(add(old.as_ref().unwrap(), &n, false)));
                        }
                    }
                }
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
