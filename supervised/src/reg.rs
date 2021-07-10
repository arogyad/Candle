#![allow(dead_code)]
pub mod linear;
pub mod poly;
mod traits;

enum Reg
{
  Linear,
  Poly,
}
