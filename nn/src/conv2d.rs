#![allow(unused)]
use crate::ops::Function;
use crate::tensor::Tensor;
use arrayfire::{
    assign_seq, constant, convolve2_gradient_nn, convolve2_nn, dim4, index, randn, seq, Array,
    ConvGradientType, Dim4, Seq,
};
use std::cell::RefCell;

pub struct Conv2d {
    pub w_a_i: Option<[Tensor; 2]>,
    pub output: Option<Tensor>,
    pub in_channel: u64,
    pub out_channel: u64,
    pub kernel_size: Dim4,
    pub strides: Dim4,
    pub padding: Dim4,
    pub dilation: Dim4,
}

impl Conv2d {
    pub fn new(
        in_channel: u64,
        out_channel: u64,
        kernel_size: [u64; 2],
        strides: [u64; 2],
        padding: [u64; 2],
        dilation: [u64; 2],
    ) -> Self {
        Self {
            w_a_i: None,
            output: None,
            in_channel,
            out_channel,
            kernel_size: dim4!(kernel_size[0], kernel_size[1]),
            strides: dim4!(strides[0], strides[1]),
            padding: dim4!(
                if padding[0] == 0 { 1 } else { padding[0] },
                if padding[1] == 0 { 1 } else { padding[1] }
            ),
            dilation: dim4!(dilation[0], dilation[1]),
        }
    }

    pub fn apply(&mut self, data: Tensor) {
        // data -> [H x W x C x O]
        let dim = data.data.dims();
        if !&self.w_a_i.is_some() {
            // Weight size -> [KS[0] x KS[1] x C x O]
            self.w_a_i = Some([
                Tensor::new(
                    randn(dim4!(
                        self.kernel_size[0],
                        self.kernel_size[1],
                        self.in_channel,
                        self.out_channel
                    )),
                    None,
                ),
                data.get(),
            ]);
        }
        self.output = Some(Tensor::new(self.forward(&dim), None));
    }

    // dim -> dimension of initial data
    pub fn forward(&self, dim: &Dim4) -> Array<f64> {
        let dim = self.w_a_i.as_ref().unwrap()[1].data.dims();

        // Out Shapes
        let ox =
            ((dim[0] + 2 * self.padding[0] - (self.dilation[0] * (self.kernel_size[0] - 1)) - 1)
                / self.strides[0])
                + 1;

        let oy =
            ((dim[1] + 2 * self.padding[1] - (self.dilation[1] * (self.kernel_size[1] - 1)) - 1)
                / self.strides[1])
                + 1;

        // The array to be returned, named after the beloved arrayfire
        let mut af_out = constant(0., dim4!(ox, oy, self.out_channel, dim[3]));
        for i in 0..self.out_channel {
            let mut af_t = constant(0., dim4!(ox, oy, 1, 1));

            // i -> num of output channels
            // j -> num of in channels
            // Each Operation -> input (*) Weight[..,.., i, j]
            // Out -> [ox, oy, out_channel, N]
            for j in 0..self.in_channel {
                // Weight[i][j]
                // This creates a 2d array of weight
                let seqs_w = [
                    seq!(),
                    seq!(),
                    Seq::new(j as i32, j as i32, 1),
                    Seq::new(i as i32, i as i32, 1),
                ];
                // This creates the 2d array of data
                let seqs_d = [seq!(), seq!(), Seq::new(j as i32, j as i32, 1)];

                let s1 = index(&self.w_a_i.as_ref().unwrap()[0].data, &seqs_w);
                // This takes out 1 dimension from data
                let s2 = index(&self.w_a_i.as_ref().unwrap()[1].data, &seqs_d);

                // Arrayfire has conv operation, god bless.
                let af_conv = convolve2_nn(&s2, &s1, self.strides, self.padding, self.dilation);

                // Σ over the given dimension. This operation needs to be made more efficient
                af_t += af_conv;
            }
            let seqs_o = [seq!(), seq!(), Seq::new(i as i32, i as i32, 1)];
            assign_seq(&mut af_out, &seqs_o, &af_t);
        }
        af_out
    }
}

impl Function for Conv2d {
    fn parents(&self) -> &[Tensor] {
        self.w_a_i.as_ref().unwrap()
    }

    fn backward(&self, grad: &RefCell<Array<f64>>) -> [Array<f64>; 2] {
        unsafe {
            let gdim = grad.as_ptr().as_ref().unwrap().dims();
            let ddim = self.w_a_i.as_ref().unwrap()[1].data.dims();
            let mut dx = constant(0., ddim);
            let mut dw = constant(0., self.w_a_i.as_ref().unwrap()[0].data.dims());
            for i in 0..self.in_channel {
                let mut af_dx = constant(0., dim4!(ddim[0], ddim[1], 1, 1));
                for j in 0..self.out_channel {
                    let seqs_w = [
                        seq!(),
                        seq!(),
                        Seq::new(i as i32, i as i32, 1),
                        Seq::new(j as i32, j as i32, 1),
                    ];
                    let seqs_d = [seq!(), seq!(), Seq::new(i as i32, i as i32, 1)];
                    let seqs_g = [seq!(), seq!(), Seq::new(j as i32, j as i32, 1)];

                    let s1 = index(&self.w_a_i.as_ref().unwrap()[0].data, &seqs_w);
                    let s2 = index(&self.w_a_i.as_ref().unwrap()[1].data, &seqs_d);
                    let s3 = index(grad.as_ptr().as_ref().unwrap(), &seqs_g);

                    // God bless arrayfire
                    let data_grad = convolve2_gradient_nn(
                        &s3,
                        &s2,
                        &s1,
                        &s3,
                        self.strides,
                        self.padding,
                        self.dilation,
                        ConvGradientType::DATA,
                    );

                    // God bless arrayfire
                    let w_grad = convolve2_gradient_nn(
                        &s3,
                        &s2,
                        &s1,
                        &s3,
                        self.strides,
                        self.padding,
                        self.dilation,
                        ConvGradientType::FILTER,
                    );
                    af_dx += data_grad;
                    assign_seq(&mut dw, &seqs_w, &w_grad);
                }
                let seqs_o = [seq!(), seq!(), Seq::new(i as i32, i as i32, 1)];
                assign_seq(&mut dx, &seqs_o, &af_dx);
            }
            [dw, dx]
        }
    }
}
