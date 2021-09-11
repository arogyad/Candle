#![allow(unused)]
use arrayfire::{
    assign_seq, constant, dim4, dot, flat, index, print, randn, seq, Array, Dim4, Seq,
};

// Convolution as described in The Deep Learning Book. No kernel flipping, no bias or dilation
pub struct Conv2d {
    weight: Option<Array<f64>>,
    input: Array<f64>,
    in_channel: u64,
    out_channel: u64,
    kernel_size: [u64; 2],
    strides: [u64; 2],
    padding: [u64; 2],
}

impl Conv2d {
    fn apply(&mut self, data: &Array<f64>) -> Array<f64> {
        // data -> [H x W x C x O]
        let dim = data.dims();
        if let Some(w) = &self.weight {
        } else {
            // Weight size -> [KS[0] x KS[1] x C x O]
            self.weight = Some(randn(dim4!(
                self.kernel_size[0],
                self.kernel_size[1],
                self.in_channel,
                self.out_channel
            )));
        }
        self.forward(data, &dim)
    }

    fn forward(&mut self, data: &Array<f64>, dim: &Dim4) -> Array<f64> {
        // Out Shape
        let r_d = self.make_padding(data);
        let ox =
            ((dim[0] + 2 * self.padding[0] - (self.kernel_size[0] - 1) - 1) / self.strides[0]) + 1;
        let oy =
            ((dim[1] + 2 * self.padding[1] - (self.kernel_size[1] - 1) - 1) / self.strides[1]) + 1;
        let mut out: Array<f64> = constant(0.0, dim4!(ox, oy, self.out_channel, dim[4]));
        // i -> num of output channels
        // j -> num of in channels
        // Each Operation -> input (*) Weight[..,.., i, j]
        // Out -> [ox, oy, out_channel, N]
        for i in 0..self.out_channel {
            for j in 0..self.in_channel {
                // Weight[i][j]
                // This creates a 2d array of weight
                let seqs_w = [
                    seq!(),
                    seq!(),
                    Seq::new(j as i32, (j + 1) as i32, 1),
                    Seq::new(i as i32, (i + 1) as i32, 1),
                ];
                // This creates the 2d array of data
                let seqs_d = [seq!(), seq!(), Seq::new(j as i32, (j + 1) as i32, 1)];

                // this creates the 2d array of output
                let seqs_o = [seq!(), seq!(), Seq::new(i as i32, (i + 1) as i32, 1)];

                // This is the 2dimensional weight
                let s1 = index(self.weight.as_ref().unwrap(), &seqs_w);

                // This takes out 1 dimension from data
                let s2 = index(&r_d, &seqs_d);

                let d_r_d = s2.dims();

                let mut s3 = index(&out, &seqs_o);

                Conv2d::do_conv(&s2, &s1, self, &mut s3, d_r_d);
            }
            // Change the dimension value of out here left to do
        }
        constant(1., dim4!(1, 1, 1, 1))
    }

    // input -> 2 dimensional input array of data of size [H x W x C[j] x N]
    // filter -> 2 dimensional filter array of weight size of [ Kernel_Size ]
    // s -> current thing, to get padding and stuff
    // out -> Two dimensional array from the output, size [ox X oy]
    // dim: dimension of data to keep track of breaking condition
    // current: the current location of the kernal
    fn do_conv(
        input: &Array<f64>,
        filter: &Array<f64>,
        s: &Conv2d,
        out: &mut Array<f64>,
        dim: Dim4,
    ) {
        let mut current = [0; 2];
        let mut out_current = [0; 2];
        while !((s.kernel_size[0] + current[0] - 1) > dim[0]
            || (s.kernel_size[1] + current[1] - 1) > dim[1])
        {
            let s1 = [
                Seq::new(
                    current[0] as u32,
                    (current[0] + s.kernel_size[0] - 1) as u32,
                    1,
                ),
                Seq::new(
                    current[1] as u32,
                    (current[1] + s.kernel_size[1] - 1) as u32,
                    1,
                ),
            ];

            let out_c_s = [
                Seq::new(out_current[0], out_current[0], 1),
                Seq::new(out_current[1], out_current[1], 1),
            ];
            assign_seq(out, &out_c_s, &Conv2d::_do_conv(&index(input, &s1), filter));
            match current[0] + s.kernel_size[0] - 1 + s.strides[0] {
                x if x > dim[3] => {
                    current = [0, current[1] + s.strides[1]];
                    out_current = [0, 1 + out_current[1]];
                }
                _ => {
                    current[0] += s.strides[0];
                    out_current[0] += 1;
                }
            }
        }
    }

    // Perform vector product
    fn _do_conv(data_s: &Array<f64>, filter: &Array<f64>) -> Array<f64> {
        dot(
            &flat(data_s),
            &flat(filter),
            arrayfire::MatProp::NONE,
            arrayfire::MatProp::NONE,
        )
    }

    // Creating of padding, currently only 0 padding.
    fn make_padding(&self, data: &Array<f64>) -> Array<f64> {
        let dim = data.dims();
        let mut _padded_data = constant(
            0.,
            dim4!(
                dim[0] + 2 * self.padding[0],
                dim[1] + 2 * self.padding[1],
                dim[3],
                dim[4]
            ),
        );
        let s0 = Seq::new(
            self.padding[0] as u32,
            (dim[0] + self.padding[0] - 1) as u32,
            1,
        );
        let s1 = Seq::new(
            self.padding[1] as u32,
            (self.padding[1] + dim[1] - 1) as u32,
            1,
        );
        assign_seq(&mut _padded_data, &[s0, s1], data);
        _padded_data
    }
}
