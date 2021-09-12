use arrayfire::{assign_seq, constant, dim4, dot, flat, index, randn, seq, Array, Dim4, Seq};

// Convolution as described in The Deep Learning Book. No kernel flipping, no bias or dilation
pub struct Conv2d {
    weight: Option<Array<f64>>,
    // input: Array<f64>,
    in_channel: u64,
    out_channel: u64,
    kernel_size: [u64; 2],
    strides: [u64; 2],
    padding: [u64; 2],
}

impl Conv2d {
    pub fn new(
        in_channel: u64,
        out_channel: u64,
        kernel_size: [u64; 2],
        strides: [u64; 2],
        padding: [u64; 2],
    ) -> Self {
        Self {
            weight: None,
            in_channel,
            out_channel,
            kernel_size,
            strides,
            padding,
        }
    }

    pub fn apply(&mut self, data: &Array<f64>) -> Array<f64> {
        // data -> [H x W x C x O]
        let dim = data.dims();
        if !&self.weight.is_some() {
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

    // dim -> dimension of initial data
    fn forward(&mut self, data: &Array<f64>, dim: &Dim4) -> Array<f64> {
        // Out Shape
        let r_d = self.make_padding(data);
        let ox =
            ((dim[0] + 2 * self.padding[0] - (self.kernel_size[0] - 1) - 1) / self.strides[0]) + 1;
        let oy =
            ((dim[1] + 2 * self.padding[1] - (self.kernel_size[1] - 1) - 1) / self.strides[1]) + 1;

        // The array to be returned
        let mut out: Array<f64> = constant(0.0, dim4!(ox, oy, self.out_channel, dim[3]));
        // i -> num of output channels
        // j -> num of in channels
        // Each Operation -> input (*) Weight[..,.., i, j]
        // Out -> [ox, oy, out_channel, N]
        for i in 0..self.out_channel {
            let mut to_be_added = constant(0., dim4!(ox, oy, 1, 1));
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

                // this creates the 2d array of output
                let seqs_o = [seq!(), seq!(), Seq::new(i as i32, i as i32, 1)];

                // This is the 2dimensional weight
                let s1 = index(self.weight.as_ref().unwrap(), &seqs_w);
                // This takes out 1 dimension from data
                let s2 = index(&r_d, &seqs_d);

                let mut s3 = index(&out, &seqs_o);

                // This does the convolution operation and the moving around in the image space
                Conv2d::do_conv(&s2, &s1, self, &mut s3, s2.dims());

                // Σ over the given dimension. This operation needs to be made more efficient
                to_be_added += s3;
            }
            let seqs_o = [seq!(), seq!(), Seq::new(i as i32, i as i32, 1)];
            assign_seq(&mut out, &seqs_o, &to_be_added);
        }
        out
    }

    // input -> 2 dimensional input array of data of size [H x W x C[j] x N]
    // filter -> 2 dimensional filter array of weight size of [ Kernel_Size ]
    // s -> current thing, to get padding and stuff
    // out -> Two dimensional array from the output, size [ox X oy]
    // dim: dimension of data to keep track of breaking condition
    // current: the current location of the kernal
    pub fn do_conv(
        input: &Array<f64>,
        filter: &Array<f64>,
        s: &Conv2d,
        out: &mut Array<f64>,
        dim: Dim4,
    ) {
        // Current -> current position of the filter in the input array
        let mut current = [0; 2];
        // out_current -> current position of output in the output array
        let mut out_current = [0; 2];

        // We check if the 1st dimension has exhausted or not. If exhausted we quit, else we
        // continue. Exhaused as in, the kernel_size[0] + the current position from where we
        // perform the conv operation is greater than the 0th dimension. We don't have to check the
        // 1st dimension as it is checked by the match statement below.
        while s.kernel_size[0] + current[0] <= dim[0] {
            // This sequence is used to form the 2dimensional array from the input which is the
            // size of the kernels. Arrayfire is weird in this regard. The sequence [0,1,1] takes 2
            // rows the 0th and the 1th unlike anyother library.
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

            // The output array's position must also be managed, as the kernel's position in the
            // input array changes, the output position changes as well.
            let out_c_s = [
                Seq::new(out_current[0], out_current[0], 1),
                Seq::new(out_current[1], out_current[1], 1),
            ];

            // From the right, we perform the conv operation on the 2d array of input and filter.
            // Then the result is assigned to the corresponding position in out.
            assign_seq(out, &out_c_s, &Conv2d::_do_conv(&index(input, &s1), filter));

            // This checks if we have/are about to jump out of the bound in the 1st dimension and
            // performs the action accordingly.
            match current[1] + s.kernel_size[1] - 1 + s.strides[1] {
                x if x >= dim[1] => {
                    current = [current[0] + s.strides[0], 0];
                    out_current = [1 + out_current[0], 0];
                }
                _ => {
                    current[1] += s.strides[1];
                    out_current[1] += 1;
                }
            }
        }
    }

    // Perform vector product
    pub fn _do_conv(data_s: &Array<f64>, filter: &Array<f64>) -> Array<f64> {
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
                dim[2],
                dim[3]
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
        let s2 = Seq::new(0, dim[2] as u32 - 1, 1);
        assign_seq(&mut _padded_data, &[s0, s1, s2], data);
        _padded_data
    }
}
