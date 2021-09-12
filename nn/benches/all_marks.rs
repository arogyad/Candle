#![allow(unused)]
use arrayfire::{constant, dim4, randn, Array};
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use nn::conv2d::Conv2d;
use rand::{self, thread_rng, Rng};
fn conv(
    in_channel: u64,
    out_channel: u64,
    kernel_size: [u64; 2],
    strides: [u64; 2],
    padding: [u64; 2],
) {
    let mut c = Conv2d::new(in_channel, out_channel, kernel_size, strides, padding);
    let data: Array<f64> = randn(dim4!(256, 256, in_channel, 1));
    print!("\n");
    println!(
        "In channel: {} Out channel: {}, Kernel_size:{:?} Stride: {:?} Padding: {:?}",
        in_channel, out_channel, kernel_size, strides, padding
    );
    c.apply(&data);
}

fn criterion_benchmark(c: &mut Criterion) {
    let mut rng = thread_rng();
    c.bench_function("conv random", |b| {
        b.iter(|| {
            conv(
                black_box(rng.gen_range(3..=5)),
                black_box(rng.gen_range(8..=12)),
                black_box([rng.gen_range(1..=5), rng.gen_range(1..=5)]),
                black_box([rng.gen_range(1..=3), rng.gen_range(1..=3)]),
                black_box([rng.gen_range(1..=4), rng.gen_range(1..=4)]),
            )
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
