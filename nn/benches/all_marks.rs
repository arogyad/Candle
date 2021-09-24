use arrayfire::{constant, dim4};
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use nn::conv2d::Conv2d;
use nn::tensor::Tensor;
use rand::{self, Rng};

fn conv_test(in_channel: u64, out_channel: u64) {
    let data = Tensor::new(constant(1., dim4!(5, 5, in_channel, 1)), None);
    let mut c = Conv2d::new(in_channel, out_channel, [2, 3], [3, 1], [1, 1], [1, 1]);
    let _o = c.apply(data);
}

fn criterion_benchmark(c: &mut Criterion) {
    let mut rng = rand::thread_rng();
    c.bench_function("conv all_rand", |b| {
        b.iter(|| {
            conv_test(
                black_box(rng.gen_range(3..5)),
                black_box(rng.gen_range(5..6)),
            )
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
