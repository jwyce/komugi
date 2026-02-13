use criterion::{black_box, criterion_group, criterion_main, Criterion};
use komugi_core::fen::{parse_fen, ADVANCED_POSITION, BEGINNER_POSITION, INTERMEDIATE_POSITION};

fn eval_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("eval");
    group.sample_size(100);

    group.bench_function("parse_beginner_position", |b| {
        b.iter(|| parse_fen(black_box(BEGINNER_POSITION)))
    });

    group.bench_function("parse_intermediate_position", |b| {
        b.iter(|| parse_fen(black_box(INTERMEDIATE_POSITION)))
    });

    group.bench_function("parse_advanced_position", |b| {
        b.iter(|| parse_fen(black_box(ADVANCED_POSITION)))
    });

    group.finish();
}

criterion_group!(benches, eval_benchmarks);
criterion_main!(benches);
