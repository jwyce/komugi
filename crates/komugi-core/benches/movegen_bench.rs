use criterion::{black_box, criterion_group, criterion_main, Criterion};
use komugi_core::fen::{parse_fen, ADVANCED_POSITION, BEGINNER_POSITION, INTERMEDIATE_POSITION};
use komugi_core::movegen::generate_all_legal_moves_in_state;

fn movegen_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("movegen");
    group.sample_size(100);

    group.bench_function("beginner_position", |b| {
        b.iter(|| {
            let state = parse_fen(black_box(BEGINNER_POSITION)).expect("parse");
            generate_all_legal_moves_in_state(&state)
        })
    });

    group.bench_function("intermediate_position", |b| {
        b.iter(|| {
            let state = parse_fen(black_box(INTERMEDIATE_POSITION)).expect("parse");
            generate_all_legal_moves_in_state(&state)
        })
    });

    group.bench_function("advanced_position", |b| {
        b.iter(|| {
            let state = parse_fen(black_box(ADVANCED_POSITION)).expect("parse");
            generate_all_legal_moves_in_state(&state)
        })
    });

    group.finish();
}

criterion_group!(benches, movegen_benchmarks);
criterion_main!(benches);
