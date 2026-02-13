use criterion::{black_box, criterion_group, criterion_main, Criterion};
use komugi_core::fen::{
    apply_move_to_fen, parse_fen, ADVANCED_POSITION, BEGINNER_POSITION, INTERMEDIATE_POSITION,
};
use komugi_core::movegen::generate_all_legal_moves_in_state;

fn perft(fen: &str, depth: u8) -> u64 {
    if depth == 0 {
        return 1;
    }

    let state = parse_fen(fen).expect("valid fen");
    let moves = generate_all_legal_moves_in_state(&state);
    if depth == 1 {
        return moves.len() as u64;
    }

    let mut nodes = 0u64;
    for mv in moves {
        let next = apply_move_to_fen(fen, &mv).expect("apply move");
        nodes += perft(&next, depth - 1);
    }
    nodes
}

fn perft_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("perft");
    group.sample_size(10);

    group.bench_function("beginner_depth_1", |b| {
        b.iter(|| perft(black_box(BEGINNER_POSITION), black_box(1)))
    });

    group.bench_function("beginner_depth_2", |b| {
        b.iter(|| perft(black_box(BEGINNER_POSITION), black_box(2)))
    });

    group.bench_function("intermediate_depth_1", |b| {
        b.iter(|| perft(black_box(INTERMEDIATE_POSITION), black_box(1)))
    });

    group.bench_function("intermediate_depth_2", |b| {
        b.iter(|| perft(black_box(INTERMEDIATE_POSITION), black_box(2)))
    });

    group.bench_function("advanced_depth_1", |b| {
        b.iter(|| perft(black_box(ADVANCED_POSITION), black_box(1)))
    });

    group.bench_function("advanced_depth_2", |b| {
        b.iter(|| perft(black_box(ADVANCED_POSITION), black_box(2)))
    });

    group.finish();
}

criterion_group!(benches, perft_benchmarks);
criterion_main!(benches);
