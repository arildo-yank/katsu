//! Mede o tempo de cada operação: `cargo run --release -p katu-math --example tempo`.
use katu_math::{bisect, diff, eval, integrate, parse, rk4, Vars};
use std::{hint::black_box, time::Instant};

fn medir<T>(nome: &str, vezes: u32, mut f: impl FnMut() -> T) {
    let inicio = Instant::now();
    for _ in 0..vezes {
        black_box(f());
    }
    let micros = inicio.elapsed().as_secs_f64() * 1e6 / f64::from(vezes);
    println!("{nome}: {micros:.2} µs");
}

fn main() {
    let v = Vars::new();
    let e = parse("x^2 * sin(x) + exp(x / 3)").unwrap();
    let g = parse("x^3 - 2 * x - 5").unwrap();
    let y = parse("-2 * y").unwrap();
    let c = parse("(12 + 8) * 3").unwrap();
    medir("analisar expressão", 10_000, || {
        parse(black_box("x^2 * sin(x) + exp(x / 3)"))
    });
    medir("avaliar", 100_000, || eval(black_box(&c), &v));
    medir("derivar", 10_000, || diff(black_box(&e), "x"));
    medir("integrar (tol 1e-9)", 1_000, || {
        integrate(black_box(&e), "x", 0.0, 3.0, 1e-9, &v)
    });
    medir("bisseção", 1_000, || {
        bisect(black_box(&g), "x", 2.0, 3.0, 1e-10, &v)
    });
    medir("RK4, 1.000 passos", 100, || {
        rk4(black_box(&y), "t", "y", 0.0, 1.0, 5.0, 1000, &v)
    });
}
