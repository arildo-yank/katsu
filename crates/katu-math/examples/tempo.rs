//! Mede o tempo de cada operação: `cargo run --release -p katu-math --example tempo`.
use katu_math::{bisect, diff, eval, exact_eval, integrate, parse, rk4, Int, Rational, Vars};
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

    // Aritmética exata: caso comum e o pior caso dentro de MAX_BITS.
    medir("exato: 0.1 + 0.2", 10_000, || {
        exact_eval("0.1 + 0.2").unwrap()
    });
    medir("exato: 2^100", 10_000, || exact_eval("2^100").unwrap());
    medir("exato: recusa de 2^(10^9)", 10_000, || {
        exact_eval("2^(10^9)").is_err()
    });
    let grande = Int::from(3).pow(165_000).unwrap();
    let outro = Int::from(7).pow(93_000).unwrap();
    println!("operandos: {} e {} bits", grande.bits(), outro.bits());
    medir("exato: soma no limite", 100, || grande.add(&outro).unwrap());
    medir("exato: produto de metades", 3, || {
        Int::from(3)
            .pow(82_000)
            .unwrap()
            .mul(&Int::from(7).pow(46_000).unwrap())
            .unwrap()
    });
    medir("exato: divisão", 3, || grande.div_rem(&outro).unwrap());
    medir("exato: mdc, pior caso prático", 1, || grande.gcd(&outro));
    medir("exato: racional reduzido", 1, || {
        Rational::new(grande.clone(), outro.clone()).unwrap()
    });
    medir("exato: texto decimal", 1, || grande.to_string());
    medir("exato: para f64", 100, || {
        Rational::new(Int::from(1), Int::from(3))
            .unwrap()
            .to_f64()
            .unwrap()
    });
}
