//! Aritmética exata contra os gabaritos de `tests/fixtures/exact` (gerados por `gen.py`, sem Python no teste).
use katu_math::{exact_eval, Int, Rational};

fn lines(name: &str) -> Vec<Vec<String>> {
    let path = format!("{}/tests/fixtures/exact/{name}", env!("CARGO_MANIFEST_DIR"));
    std::fs::read_to_string(&path)
        .unwrap_or_else(|why| panic!("{path}: {why}"))
        .lines()
        .map(|line| line.split('\t').map(str::to_owned).collect())
        .collect()
}

fn int(text: &str) -> Int {
    Int::parse(text).unwrap()
}

fn fraction(text: &str) -> Rational {
    Rational::parse(text).unwrap()
}

/// O texto de `float.hex()` do Python: sinal, mantissa hexadecimal e expoente binário.
fn from_hex(text: &str) -> f64 {
    let (neg, rest) = match text.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, text),
    };
    let rest = rest.strip_prefix("0x").unwrap();
    let (mantissa, exponent) = rest.split_once('p').unwrap();
    let (whole, fraction) = mantissa.split_once('.').unwrap_or((mantissa, ""));
    let mut bits = u64::from_str_radix(whole, 16).unwrap();
    for digit in fraction.bytes() {
        bits = (bits << 4) | u64::from((digit as char).to_digit(16).unwrap());
    }
    let mut power = exponent.parse::<i64>().unwrap() - 4 * fraction.len() as i64;
    let mut value = bits as f64;
    while power != 0 {
        let step = power.clamp(-1000, 1000);
        value *= 2f64.powi(step as i32);
        power -= step;
    }
    if neg {
        -value
    } else {
        value
    }
}

#[test]
fn integers_match_python() {
    let cases = lines("int_ops.txt");
    assert_eq!(cases.len(), 400);
    for case in cases {
        let (a, b) = (int(&case[0]), int(&case[1]));
        assert_eq!(a.to_string(), case[0]);
        assert_eq!(a.add(&b).unwrap().to_string(), case[2], "{a} + {b}");
        assert_eq!(a.sub(&b).unwrap().to_string(), case[3], "{a} - {b}");
        assert_eq!(a.mul(&b).unwrap().to_string(), case[4], "{a} * {b}");
        let (q, r) = a.div_rem(&b).unwrap();
        assert_eq!(q.to_string(), case[5], "{a} / {b}");
        assert_eq!(r.to_string(), case[6], "{a} % {b}");
    }
}

#[test]
fn powers_match_python() {
    let cases = lines("int_pow.txt");
    assert_eq!(cases.len(), 60);
    for case in cases {
        let power = int(&case[0]).pow(case[1].parse().unwrap()).unwrap();
        assert_eq!(power.to_string(), case[2], "{}^{}", case[0], case[1]);
    }
}

#[test]
fn rationals_match_python() {
    let cases = lines("rational_ops.txt");
    assert_eq!(cases.len(), 300);
    let text = |r: Rational| format!("{}/{}", r.numerator(), r.denominator());
    for case in cases {
        let (a, b) = (fraction(&case[0]), fraction(&case[1]));
        assert_eq!(text(a.clone()), case[0], "forma reduzida");
        assert_eq!(text(a.add(&b).unwrap()), case[2], "{a} + {b}");
        assert_eq!(text(a.sub(&b).unwrap()), case[3], "{a} - {b}");
        assert_eq!(text(a.mul(&b).unwrap()), case[4], "{a} * {b}");
        assert_eq!(text(a.div(&b).unwrap()), case[5], "{a} / {b}");
    }
}

/// Bit a bit: inclui subnormais e valores no meio de dois f64 vizinhos.
#[test]
fn float_conversion_matches_python() {
    let cases = lines("rational_float.txt");
    assert!(cases.len() > 500, "{}", cases.len());
    for case in cases {
        let got = fraction(&case[0]).to_f64().unwrap();
        let want = from_hex(&case[1]);
        assert_eq!(
            got.to_bits(),
            want.to_bits(),
            "{} -> {got:e} != {want:e}",
            case[0]
        );
    }
}

#[test]
fn expressions_match_python() {
    let cases = lines("eval.txt");
    assert_eq!(cases.len(), 300);
    for case in cases {
        let got = exact_eval(&case[0]).unwrap_or_else(|why| panic!("{}: {why}", case[0]));
        let text = format!("{}/{}", got.numerator(), got.denominator());
        assert_eq!(text, case[1], "{}", case[0]);
    }
}
