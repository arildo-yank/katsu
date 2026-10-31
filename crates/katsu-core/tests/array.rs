//! NdArray contra os gabaritos do NumPy em `tests/fixtures/array` (gerados por `gen.py`, sem Python no teste).

use std::path::PathBuf;

use katsu_core::{bf16_to_f32, f32_to_bf16, promote, Bf16, Compare, DType, Index, NdArray, F16};

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/array")
}

fn load(name: &str) -> NdArray {
    NdArray::read_npy(fixtures().join(name)).unwrap_or_else(|e| panic!("{name}: {e}"))
}

fn indices(spec: &str) -> Vec<Index> {
    spec.split('|')
        .map(|part| {
            let fields: Vec<&str> = part.split(':').collect();
            let number = |s: &str| (!s.is_empty()).then(|| s.parse::<isize>().unwrap());
            match fields[0] {
                "i" => Index::At(fields[1].parse().unwrap()),
                _ => Index::Range {
                    start: number(fields[1]),
                    stop: number(fields[2]),
                    step: fields.get(3).and_then(|s| number(s)).unwrap_or(1),
                },
            }
        })
        .collect()
}

fn numbers(spec: &str) -> Vec<usize> {
    spec.split(',').map(|s| s.parse().unwrap()).collect()
}

fn axis(spec: &str) -> Option<usize> {
    (spec != "none").then(|| spec.parse().unwrap())
}

fn run(op: &str, inputs: &[NdArray], args: &str) -> NdArray {
    let a = &inputs[0];
    let b = inputs.get(1);
    match op {
        "slice" => a.slice(&indices(args)).unwrap(),
        "slice_reshape" => {
            let (spec, shape) = args.split_once(';').unwrap();
            a.slice(&indices(spec))
                .unwrap()
                .reshape(&numbers(shape))
                .unwrap()
        }
        "transpose" => a.transpose(),
        "permute" => a.permute(&numbers(args)).unwrap(),
        "broadcast_to" => a.broadcast_to(&numbers(args)).unwrap(),
        "add" if args == "broadcast_row" => a
            .add(&b.unwrap().slice(&indices("r::3:")).unwrap())
            .unwrap(),
        "add" => a.add(b.unwrap()).unwrap(),
        "sub" => a.sub(b.unwrap()).unwrap(),
        "mul" => a.mul(b.unwrap()).unwrap(),
        "div" => a.div(b.unwrap()).unwrap(),
        "sub_transposed" => {
            let window = b.unwrap().slice(&indices("i:0|r::3:|r::4:")).unwrap();
            a.transpose().sub(&window).unwrap()
        }
        "lt" => a.compare(b.unwrap(), Compare::Lt).unwrap(),
        "eq" => a.compare(b.unwrap(), Compare::Eq).unwrap(),
        "sum" => a.sum(axis(args)).unwrap(),
        "mean" => a.mean(axis(args)).unwrap(),
        "max" => a.max(axis(args)).unwrap(),
        "min" => a.min(axis(args)).unwrap(),
        "astype" => a.astype(dtype(args)).unwrap(),
        "var0" => a.var(axis(args), 0).unwrap(),
        "var1" => a.var(axis(args), 1).unwrap(),
        "std0" => a.std(axis(args), 0).unwrap(),
        "std1" => a.std(axis(args), 1).unwrap(),
        "argmax" => a.argmax(axis(args)).unwrap(),
        "argmin" => a.argmin(axis(args)).unwrap(),
        "select" => a.select(b.unwrap()).unwrap(),
        "take" => {
            let (ax, list) = args.split_once(';').unwrap();
            let list: Vec<i64> = list.split(',').map(|s| s.parse().unwrap()).collect();
            a.take(&list, ax.parse().unwrap()).unwrap()
        }
        "concat" => {
            NdArray::concatenate(&inputs.iter().collect::<Vec<_>>(), args.parse().unwrap()).unwrap()
        }
        "stack" => {
            NdArray::stack(&inputs.iter().collect::<Vec<_>>(), args.parse().unwrap()).unwrap()
        }
        "identity" => a.clone(),
        other => panic!("operação desconhecida: {other}"),
    }
}

fn dtype(name: &str) -> DType {
    match name {
        "bool" => DType::Bool,
        "u8" => DType::U8,
        "i32" => DType::I32,
        "i64" => DType::I64,
        "f16" => DType::F16,
        "f32" => DType::F32,
        "f64" => DType::F64,
        other => panic!("tipo desconhecido: {other}"),
    }
}

fn close(got: f64, want: f64, tol: f64) -> bool {
    (got.is_nan() && want.is_nan())
        || got == want
        || (got - want).abs() <= tol * want.abs().max(1.0)
}

fn check(name: &str, got: &NdArray, want: &NdArray, tol: f64) {
    assert_eq!(got.dtype(), want.dtype(), "{name}: tipo");
    assert_eq!(got.shape(), want.shape(), "{name}: forma");
    match want.dtype() {
        DType::Bool => assert_eq!(
            got.to_vec::<bool>().unwrap(),
            want.to_vec::<bool>().unwrap(),
            "{name}"
        ),
        DType::U8 | DType::I32 | DType::I64 => {
            assert_eq!(got.values_as::<i64>(), want.values_as::<i64>(), "{name}")
        }
        DType::F16 | DType::Bf16 | DType::F32 | DType::F64 => {
            let (g, w): (Vec<f64>, Vec<f64>) = (got.values_as(), want.values_as());
            for (i, (&x, &y)) in g.iter().zip(&w).enumerate() {
                assert!(
                    close(x, y, tol),
                    "{name}[{i}]: {x} contra {y} (tolerância {tol})"
                );
            }
        }
    }
}

#[test]
fn matches_numpy_fixtures() {
    let text = std::fs::read_to_string(fixtures().join("cases.txt")).unwrap();
    let mut count = 0;
    for line in text
        .lines()
        .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
    {
        let fields: Vec<&str> = line.split(" ; ").collect();
        let [name, op, inputs, args, expected, tol] = fields[..] else {
            panic!("linha malformada: {line}");
        };
        let inputs: Vec<NdArray> = inputs.split(',').map(load).collect();
        let got = run(op, &inputs, args);
        check(name, &got, &load(expected), tol.parse().unwrap());
        count += 1;
    }
    assert_eq!(
        count, 106,
        "o número de casos mudou: rode gen.py e atualize este teste"
    );
}

#[test]
fn views_do_not_copy() {
    let cube = load("cube_f64.npy");
    let view = cube.slice(&indices("r:::-1|i:2|r:1::2")).unwrap();
    assert!(view.shares_memory(&cube));
    assert!(cube.transpose().shares_memory(&cube));
    assert!(cube
        .broadcast_to(&[2, 3, 4, 5])
        .unwrap()
        .shares_memory(&cube));
    assert!(cube.reshape(&[12, 5]).unwrap().shares_memory(&cube));
    assert!(!view.is_contiguous());
    assert!(
        !view.reshape(&[view.len()]).unwrap().shares_memory(&cube),
        "view não contígua precisa de cópia"
    );
    assert_eq!(cube.strides(), &[20, 5, 1]);
    assert_eq!(view.strides(), &[-20, 2]);
}

#[test]
fn writes_the_same_bytes_as_numpy() {
    for name in [
        "cube_f64.npy",
        "mat_i64.npy",
        "mask_a.npy",
        "row_f32.npy",
        "scalar_f64.npy",
        "u8a.npy",
        "i32a.npy",
        "f16a.npy",
    ] {
        let bytes = std::fs::read(fixtures().join(name)).unwrap();
        let array = NdArray::from_npy_bytes(&bytes).unwrap();
        assert_eq!(array.to_npy_bytes().unwrap(), bytes, "{name}");
    }
}

#[test]
fn refuses_what_numpy_refuses_or_leaves_undefined() {
    let mask = load("mask_a.npy");
    assert!(mask.sub(&mask).is_err(), "bool − bool é erro no NumPy");
    let nan = NdArray::from_vec(vec![f64::NAN, 1.0], &[2]).unwrap();
    assert!(
        nan.astype(DType::I64).is_err(),
        "NaN para inteiro é indefinido no NumPy; aqui é erro"
    );
    let cube = load("cube_f64.npy");
    assert!(
        cube.add(&load("row_f32.npy")).is_err(),
        "formas (3, 4, 5) e (4,) não combinam"
    );
    assert!(cube.slice(&indices("i:3")).is_err());
    assert!(cube.slice(&indices("r:::0")).is_err());
    assert!(cube.sum(Some(3)).is_err());
    let empty = NdArray::zeros(DType::F64, &[0, 3]).unwrap();
    assert!(empty.max(Some(0)).is_err());
    assert_eq!(
        empty.sum(Some(0)).unwrap().to_vec::<f64>().unwrap(),
        vec![0.0; 3]
    );
    assert!(empty.mean(None).unwrap().to_vec::<f64>().unwrap()[0].is_nan());
}

#[test]
fn promotion_matches_numpy_table() {
    let text = std::fs::read_to_string(fixtures().join("promotion.txt")).unwrap();
    let mut rows = 0;
    for line in text.lines() {
        let names: Vec<&str> = line.split_whitespace().collect();
        let want = match names[2] {
            "bool_" => "bool",
            "uint8" => "u8",
            "int32" => "i32",
            "int64" => "i64",
            "float16" => "f16",
            "float32" => "f32",
            "float64" => "f64",
            other => other,
        };
        assert_eq!(
            promote(dtype(names[0]), dtype(names[1])),
            dtype(want),
            "{line}"
        );
        rows += 1;
    }
    assert_eq!(rows, 49);
    assert_eq!(promote(DType::F16, DType::Bf16), DType::F32);
    assert_eq!(promote(DType::U8, DType::Bf16), DType::F32);
    assert_eq!(promote(DType::Bf16, DType::F32), DType::F32);
}

#[test]
fn every_f16_survives_the_round_trip() {
    for bits in 0..=u16::MAX {
        let value = F16(bits).to_f64();
        if value.is_nan() {
            assert!(F16::from_f64(value).to_f64().is_nan());
        } else {
            assert_eq!(F16::from_f64(value), F16(bits), "{bits:#06x}");
        }
    }
}

#[test]
fn bf16_matches_the_training_engine_conversion() {
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    let specials = [
        0.0f32,
        -0.0,
        1.0,
        f32::MAX,
        f32::MIN_POSITIVE,
        1e-40,
        -1e-45,
        f32::INFINITY,
        3.0e38,
    ];
    let random = (0..200_000).map(|_| {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        f32::from_bits((state >> 32) as u32)
    });
    for x in specials.into_iter().chain(random).filter(|x| !x.is_nan()) {
        let ours = Bf16::from_f64(x as f64);
        assert_eq!(ours.0, f32_to_bf16(x), "{x:e}");
        assert_eq!(ours.to_f64(), bf16_to_f32(ours.0) as f64);
    }
    let a = NdArray::from_vec(vec![Bf16::from_f64(1.5), Bf16::from_f64(-2.0)], &[2]).unwrap();
    let sum = a.add(&a).unwrap();
    assert_eq!(sum.dtype(), DType::Bf16);
    assert_eq!(sum.values_as::<f64>(), vec![3.0, -4.0]);
    assert!(sum.to_npy_bytes().is_err(), "bf16 não existe no .npy");
}
