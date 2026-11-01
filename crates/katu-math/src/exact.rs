//! Aritmética exata, sem dependências: inteiro de tamanho arbitrário ([`Int`]) e racional ([`Rational`]).
//! Só cobre o que fecha em racionais: soma, produto, divisão e potência inteira. Raiz, pi, seno e logaritmo
//! continuam em f64 e são recusados aqui. Tamanho arbitrário não é tamanho infinito: passar de [`MAX_BITS`]
//! é recusa, estimada antes de calcular sempre que dá. Contrato em `docs/KATU_CONTRATO_RESULTADOS.md`.
use crate::{fail, ErrorCode, Func, MathError};
use std::{cmp::Ordering, fmt};

/// Limite por inteiro. Ajustável depois de medir; ver o relatório.
pub const MAX_BITS: u64 = 262_144;
const BASE_BITS: u32 = 32;
const DECIMAL_CHUNK: u32 = 1_000_000_000;

fn too_big<T>() -> Result<T, MathError> {
    fail(
        ErrorCode::LimitSize,
        format!("o número exato passaria de {MAX_BITS} bits"),
    )
}

// Magnitudes: dígitos em base 2^32, do menos para o mais significativo, sem zero no fim.

fn trim(mut a: Vec<u32>) -> Vec<u32> {
    while a.last() == Some(&0) {
        a.pop();
    }
    a
}

fn mag_bits(a: &[u32]) -> u64 {
    match a.last() {
        None => 0,
        Some(top) => {
            (a.len() as u64 - 1) * u64::from(BASE_BITS) + u64::from(32 - top.leading_zeros())
        }
    }
}

fn mag_cmp(a: &[u32], b: &[u32]) -> Ordering {
    a.len()
        .cmp(&b.len())
        .then_with(|| a.iter().rev().cmp(b.iter().rev()))
}

fn mag_add(a: &[u32], b: &[u32]) -> Vec<u32> {
    let (long, short) = if a.len() >= b.len() { (a, b) } else { (b, a) };
    let mut out = Vec::with_capacity(long.len() + 1);
    let mut carry = 0u64;
    for (i, digit) in long.iter().enumerate() {
        let sum = u64::from(*digit) + u64::from(short.get(i).copied().unwrap_or(0)) + carry;
        out.push(sum as u32);
        carry = sum >> BASE_BITS;
    }
    if carry > 0 {
        out.push(carry as u32);
    }
    out
}

/// a - b, com a >= b.
fn mag_sub(a: &[u32], b: &[u32]) -> Vec<u32> {
    let mut out = Vec::with_capacity(a.len());
    let mut borrow = 0i64;
    for (i, digit) in a.iter().enumerate() {
        let mut diff = i64::from(*digit) - i64::from(b.get(i).copied().unwrap_or(0)) - borrow;
        borrow = i64::from(diff < 0);
        if diff < 0 {
            diff += 1 << BASE_BITS;
        }
        out.push(diff as u32);
    }
    trim(out)
}

fn mag_mul(a: &[u32], b: &[u32]) -> Vec<u32> {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }
    let mut out = vec![0u32; a.len() + b.len()];
    for (i, x) in a.iter().enumerate() {
        let mut carry = 0u64;
        for (j, y) in b.iter().enumerate() {
            let cur = u64::from(out[i + j]) + u64::from(*x) * u64::from(*y) + carry;
            out[i + j] = cur as u32;
            carry = cur >> BASE_BITS;
        }
        out[i + b.len()] = carry as u32;
    }
    trim(out)
}

fn mag_shl(a: &[u32], bits: u64) -> Vec<u32> {
    if a.is_empty() {
        return Vec::new();
    }
    let words = (bits / u64::from(BASE_BITS)) as usize;
    let shift = (bits % u64::from(BASE_BITS)) as u32;
    let mut out = vec![0u32; words];
    let mut carry = 0u32;
    for digit in a {
        let wide = (u64::from(*digit) << shift) | u64::from(carry);
        out.push(wide as u32);
        carry = (wide >> BASE_BITS) as u32;
    }
    out.push(carry);
    trim(out)
}

fn mag_divrem_small(a: &[u32], d: u32) -> (Vec<u32>, u32) {
    let mut out = vec![0u32; a.len()];
    let mut rem = 0u64;
    for i in (0..a.len()).rev() {
        let cur = (rem << BASE_BITS) | u64::from(a[i]);
        out[i] = (cur / u64::from(d)) as u32;
        rem = cur % u64::from(d);
    }
    (trim(out), rem as u32)
}

/// Divisão longa de Knuth (TAOCP vol. 2, algoritmo D), com divisor de 2 dígitos ou mais.
fn mag_divrem(a: &[u32], b: &[u32]) -> (Vec<u32>, Vec<u32>) {
    if mag_cmp(a, b) == Ordering::Less {
        return (Vec::new(), a.to_vec());
    }
    if b.len() == 1 {
        let (q, r) = mag_divrem_small(a, b[0]);
        return (q, if r == 0 { Vec::new() } else { vec![r] });
    }
    let shift = b[b.len() - 1].leading_zeros();
    let v = mag_shl(b, u64::from(shift));
    let mut u = mag_shl(a, u64::from(shift));
    u.resize(a.len() + 1, 0);
    let n = v.len();
    let m = u.len() - n - 1;
    let base = 1u64 << BASE_BITS;
    let mut q = vec![0u32; m + 1];
    for j in (0..=m).rev() {
        let top = (u64::from(u[j + n]) << BASE_BITS) | u64::from(u[j + n - 1]);
        let mut qhat = top / u64::from(v[n - 1]);
        let mut rhat = top % u64::from(v[n - 1]);
        while qhat >= base
            || qhat * u64::from(v[n - 2]) > (rhat << BASE_BITS) + u64::from(u[j + n - 2])
        {
            qhat -= 1;
            rhat += u64::from(v[n - 1]);
            if rhat >= base {
                break;
            }
        }
        let mut borrow = 0i64;
        let mut carry = 0u64;
        for i in 0..n {
            let product = qhat * u64::from(v[i]) + carry;
            carry = product >> BASE_BITS;
            let diff = i64::from(u[i + j]) - borrow - (product & (base - 1)) as i64;
            u[i + j] = diff as u32;
            borrow = i64::from(diff < 0);
        }
        let diff = i64::from(u[j + n]) - borrow - carry as i64;
        u[j + n] = diff as u32;
        // qhat ainda pode passar por um: soma o divisor de volta.
        if diff < 0 {
            qhat -= 1;
            let mut carry = 0u64;
            for i in 0..n {
                let sum = u64::from(u[i + j]) + u64::from(v[i]) + carry;
                u[i + j] = sum as u32;
                carry = sum >> BASE_BITS;
            }
            u[j + n] = (u64::from(u[j + n]) + carry) as u32;
        }
        q[j] = qhat as u32;
    }
    u.truncate(n);
    let mut r = trim(u);
    if shift > 0 && !r.is_empty() {
        let mut carry = 0u32;
        for digit in r.iter_mut().rev() {
            let next = *digit << (BASE_BITS - shift);
            *digit = (*digit >> shift) | carry;
            carry = next;
        }
        r = trim(r);
    }
    (trim(q), r)
}

/// Inteiro de tamanho arbitrário, até [`MAX_BITS`]. Zero nunca é negativo.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Int {
    neg: bool,
    mag: Vec<u32>,
}

impl Int {
    fn build(neg: bool, mag: Vec<u32>) -> Result<Self, MathError> {
        if mag_bits(&mag) > MAX_BITS {
            return too_big();
        }
        Ok(Self {
            neg: neg && !mag.is_empty(),
            mag,
        })
    }

    pub fn zero() -> Self {
        Self {
            neg: false,
            mag: Vec::new(),
        }
    }

    pub fn one() -> Self {
        Self::from(1)
    }

    /// Decimal, com sinal opcional.
    pub fn parse(text: &str) -> Result<Self, MathError> {
        let (neg, digits) = match text.strip_prefix('-') {
            Some(rest) => (true, rest),
            None => (false, text.strip_prefix('+').unwrap_or(text)),
        };
        if digits.is_empty() || !digits.bytes().all(|c| c.is_ascii_digit()) {
            return fail(ErrorCode::ParseError, format!("inteiro inválido: {text}"));
        }
        // Cada dígito decimal passa de 3 bits: corta antes de gastar tempo.
        if digits.len() as u64 > MAX_BITS / 3 {
            return too_big();
        }
        let mut mag: Vec<u32> = Vec::new();
        for chunk in digits.as_bytes().rchunks(9).rev() {
            let value = chunk.iter().fold(0u32, |v, c| v * 10 + u32::from(c - b'0'));
            let mut carry = u64::from(value);
            let width = 10u64.pow(chunk.len() as u32);
            for digit in &mut mag {
                let cur = u64::from(*digit) * width + carry;
                *digit = cur as u32;
                carry = cur >> BASE_BITS;
            }
            if carry > 0 {
                mag.push(carry as u32);
            }
        }
        Self::build(neg, trim(mag))
    }

    pub fn is_zero(&self) -> bool {
        self.mag.is_empty()
    }

    pub fn is_negative(&self) -> bool {
        self.neg
    }

    pub fn bits(&self) -> u64 {
        mag_bits(&self.mag)
    }

    pub fn neg(&self) -> Self {
        Self {
            neg: !self.neg && !self.is_zero(),
            mag: self.mag.clone(),
        }
    }

    pub fn abs(&self) -> Self {
        Self {
            neg: false,
            mag: self.mag.clone(),
        }
    }

    fn is_unit(&self) -> bool {
        self.mag == [1]
    }

    pub fn add(&self, other: &Self) -> Result<Self, MathError> {
        if self.neg == other.neg {
            return Self::build(self.neg, mag_add(&self.mag, &other.mag));
        }
        match mag_cmp(&self.mag, &other.mag) {
            Ordering::Less => Self::build(other.neg, mag_sub(&other.mag, &self.mag)),
            _ => Self::build(self.neg, mag_sub(&self.mag, &other.mag)),
        }
    }

    pub fn sub(&self, other: &Self) -> Result<Self, MathError> {
        self.add(&other.neg())
    }

    pub fn mul(&self, other: &Self) -> Result<Self, MathError> {
        // O produto tem pelo menos a + b - 1 bits: dá para recusar sem multiplicar.
        if !self.is_zero() && !other.is_zero() && self.bits() + other.bits() - 1 > MAX_BITS {
            return too_big();
        }
        Self::build(self.neg != other.neg, mag_mul(&self.mag, &other.mag))
    }

    /// Quociente truncado em direção a zero e resto com o sinal do dividendo, como os inteiros de Rust.
    pub fn div_rem(&self, other: &Self) -> Result<(Self, Self), MathError> {
        if other.is_zero() {
            return fail(ErrorCode::DivisionByZero, "divisão por zero");
        }
        let (q, r) = mag_divrem(&self.mag, &other.mag);
        Ok((
            Self::build(self.neg != other.neg, q)?,
            Self::build(self.neg, r)?,
        ))
    }

    /// Potência, com o tamanho estimado antes: 2^(10^9) é recusado sem alocar.
    pub fn pow(&self, exponent: u64) -> Result<Self, MathError> {
        if exponent == 0 {
            return Ok(Self::one());
        }
        if self.is_zero() || self.is_unit() {
            let neg = self.neg && exponent & 1 == 1;
            return Self::build(neg, self.mag.clone());
        }
        if (self.bits() - 1).saturating_mul(exponent) >= MAX_BITS {
            return too_big();
        }
        let mut result = Self::one();
        let mut base = self.clone();
        let mut left = exponent;
        loop {
            if left & 1 == 1 {
                result = result.mul(&base)?;
            }
            left >>= 1;
            if left == 0 {
                return Ok(result);
            }
            base = base.mul(&base)?;
        }
    }

    pub fn gcd(&self, other: &Self) -> Self {
        let (mut a, mut b) = (self.mag.clone(), other.mag.clone());
        while !b.is_empty() {
            let (_, r) = mag_divrem(&a, &b);
            a = b;
            b = r;
        }
        Self { neg: false, mag: a }
    }

    fn to_u64(&self) -> Option<u64> {
        match (self.neg, self.mag.as_slice()) {
            (true, _) => None,
            (false, []) => Some(0),
            (false, [low]) => Some(u64::from(*low)),
            (false, [low, high]) => Some(u64::from(*low) | (u64::from(*high) << BASE_BITS)),
            _ => None,
        }
    }
}

impl From<i64> for Int {
    fn from(value: i64) -> Self {
        let raw = value.unsigned_abs();
        Self {
            neg: value < 0,
            mag: trim(vec![raw as u32, (raw >> BASE_BITS) as u32]),
        }
    }
}

impl PartialOrd for Int {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Int {
    fn cmp(&self, other: &Self) -> Ordering {
        match (self.neg, other.neg) {
            (false, true) => Ordering::Greater,
            (true, false) => Ordering::Less,
            (false, false) => mag_cmp(&self.mag, &other.mag),
            (true, true) => mag_cmp(&other.mag, &self.mag),
        }
    }
}

impl fmt::Display for Int {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_zero() {
            return f.write_str("0");
        }
        let mut chunks = Vec::new();
        let mut rest = self.mag.clone();
        while !rest.is_empty() {
            let (q, r) = mag_divrem_small(&rest, DECIMAL_CHUNK);
            chunks.push(r);
            rest = q;
        }
        if self.neg {
            f.write_str("-")?;
        }
        let mut parts = chunks.iter().rev();
        if let Some(first) = parts.next() {
            write!(f, "{first}")?;
        }
        for part in parts {
            write!(f, "{part:09}")?;
        }
        Ok(())
    }
}

/// Racional sempre reduzido, com denominador positivo: igualdade de valor é igualdade de campos.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rational {
    num: Int,
    den: Int,
}

impl Rational {
    pub fn new(num: Int, den: Int) -> Result<Self, MathError> {
        if den.is_zero() {
            return fail(ErrorCode::DivisionByZero, "divisão por zero");
        }
        let common = num.gcd(&den);
        let (mut num, _) = num.div_rem(&common)?;
        let (mut den, _) = den.div_rem(&common)?;
        if den.is_negative() {
            num = num.neg();
            den = den.neg();
        }
        Ok(Self { num, den })
    }

    pub fn from_int(value: Int) -> Self {
        Self {
            num: value,
            den: Int::one(),
        }
    }

    /// `0.1` é 1/10, o decimal escrito, e não o binário mais próximo. Aceita inteiro, sinal e `1/3`.
    pub fn parse(text: &str) -> Result<Self, MathError> {
        if let Some((num, den)) = text.split_once('/') {
            return Self::new(Int::parse(num)?, Int::parse(den)?);
        }
        let invalid = || MathError::new(ErrorCode::ParseError, format!("número inválido: {text}"));
        let (neg, unsigned) = match text.strip_prefix('-') {
            Some(rest) => (true, rest),
            None => (false, text.strip_prefix('+').unwrap_or(text)),
        };
        let (whole, fraction) = unsigned.split_once('.').unwrap_or((unsigned, ""));
        let digits = format!("{whole}{fraction}");
        if digits.is_empty() || !digits.bytes().all(|c| c.is_ascii_digit()) {
            return Err(invalid());
        }
        let num = Int::parse(&digits)?;
        let den = Int::from(10).pow(fraction.len() as u64)?;
        Self::new(if neg { num.neg() } else { num }, den)
    }

    pub fn numerator(&self) -> &Int {
        &self.num
    }

    pub fn denominator(&self) -> &Int {
        &self.den
    }

    pub fn is_zero(&self) -> bool {
        self.num.is_zero()
    }

    pub fn is_integer(&self) -> bool {
        self.den.is_unit()
    }

    pub fn neg(&self) -> Self {
        Self {
            num: self.num.neg(),
            den: self.den.clone(),
        }
    }

    pub fn abs(&self) -> Self {
        Self {
            num: self.num.abs(),
            den: self.den.clone(),
        }
    }

    pub fn add(&self, other: &Self) -> Result<Self, MathError> {
        let left = self.num.mul(&other.den)?;
        let right = other.num.mul(&self.den)?;
        Self::new(left.add(&right)?, self.den.mul(&other.den)?)
    }

    pub fn sub(&self, other: &Self) -> Result<Self, MathError> {
        self.add(&other.neg())
    }

    pub fn mul(&self, other: &Self) -> Result<Self, MathError> {
        Self::new(self.num.mul(&other.num)?, self.den.mul(&other.den)?)
    }

    pub fn div(&self, other: &Self) -> Result<Self, MathError> {
        Self::new(self.num.mul(&other.den)?, self.den.mul(&other.num)?)
    }

    /// Potência de expoente inteiro. 0^0 é recusado: a convenção depende do contexto, e aqui não há contexto.
    pub fn pow(&self, exponent: &Int) -> Result<Self, MathError> {
        if self.is_zero() {
            if exponent.is_zero() {
                return fail(ErrorCode::DomainError, "0^0 não é definido aqui");
            }
            if exponent.is_negative() {
                return fail(ErrorCode::DivisionByZero, "divisão por zero");
            }
            return Ok(self.clone());
        }
        let size = exponent.abs();
        let result = match size.to_u64() {
            Some(power) => Self {
                num: self.num.pow(power)?,
                den: self.den.pow(power)?,
            },
            // Expoente enorme só cabe com base 1 ou -1, e aí só a paridade importa.
            None if self.num.is_unit() && self.den.is_unit() => {
                let odd = size.mag.first().is_some_and(|low| low & 1 == 1);
                if self.num.is_negative() && odd {
                    self.clone()
                } else {
                    Self::from_int(Int::one())
                }
            }
            None => return too_big(),
        };
        if exponent.is_negative() {
            Self::new(result.den, result.num)
        } else {
            Ok(result)
        }
    }

    /// f64 mais próximo, com empate para par. Só por pedido explícito: a conversão perde valor.
    /// Fora da faixa de f64 é erro; abaixo da menor faixa, arredonda como o próprio f64.
    pub fn to_f64(&self) -> Result<f64, MathError> {
        if self.is_zero() {
            return Ok(0.0);
        }
        // Quociente com 66 ou 67 bits: sobra margem para arredondar uma vez só.
        let shift = 66 - (self.num.bits() as i64 - self.den.bits() as i64);
        let (top, bottom) = if shift >= 0 {
            (mag_shl(&self.num.mag, shift as u64), self.den.mag.clone())
        } else {
            (
                self.num.mag.clone(),
                mag_shl(&self.den.mag, (-shift) as u64),
            )
        };
        let (quotient, remainder) = mag_divrem(&top, &bottom);
        let sticky = !remainder.is_empty();
        let q = quotient
            .iter()
            .rev()
            .fold(0u128, |v, digit| (v << BASE_BITS) | u128::from(*digit));
        let q_bits = 128 - i64::from(q.leading_zeros());
        let exponent = q_bits - 1 - shift;
        let precision = if exponent >= -1022 {
            53
        } else {
            53 - (-1022 - exponent)
        };
        if precision < 0 {
            return Ok(if self.num.is_negative() { -0.0 } else { 0.0 });
        }
        let drop = q_bits - precision;
        let mantissa = q >> drop;
        let half = 1u128 << (drop - 1);
        let tail = q & ((1u128 << drop) - 1);
        let up = tail > half || (tail == half && (sticky || mantissa & 1 == 1));
        let mantissa = mantissa + u128::from(up);
        let mut value = mantissa as f64;
        let mut power = drop - shift;
        // Passos de potência de 2: o valor final é representável, e nenhum passo intermediário arredonda.
        while power > 0 {
            let step = power.min(1000);
            value *= 2f64.powi(step as i32);
            power -= step;
        }
        while power < 0 {
            let step = (-power).min(1000);
            value *= 2f64.powi(-step as i32);
            power += step;
        }
        if value.is_infinite() {
            return fail(
                ErrorCode::NotRepresentable,
                "o valor exato não cabe em ponto flutuante",
            );
        }
        Ok(if self.num.is_negative() {
            -value
        } else {
            value
        })
    }

    /// Decimal cortado em `places` casas, sem arredondar, e se esse texto é o valor inteiro.
    pub fn decimal_preview(&self, places: u32) -> Result<(String, bool), MathError> {
        let scale = Int::from(10).pow(u64::from(places))?;
        let (scaled, rest) = self.num.abs().mul(&scale)?.div_rem(&self.den)?;
        let digits = scaled.to_string();
        let width = places as usize + 1;
        let padded = format!("{digits:0>width$}");
        let (whole, fraction) = padded.split_at(padded.len() - places as usize);
        let fraction = fraction.trim_end_matches('0');
        let sign = if self.num.is_negative() { "-" } else { "" };
        let text = if fraction.is_empty() {
            format!("{sign}{whole}")
        } else {
            format!("{sign}{whole}.{fraction}")
        };
        Ok((text, rest.is_zero()))
    }
}

impl PartialOrd for Rational {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Rational {
    fn cmp(&self, other: &Self) -> Ordering {
        // Compara por produto cruzado sem o limite de MAX_BITS: comparar não pode falhar.
        let left = mag_mul(&self.num.mag, &other.den.mag);
        let right = mag_mul(&other.num.mag, &self.den.mag);
        match (self.num.is_negative(), other.num.is_negative()) {
            (false, true) => Ordering::Greater,
            (true, false) => Ordering::Less,
            (false, false) => mag_cmp(&left, &right),
            (true, true) => mag_cmp(&right, &left),
        }
    }
}

impl fmt::Display for Rational {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_integer() {
            write!(f, "{}", self.num)
        } else {
            write!(f, "{}/{}", self.num, self.den)
        }
    }
}

/// Avalia uma expressão só com números, `+ - * / ^`, parênteses e `abs`, sem arredondar. A precedência é a
/// mesma de [`crate::parse`]. O que não fecha em racionais é recusado, nunca aproximado.
pub fn exact_eval(text: &str) -> Result<Rational, MathError> {
    if text.trim().is_empty() {
        return fail(ErrorCode::ParseError, "expressão vazia ou longa demais");
    }
    if text.len() > crate::MAX_LEN {
        return fail(ErrorCode::LimitSize, "expressão vazia ou longa demais");
    }
    let mut p = Exact {
        s: text.as_bytes(),
        i: 0,
        depth: 0,
    };
    let value = p.sum()?;
    p.space();
    if p.i != p.s.len() {
        return fail(
            ErrorCode::ParseError,
            format!("símbolo inesperado na posição {}", p.i + 1),
        );
    }
    Ok(value)
}

struct Exact<'a> {
    s: &'a [u8],
    i: usize,
    depth: usize,
}

fn unavailable<T>(what: &str) -> Result<T, MathError> {
    fail(
        ErrorCode::ExactModeUnavailable,
        format!("{what} ainda não tem resultado exato"),
    )
}

impl Exact<'_> {
    fn space(&mut self) {
        while self.s.get(self.i) == Some(&b' ') {
            self.i += 1;
        }
    }
    fn eat(&mut self, c: u8) -> bool {
        self.space();
        let hit = self.s.get(self.i) == Some(&c);
        if hit {
            self.i += 1;
        }
        hit
    }
    fn enter(&mut self) -> Result<(), MathError> {
        self.depth += 1;
        if self.depth > crate::MAX_DEPTH {
            return fail(ErrorCode::LimitDepth, "expressão aninhada demais");
        }
        Ok(())
    }
    fn sum(&mut self) -> Result<Rational, MathError> {
        let mut v = self.product()?;
        loop {
            if self.eat(b'+') {
                v = v.add(&self.product()?)?;
            } else if self.eat(b'-') {
                v = v.sub(&self.product()?)?;
            } else {
                return Ok(v);
            }
        }
    }
    fn product(&mut self) -> Result<Rational, MathError> {
        let mut v = self.unary()?;
        loop {
            if self.eat(b'*') {
                v = v.mul(&self.unary()?)?;
            } else if self.eat(b'/') {
                v = v.div(&self.unary()?)?;
            } else {
                return Ok(v);
            }
        }
    }
    fn unary(&mut self) -> Result<Rational, MathError> {
        if self.eat(b'-') {
            self.enter()?;
            let v = self.unary()?.neg();
            self.depth -= 1;
            return Ok(v);
        }
        self.power()
    }
    fn power(&mut self) -> Result<Rational, MathError> {
        let base = self.atom()?;
        if self.eat(b'^') {
            self.enter()?;
            let exponent = self.unary()?;
            self.depth -= 1;
            if !exponent.is_integer() {
                return unavailable("potência de expoente não inteiro");
            }
            return base.pow(exponent.numerator());
        }
        Ok(base)
    }
    fn atom(&mut self) -> Result<Rational, MathError> {
        self.enter()?;
        self.space();
        let v = if self.eat(b'(') {
            let v = self.sum()?;
            if !self.eat(b')') {
                return fail(ErrorCode::ParseError, "parêntese não fechado");
            }
            v
        } else if self.s.get(self.i).is_some_and(u8::is_ascii_alphabetic) {
            self.word()?
        } else {
            self.number()?
        };
        self.depth -= 1;
        Ok(v)
    }
    fn word(&mut self) -> Result<Rational, MathError> {
        let start = self.i;
        while self
            .s
            .get(self.i)
            .is_some_and(|c| c.is_ascii_alphanumeric() || *c == b'_')
        {
            self.i += 1;
        }
        let name = std::str::from_utf8(&self.s[start..self.i]).unwrap_or("");
        self.space();
        if self.s.get(self.i) == Some(&b'(') {
            return match Func::parse(name) {
                Some(Func::Abs) => {
                    self.i += 1;
                    let v = self.sum()?;
                    if !self.eat(b')') {
                        return fail(ErrorCode::ParseError, "parêntese não fechado");
                    }
                    Ok(v.abs())
                }
                Some(_) => unavailable(name),
                None => fail(
                    ErrorCode::ParseError,
                    format!("função desconhecida: {name}"),
                ),
            };
        }
        match name {
            "pi" | "e" => unavailable(name),
            _ => fail(
                ErrorCode::MissingVariable,
                format!("variável sem valor: {name}"),
            ),
        }
    }
    fn number(&mut self) -> Result<Rational, MathError> {
        let start = self.i;
        while self
            .s
            .get(self.i)
            .is_some_and(|c| c.is_ascii_digit() || *c == b'.')
        {
            self.i += 1;
        }
        let text = std::str::from_utf8(&self.s[start..self.i]).unwrap_or("");
        if text.bytes().filter(|c| *c == b'.').count() > 1 || text == "." || text.is_empty() {
            return fail(
                ErrorCode::ParseError,
                format!("número esperado na posição {}", start + 1),
            );
        }
        Rational::parse(text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Sequência fixa, sem dependência: reproduz os mesmos casos em toda execução.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        /// Inteiro de até `limbs` dígitos, com bastante 0 e 0xFFFFFFFF: é onde a divisão longa erra.
        fn int(&mut self, limbs: u64) -> Int {
            let count = (self.next() % (limbs + 1)) as usize;
            let mag = (0..count)
                .map(|_| match self.next() % 5 {
                    0 => 0,
                    1 => u32::MAX,
                    2 => 1 << 31,
                    _ => self.next() as u32,
                })
                .collect();
            let neg = self.next() & 1 == 1;
            Int::build(neg, trim(mag)).unwrap()
        }
        fn rational(&mut self) -> Rational {
            let den = self.int(3);
            let den = if den.is_zero() { Int::one() } else { den };
            Rational::new(self.int(3), den).unwrap()
        }
    }

    fn eval(text: &str) -> String {
        exact_eval(text)
            .unwrap_or_else(|why| panic!("{text}: {why}"))
            .to_string()
    }

    fn code(text: &str) -> ErrorCode {
        exact_eval(text).unwrap_err().code
    }

    /// Gabaritos da seção 8 do contrato, escritos à mão.
    #[test]
    fn matches_the_contract_table() {
        assert_eq!(eval("0.1 + 0.2"), "3/10");
        assert_eq!(eval("1/3 + 1/6"), "1/2");
        assert_eq!(eval("2^100"), "1267650600228229401496703205376");
        assert_eq!(eval("2^-3"), "1/8");
        assert_eq!(eval("(1/3) * 3"), "1");
        assert_eq!(code("1/0"), ErrorCode::DivisionByZero);
        assert_eq!(code("0^0"), ErrorCode::DomainError);
        assert_eq!(code("0^-1"), ErrorCode::DivisionByZero);
        assert_eq!(code("sqrt(2)"), ErrorCode::ExactModeUnavailable);
        assert_eq!(code("2^(10^9)"), ErrorCode::LimitSize);
    }

    #[test]
    fn follows_the_same_precedence_as_the_float_parser() {
        assert_eq!(eval("2 + 3 * 4"), "14");
        assert_eq!(eval("2^3^2"), "512");
        assert_eq!(eval("-2^2"), "-4");
        assert_eq!(eval("(-2)^2"), "4");
        assert_eq!(eval("2^-1"), "1/2");
        assert_eq!(eval("15 / 100 * 2300"), "345");
        assert_eq!(eval("abs(-7/2)"), "7/2");
        assert_eq!(eval("(-1)^100000000000000000000001"), "-1");
        assert_eq!(eval("1^100000000000000000000000"), "1");
        assert_eq!(eval("0^5"), "0");
        assert_eq!(eval(".5 + 1."), "3/2");
        assert_eq!(eval("0.10"), "1/10");
    }

    #[test]
    fn refuses_instead_of_approximating() {
        for text in [
            "pi", "e", "sin(1)", "ln(2)", "sqrt(4)", "exp(1)", "2^0.5", "4^(1/2)",
        ] {
            assert_eq!(code(text), ErrorCode::ExactModeUnavailable, "{text}");
        }
        for text in ["", "2 +", "(1", "1.2.3", ".", "foo(2)", "2 2", "1e5", "x y"] {
            assert!(exact_eval(text).is_err(), "{text}");
        }
        assert_eq!(code("x + 1"), ErrorCode::MissingVariable);
        assert_eq!(code(&"(".repeat(200)), ErrorCode::LimitDepth);
        assert_eq!(code(&"1+".repeat(400)), ErrorCode::LimitSize);
        assert_eq!(code("2^262145"), ErrorCode::LimitSize);
        assert_eq!(code("3^(2^64)"), ErrorCode::LimitSize);
        assert_eq!(code("(1/3)^100000000"), ErrorCode::LimitSize);
    }

    #[test]
    fn the_size_limit_is_enforced_at_its_edge() {
        assert_eq!(exact_eval("2^262143").unwrap().numerator().bits(), MAX_BITS);
        assert_eq!(code("2^262144"), ErrorCode::LimitSize);
        let top = Int::from(2).pow(262_143).unwrap();
        assert!(top.mul(&Int::from(2)).is_err());
        assert!(top.add(&top).is_err());
        assert_eq!(
            Int::parse(&"9".repeat(100_000)).unwrap_err().code,
            ErrorCode::LimitSize
        );
    }

    #[test]
    fn integers_obey_the_ring_laws() {
        let mut rng = Rng(0x1234_5678_9abc_def1);
        for _ in 0..2000 {
            let (a, b, c) = (rng.int(6), rng.int(6), rng.int(6));
            assert_eq!(a.add(&b).unwrap(), b.add(&a).unwrap());
            assert_eq!(a.add(&b).unwrap().sub(&b).unwrap(), a);
            assert_eq!(a.mul(&b).unwrap(), b.mul(&a).unwrap());
            assert_eq!(
                a.mul(&b.add(&c).unwrap()).unwrap(),
                a.mul(&b).unwrap().add(&a.mul(&c).unwrap()).unwrap()
            );
            assert_eq!(Int::parse(&a.to_string()).unwrap(), a);
        }
    }

    #[test]
    fn division_reconstructs_the_dividend() {
        let mut rng = Rng(0xfeed_face_cafe_beef);
        for _ in 0..5000 {
            let (a, b) = (rng.int(12), rng.int(6));
            if b.is_zero() {
                assert_eq!(a.div_rem(&b).unwrap_err().code, ErrorCode::DivisionByZero);
                continue;
            }
            let (q, r) = a.div_rem(&b).unwrap();
            assert_eq!(q.mul(&b).unwrap().add(&r).unwrap(), a, "{a} / {b}");
            assert_eq!(mag_cmp(&r.mag, &b.mag), Ordering::Less, "{a} / {b}");
            assert!(
                r.is_zero() || r.is_negative() == a.is_negative(),
                "{a} / {b}"
            );
        }
    }

    /// O caso raro de Knuth em que qhat passa por um e o divisor volta a ser somado.
    #[test]
    fn division_handles_the_add_back_step() {
        let a = Int::build(false, vec![0, 0, 0x8000_0000, 0x7fff_ffff]).unwrap();
        let b = Int::build(false, vec![1, 0, 0x8000_0000]).unwrap();
        let (q, r) = a.div_rem(&b).unwrap();
        assert_eq!(q.mul(&b).unwrap().add(&r).unwrap(), a);
        assert_eq!(q.mag, vec![0xffff_fffe]);
    }

    #[test]
    fn rationals_obey_the_field_laws_and_stay_reduced() {
        let mut rng = Rng(0x0dd0_ba11_5eed_0001);
        for _ in 0..1000 {
            let (a, b, c) = (rng.rational(), rng.rational(), rng.rational());
            assert_eq!(a.add(&b).unwrap(), b.add(&a).unwrap());
            assert_eq!(a.add(&b).unwrap().sub(&b).unwrap(), a);
            assert_eq!(
                a.mul(&b.add(&c).unwrap()).unwrap(),
                a.mul(&b).unwrap().add(&a.mul(&c).unwrap()).unwrap()
            );
            if !b.is_zero() {
                assert_eq!(a.div(&b).unwrap().mul(&b).unwrap(), a);
            }
            let sum = a.add(&b).unwrap();
            assert_eq!(sum.numerator().gcd(sum.denominator()), Int::one());
            assert!(!sum.denominator().is_negative());
            assert_eq!(a.cmp(&b), a.sub(&b).unwrap().numerator().cmp(&Int::zero()));
        }
        assert_eq!(Rational::parse("6/-4").unwrap().to_string(), "-3/2");
        assert_eq!(Rational::parse("-0.0").unwrap().to_string(), "0");
        assert!(Rational::parse("1/0").is_err());
    }

    #[test]
    fn conversion_to_float_rounds_once_and_says_so() {
        let f = |text: &str| Rational::parse(text).unwrap().to_f64();
        assert_eq!(f("0.1"), Ok(0.1));
        assert_eq!(f("1/3"), Ok(1.0 / 3.0));
        assert_eq!(f("-2.5"), Ok(-2.5));
        assert_eq!(f("0"), Ok(0.0));
        // 2^53 + 1 é empate: vai para o par; 2^53 + 3 também, para cima.
        assert_eq!(f("9007199254740993"), Ok(9007199254740992.0));
        assert_eq!(f("9007199254740995"), Ok(9007199254740996.0));
        let huge = exact_eval("10^400").unwrap();
        assert_eq!(huge.to_f64().unwrap_err().code, ErrorCode::NotRepresentable);
        assert_eq!(exact_eval("10^-400").unwrap().to_f64(), Ok(0.0));
        assert_eq!(exact_eval("2^-1074").unwrap().to_f64(), Ok(5e-324));
        assert_eq!(exact_eval("2^-1075").unwrap().to_f64(), Ok(0.0));
        assert_eq!(exact_eval("3 * 2^-1075").unwrap().to_f64(), Ok(1e-323));
    }

    #[test]
    fn decimal_preview_says_whether_it_is_complete() {
        let p = |text: &str, n| exact_eval(text).unwrap().decimal_preview(n).unwrap();
        assert_eq!(p("1/2", 10), ("0.5".to_owned(), true));
        assert_eq!(p("1/3", 5), ("0.33333".to_owned(), false));
        assert_eq!(p("-2/3", 3), ("-0.666".to_owned(), false));
        assert_eq!(p("127/50", 4), ("2.54".to_owned(), true));
        assert_eq!(p("7", 4), ("7".to_owned(), true));
        assert_eq!(p("1/1000", 2), ("0".to_owned(), false));
    }
}
