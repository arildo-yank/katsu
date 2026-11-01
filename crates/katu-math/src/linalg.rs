//! Álgebra linear densa, sem dependências: determinante, inversa, produto de matrizes, produtos escalar
//! e vetorial. Em ponto flutuante: quase singular é recusado, em vez de devolver números enormes.

pub type Matrix = Vec<Vec<f64>>;

const MAX_SIZE: usize = 64;
const SINGULAR: f64 = 1e-12;

fn check(a: &Matrix) -> Result<(usize, usize), String> {
    let rows = a.len();
    let cols = a.first().map_or(0, Vec::len);
    if rows == 0 || cols == 0 {
        return Err("a matriz está vazia".into());
    }
    if rows > MAX_SIZE || cols > MAX_SIZE {
        return Err(format!(
            "matriz maior que o limite de {MAX_SIZE} por {MAX_SIZE}"
        ));
    }
    if a.iter().any(|r| r.len() != cols) {
        return Err("as linhas da matriz têm tamanhos diferentes".into());
    }
    if a.iter().flatten().any(|v| !v.is_finite()) {
        return Err("a matriz tem um valor não finito".into());
    }
    Ok((rows, cols))
}

fn square(a: &Matrix) -> Result<usize, String> {
    let (rows, cols) = check(a)?;
    if rows != cols {
        return Err(format!(
            "a matriz precisa ser quadrada; esta é {rows} por {cols}"
        ));
    }
    Ok(rows)
}

fn finite(value: f64) -> Result<f64, String> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err("resultado não finito".into())
    }
}

/// v * 2^e em passos, sem estourar no meio quando o resultado cabe. Potência de 2 não arredonda.
fn ldexp(mut v: f64, mut e: i64) -> f64 {
    const STEP: i64 = 1000;
    e = e.clamp(-8 * STEP, 8 * STEP);
    while e > STEP {
        v *= 2f64.powi(STEP as i32);
        e -= STEP;
    }
    while e < -STEP {
        v *= 2f64.powi(-STEP as i32);
        e += STEP;
    }
    v * 2f64.powi(e as i32)
}

/// Expoente de base 2 do maior valor absoluto; 0 quando tudo é zero.
fn exponent(values: impl Iterator<Item = f64>) -> i64 {
    let top = values.fold(0.0f64, |m, v| m.max(v.abs()));
    if top == 0.0 {
        0
    } else {
        top.log2().floor() as i64
    }
}

/// Matriz com linhas e colunas de tamanho próximo de 1, e os expoentes retirados: A[i][j] = M[i][j] *
/// 2^(linha[i] + coluna[j]). Dizer que um pivô é zero só faz sentido nessa escala: diag(1e13, 1) não é
/// quase singular, só tem linhas de tamanhos diferentes.
fn balanced(a: &Matrix) -> (Matrix, Vec<i64>, Vec<i64>) {
    let n = a.len();
    let rows: Vec<i64> = a.iter().map(|r| exponent(r.iter().copied())).collect();
    let mut m: Matrix = a
        .iter()
        .zip(&rows)
        .map(|(r, e)| r.iter().map(|v| ldexp(*v, -e)).collect())
        .collect();
    let cols: Vec<i64> = (0..n).map(|j| exponent(m.iter().map(|r| r[j]))).collect();
    for row in &mut m {
        for (v, e) in row.iter_mut().zip(&cols) {
            *v = ldexp(*v, -e);
        }
    }
    (m, rows, cols)
}

fn best_pivot(m: &Matrix, col: usize) -> usize {
    (col..m.len())
        .max_by(|&i, &j| m[i][col].abs().total_cmp(&m[j][col].abs()))
        .unwrap_or(col)
}

/// Eliminação de Gauss com pivoteamento parcial, na matriz balanceada. Zero só quando o pivô é
/// indistinguível do arredondamento; determinante muito pequeno, mas distinguível, sai como valor. O que
/// não cabe em f64 é recusado.
#[allow(clippy::needless_range_loop)]
pub fn det(a: &Matrix) -> Result<f64, String> {
    let n = square(a)?;
    let (mut m, rows, cols) = balanced(a);
    let tiny = 8.0 * n as f64 * f64::EPSILON;
    let (mut mantissa, mut power) = (1.0f64, rows.iter().chain(&cols).sum::<i64>());
    for col in 0..n {
        let pivot = best_pivot(&m, col);
        if m[pivot][col].abs() <= tiny {
            return Ok(0.0);
        }
        if pivot != col {
            m.swap(pivot, col);
            mantissa = -mantissa;
        }
        mantissa *= m[col][col];
        let shift = exponent(std::iter::once(mantissa));
        mantissa = ldexp(mantissa, -shift);
        power += shift;
        for row in col + 1..n {
            let factor = m[row][col] / m[col][col];
            for k in col..n {
                m[row][k] -= factor * m[col][k];
            }
        }
    }
    let value = finite(ldexp(mantissa, power))?;
    if value.abs() < f64::MIN_POSITIVE {
        return Err("o determinante é pequeno demais para ponto flutuante".into());
    }
    Ok(value)
}

/// Gauss-Jordan com pivoteamento parcial, na matriz balanceada. Matriz singular, ou quase, é erro.
pub fn inverse(a: &Matrix) -> Result<Matrix, String> {
    let n = square(a)?;
    let (balanced, rows, cols) = balanced(a);
    let mut m: Matrix = balanced
        .iter()
        .enumerate()
        .map(|(i, row)| {
            let mut wide = row.clone();
            wide.extend((0..n).map(|j| if i == j { 1.0 } else { 0.0 }));
            wide
        })
        .collect();
    for col in 0..n {
        let pivot = best_pivot(&m, col);
        if m[pivot][col].abs() <= SINGULAR {
            return Err("a matriz é singular: não tem inversa".into());
        }
        m.swap(pivot, col);
        let p = m[col][col];
        m[col].iter_mut().for_each(|v| *v /= p);
        let line = m[col].clone();
        for (row, other) in m.iter_mut().enumerate() {
            if row != col && other[col] != 0.0 {
                let factor = other[col];
                other
                    .iter_mut()
                    .zip(&line)
                    .for_each(|(v, l)| *v -= factor * l);
            }
        }
    }
    // A = R M C, logo a inversa é C⁻¹ M⁻¹ R⁻¹.
    let mut out = Matrix::with_capacity(n);
    for (i, row) in m.iter().enumerate() {
        let mut line = Vec::with_capacity(n);
        for (j, v) in row[n..].iter().enumerate() {
            let value = finite(ldexp(*v, -cols[i] - rows[j]))?;
            if value == 0.0 && *v != 0.0 {
                return Err("a inversa tem valor pequeno demais para ponto flutuante".into());
            }
            line.push(value);
        }
        out.push(line);
    }
    Ok(out)
}

pub fn multiply(a: &Matrix, c: &Matrix) -> Result<Matrix, String> {
    let (rows, inner) = check(a)?;
    let (other, cols) = check(c)?;
    if inner != other {
        return Err(format!(
            "tamanhos incompatíveis: {rows} por {inner} vezes {other} por {cols}"
        ));
    }
    let out: Matrix = (0..rows)
        .map(|i| {
            (0..cols)
                .map(|j| (0..inner).map(|k| a[i][k] * c[k][j]).sum())
                .collect()
        })
        .collect();
    if out.iter().flatten().any(|v: &f64| !v.is_finite()) {
        return Err("resultado não finito".into());
    }
    Ok(out)
}

pub fn dot(u: &[f64], v: &[f64]) -> Result<f64, String> {
    if u.is_empty() || u.len() != v.len() {
        return Err("os vetores precisam ter o mesmo tamanho".into());
    }
    finite(u.iter().zip(v).map(|(a, c)| a * c).sum())
}

pub fn cross(u: &[f64], v: &[f64]) -> Result<Vec<f64>, String> {
    if u.len() != 3 || v.len() != 3 {
        return Err("o produto vetorial pede dois vetores de 3 componentes".into());
    }
    let out = vec![
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    ];
    for value in &out {
        finite(*value)?;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, c: f64) {
        assert!((a - c).abs() < 1e-9, "{a} != {c}");
    }

    #[test]
    fn determinants_match_known_answers() {
        close(det(&vec![vec![2.0]]).unwrap(), 2.0);
        close(det(&vec![vec![1.0, 2.0], vec![3.0, 4.0]]).unwrap(), -2.0);
        close(det(&vec![vec![0.0, 1.0], vec![1.0, 0.0]]).unwrap(), -1.0);
        close(
            det(&vec![
                vec![2.0, 0.0, 1.0],
                vec![1.0, 3.0, 2.0],
                vec![1.0, 1.0, 4.0],
            ])
            .unwrap(),
            18.0,
        );
        assert_eq!(det(&vec![vec![1.0, 2.0], vec![2.0, 4.0]]).unwrap(), 0.0);
    }

    #[test]
    fn inverse_times_matrix_is_the_identity() {
        let a = vec![
            vec![4.0, 7.0, 2.0],
            vec![3.0, 6.0, 1.0],
            vec![2.0, 5.0, 3.0],
        ];
        let product = multiply(&a, &inverse(&a).unwrap()).unwrap();
        for (i, row) in product.iter().enumerate() {
            for (j, v) in row.iter().enumerate() {
                close(*v, if i == j { 1.0 } else { 0.0 });
            }
        }
        let needs_pivot = vec![vec![0.0, 1.0], vec![1.0, 0.0]];
        assert_eq!(inverse(&needs_pivot).unwrap(), needs_pivot);
    }

    #[test]
    fn vector_products_match_known_answers() {
        close(dot(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]).unwrap(), 32.0);
        assert_eq!(
            cross(&[1.0, 0.0, 0.0], &[0.0, 1.0, 0.0]).unwrap(),
            vec![0.0, 0.0, 1.0]
        );
        assert_eq!(
            cross(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]).unwrap(),
            vec![-3.0, 6.0, -3.0]
        );
    }

    #[test]
    fn refuses_invalid_input() {
        assert!(
            inverse(&vec![vec![1.0, 2.0], vec![2.0, 4.0]]).is_err(),
            "singular"
        );
        assert!(det(&vec![vec![1.0, 2.0]]).is_err(), "não quadrada");
        assert!(det(&vec![]).is_err(), "vazia");
        assert!(
            det(&vec![vec![1.0, 2.0], vec![3.0]]).is_err(),
            "linhas diferentes"
        );
        assert!(
            multiply(&vec![vec![1.0, 2.0]], &vec![vec![1.0, 2.0]]).is_err(),
            "tamanhos"
        );
        assert!(dot(&[1.0], &[1.0, 2.0]).is_err());
        assert!(cross(&[1.0, 2.0], &[3.0, 4.0]).is_err());
        assert!(det(&vec![vec![0.0; 65]; 65]).is_err(), "grande demais");
    }

    fn relative(a: f64, c: f64) {
        assert!((a - c).abs() <= 1e-12 * c.abs(), "{a} != {c}");
    }

    /// Escala não é singularidade: as linhas podem diferir em muitas ordens de grandeza.
    #[test]
    fn determinant_is_not_fooled_by_scale() {
        relative(det(&vec![vec![1e13, 0.0], vec![0.0, 1.0]]).unwrap(), 1e13);
        relative(det(&vec![vec![1.0, 0.0], vec![0.0, 1e13]]).unwrap(), 1e13);
        relative(det(&vec![vec![1e-13, 0.0], vec![0.0, 1.0]]).unwrap(), 1e-13);
        relative(
            det(&vec![vec![1e100, 0.0], vec![0.0, 1e-100]]).unwrap(),
            1.0,
        );
        relative(
            det(&vec![
                vec![1e200, 0.0, 0.0],
                vec![0.0, 1e200, 0.0],
                vec![0.0, 0.0, 1e-300],
            ])
            .unwrap(),
            1e100,
        );
        relative(det(&vec![vec![1e13, 2e13], vec![3.0, 4.0]]).unwrap(), -2e13);
        relative(
            det(&vec![vec![1.0, 1.0], vec![1.0, 1.0 + 1e-10]]).unwrap(),
            1e-10 * 1.000000082740371,
        );
    }

    #[test]
    fn determinant_of_singular_matrices_is_zero_at_any_scale() {
        assert_eq!(det(&vec![vec![0.0; 3]; 3]).unwrap(), 0.0);
        assert_eq!(
            det(&vec![
                vec![1.0, 2.0, 3.0],
                vec![4.0, 5.0, 6.0],
                vec![7.0, 8.0, 9.0],
            ])
            .unwrap(),
            0.0
        );
        assert_eq!(det(&vec![vec![1e13, 2e13], vec![2e13, 4e13]]).unwrap(), 0.0);
        assert_eq!(det(&vec![vec![1e-13, 2e-13], vec![1.0, 2.0]]).unwrap(), 0.0);
        assert_eq!(det(&vec![vec![1.0, 0.0], vec![0.0, 0.0]]).unwrap(), 0.0);
    }

    /// O que não cabe em f64 é recusado, nunca devolvido como zero ou infinito.
    #[test]
    fn determinant_refuses_what_a_double_cannot_hold() {
        assert!(det(&vec![vec![1e-200, 0.0], vec![0.0, 1e-200]]).is_err());
        assert!(det(&vec![vec![1e200, 0.0], vec![0.0, 1e200]]).is_err());
    }

    #[test]
    fn inverse_is_not_fooled_by_scale() {
        let out = inverse(&vec![vec![1e13, 0.0], vec![0.0, 1.0]]).unwrap();
        relative(out[0][0], 1e-13);
        relative(out[1][1], 1.0);
        let a = vec![vec![1e13, 1.0], vec![1.0, 1.0]];
        let product = multiply(&a, &inverse(&a).unwrap()).unwrap();
        for (i, row) in product.iter().enumerate() {
            for (j, v) in row.iter().enumerate() {
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((v - want).abs() < 1e-9, "{v} != {want}");
            }
        }
        assert!(inverse(&vec![vec![1e13, 2e13], vec![2e13, 4e13]]).is_err());
        assert!(
            inverse(&vec![vec![1.0, 1.0], vec![1.0, 1.0 + 1e-15]]).is_err(),
            "quase singular"
        );
    }
}
