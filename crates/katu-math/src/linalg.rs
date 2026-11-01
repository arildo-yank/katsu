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

/// Maior valor absoluto, a referência de escala para dizer que um pivô é zero.
fn scale(a: &Matrix) -> f64 {
    a.iter().flatten().fold(0.0f64, |m, v| m.max(v.abs()))
}

/// Eliminação de Gauss com pivoteamento parcial.
#[allow(clippy::needless_range_loop)]
pub fn det(a: &Matrix) -> Result<f64, String> {
    let n = square(a)?;
    let tiny = SINGULAR * scale(a);
    let mut m = a.clone();
    let mut value = 1.0;
    for col in 0..n {
        let pivot = (col..n)
            .max_by(|&i, &j| m[i][col].abs().total_cmp(&m[j][col].abs()))
            .unwrap_or(col);
        if m[pivot][col].abs() <= tiny {
            return Ok(0.0);
        }
        if pivot != col {
            m.swap(pivot, col);
            value = -value;
        }
        value *= m[col][col];
        for row in col + 1..n {
            let factor = m[row][col] / m[col][col];
            for k in col..n {
                m[row][k] -= factor * m[col][k];
            }
        }
    }
    finite(value)
}

/// Gauss-Jordan com pivoteamento parcial. Matriz singular, ou quase, é erro.
pub fn inverse(a: &Matrix) -> Result<Matrix, String> {
    let n = square(a)?;
    let tiny = SINGULAR * scale(a);
    let mut m: Matrix = a
        .iter()
        .enumerate()
        .map(|(i, row)| {
            let mut wide = row.clone();
            wide.extend((0..n).map(|j| if i == j { 1.0 } else { 0.0 }));
            wide
        })
        .collect();
    for col in 0..n {
        let pivot = (col..n)
            .max_by(|&i, &j| m[i][col].abs().total_cmp(&m[j][col].abs()))
            .unwrap_or(col);
        if m[pivot][col].abs() <= tiny {
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
    let out: Matrix = m.into_iter().map(|row| row[n..].to_vec()).collect();
    if out.iter().flatten().any(|v| !v.is_finite()) {
        return Err("resultado não finito".into());
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
}
