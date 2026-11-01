//! O que há dentro de um intervalo em que a função parece não se comportar: curva contínua, polo, salto ou
//! oscilação. Raiz e gráfico usam o mesmo critério, para não discordarem sobre o mesmo polo. O critério é a queda
//! de escala: estreitando o intervalo, em curva contínua a diferença cai; em polo cresce; em salto fica estável.
//! Amostragem finita pode ser enganada, e por isso o resultado é suspeita, não prova.
use crate::MathError;

/// Dez metades estreitam o intervalo cerca de mil vezes; contínua de Hölder com expoente acima de 0,1 cai
/// mais que à metade nesse trecho, e a escala em polo ou salto nunca cai.
pub const SCALE_WINDOW: usize = 10;
/// Abaixo disto o que resta é ruído de arredondamento de uma curva contínua.
const NOISE: f64 = 1e-6;
const LEVELS: usize = 40;
/// Crescimento de |f| que conta como polo, em `SCALE_WINDOW` níveis. Baixo, porque ln(abs(x)) cresce devagar.
const PEAK_GROWTH: f64 = 1.1;
/// A metade descartada ainda varia assim, em média: não há ponto isolado a estreitar, e sim oscilação.
const SIBLING_SHARE: f64 = 0.25;
const OFF_CENTER: f64 = 0.381966;

/// `true` se a escala caiu junto com o intervalo. `first` é a do início, `before` a de `SCALE_WINDOW` passos atrás.
pub(crate) fn scale_falls(first: f64, before: f64, last: f64) -> bool {
    last <= NOISE * first || last < 0.5 * before
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Verdict {
    Continuous,
    Pole,
    Jump,
    Oscillating,
    /// A função deixa de estar definida no intervalo.
    Undefined,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Bracket {
    pub verdict: Verdict,
    /// Onde a suspeita ficou, depois de estreitar.
    pub from: f64,
    pub to: f64,
}

/// Julga [a, c] por duas buscas: uma estreita para a metade de maior diferença, e acha salto e polo que troca de
/// sinal; a outra persegue o maior |f|, e acha o polo de mesmo sinal dos dois lados, que a primeira perde de vista.
/// `f` devolve `None` onde a função não está definida. Custa no máximo 40 avaliações na primeira e 80 na segunda, e
/// em curva bem comportada para depois de `SCALE_WINDOW` níveis.
pub fn classify_bracket<F>(mut f: F, a: f64, c: f64) -> Result<Bracket, MathError>
where
    F: FnMut(f64) -> Result<Option<f64>, MathError>,
{
    let (Some(f_a), Some(f_c)) = (f(a)?, f(c)?) else {
        return Ok(Bracket {
            verdict: Verdict::Undefined,
            from: a,
            to: c,
        });
    };
    let by_jump = narrow_by_jump(&mut f, a, f_a, c, f_c)?;
    if by_jump.verdict == Verdict::Pole || by_jump.verdict == Verdict::Undefined {
        return Ok(by_jump);
    }
    Ok(chase_peak(&mut f, a, f_a, c, f_c)?.unwrap_or(by_jump))
}

fn narrow_by_jump<F>(f: &mut F, a: f64, f_a: f64, c: f64, f_c: f64) -> Result<Bracket, MathError>
where
    F: FnMut(f64) -> Result<Option<f64>, MathError>,
{
    let (mut left, mut f_left, mut right, mut f_right) = (a, f_a, c, f_c);
    let first = (f_left - f_right).abs();
    let mut jumps = vec![first];
    let mut siblings = Vec::new();
    for level in 1..=LEVELS {
        let mut middle = (left + right) / 2.0;
        if middle <= left || middle >= right {
            break;
        }
        let mut value = f(middle)?;
        // O meio pode cair em cima do próprio polo; um ponto ao lado resolve, e dois indefinidos são lacuna.
        if value.is_none() {
            middle = left + (right - left) * OFF_CENTER;
            if middle <= left || middle >= right {
                break;
            }
            value = f(middle)?;
        }
        let Some(f_middle) = value else {
            // Sem número entre os dois: o indefinido é um ponto só, e o julgamento fica com o que já se viu.
            if right - left <= 8.0 * f64::EPSILON * left.abs().max(right.abs()) {
                break;
            }
            return Ok(Bracket {
                verdict: Verdict::Undefined,
                from: left,
                to: right,
            });
        };
        let (jump_left, jump_right) = ((f_left - f_middle).abs(), (f_middle - f_right).abs());
        if jump_left >= jump_right {
            (right, f_right) = (middle, f_middle);
            jumps.push(jump_left);
            siblings.push(if jump_left > 0.0 {
                jump_right / jump_left
            } else {
                0.0
            });
        } else {
            (left, f_left) = (middle, f_middle);
            jumps.push(jump_right);
            siblings.push(jump_left / jump_right);
        }
        // Antes de duas janelas, a inclinação em volta de um salto ainda pode passar por queda.
        if level >= 2 * SCALE_WINDOW
            && scale_falls(first, jumps[level - SCALE_WINDOW], jumps[level])
        {
            return Ok(Bracket {
                verdict: Verdict::Continuous,
                from: left,
                to: right,
            });
        }
    }
    let back = jumps.len().saturating_sub(SCALE_WINDOW + 1);
    let (last, before) = (jumps[jumps.len() - 1], jumps[back]);
    let verdict = if scale_falls(first, before, last) {
        Verdict::Continuous
    } else if last > 2.0 * before {
        Verdict::Pole
    } else {
        let recent = &siblings[siblings.len().saturating_sub(SCALE_WINDOW)..];
        let share = recent.iter().sum::<f64>() / recent.len().max(1) as f64;
        if share > SIBLING_SHARE {
            Verdict::Oscillating
        } else {
            Verdict::Jump
        }
    };
    Ok(Bracket {
        verdict,
        from: left,
        to: right,
    })
}

/// `Some` só se |f| cresce sem parar perto do seu maior valor. Em curva contínua, esse maior valor se estabiliza logo,
/// numa ponta ou num pico de verdade.
fn chase_peak<F>(
    f: &mut F,
    a: f64,
    f_a: f64,
    c: f64,
    f_c: f64,
) -> Result<Option<Bracket>, MathError>
where
    F: FnMut(f64) -> Result<Option<f64>, MathError>,
{
    let (mut left, mut right) = (a, c);
    let (mut best, mut height) = if f_a.abs() >= f_c.abs() {
        (a, f_a.abs())
    } else {
        (c, f_c.abs())
    };
    let mut heights = vec![height];
    for level in 1..=LEVELS {
        // Os pontos em ordem, com |f| dos novos; o maior valor conhecido é `best`, e o resto só delimita.
        let mut points = vec![(left, 0.0)];
        for x in [(left + best) / 2.0, best, (best + right) / 2.0, right] {
            if x <= points[points.len() - 1].0 {
                continue;
            }
            let probed = x != best && x != right;
            let y = if x == best {
                height
            } else if probed {
                f(x)?.map_or(0.0, f64::abs)
            } else {
                0.0
            };
            points.push((x, y));
        }
        if points.len() <= 3
            && points
                .iter()
                .all(|point| point.0 == left || point.0 == best || point.0 == right)
        {
            break;
        }
        points[0].1 = if left == best { height } else { 0.0 };
        let top = (0..points.len()).fold(
            0,
            |top, i| if points[i].1 > points[top].1 { i } else { top },
        );
        (best, height) = points[top];
        left = points[top.saturating_sub(1)].0;
        right = points[(top + 1).min(points.len() - 1)].0;
        heights.push(height);
        if level >= SCALE_WINDOW && height <= PEAK_GROWTH * heights[level - SCALE_WINDOW] {
            return Ok(None);
        }
    }
    let back = heights.len().saturating_sub(SCALE_WINDOW + 1);
    Ok((height > PEAK_GROWTH * heights[back]).then_some(Bracket {
        verdict: Verdict::Pole,
        from: left,
        to: right,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{eval, parse, Vars};

    fn classify(text: &str, a: f64, c: f64) -> Verdict {
        let e = parse(text).unwrap();
        let f = |x: f64| {
            let vars = Vars::from([("x".to_owned(), x)]);
            match eval(&e, &vars) {
                Ok(y) => Ok(Some(y)),
                Err(why) if why.code.is_point_failure() => Ok(None),
                Err(why) => Err(why),
            }
        };
        classify_bracket(f, a, c).unwrap().verdict
    }

    /// Os mesmos casos em que a bisseção recusa a raiz: aqui não podem sair contínuos.
    #[test]
    fn what_bisection_refuses_is_a_break_here() {
        for (text, a, c, expected) in [
            ("1/(x-0.1)", 0.0, 1.0, Verdict::Pole),
            ("1/(x-0.3)^3", 0.0, 1.0, Verdict::Pole),
            ("-1000000/(x-0.7)", 0.0, 1.0, Verdict::Pole),
            ("0.000000001/(x-0.1)", 0.0, 1.0, Verdict::Pole),
            ("tan(x)", 1.0, 2.0, Verdict::Pole),
            ("abs(x-0.3)/(x-0.3)", 0.0, 1.0, Verdict::Jump),
            ("abs(x-0.3)/(x-0.3) + x", -2.0, 1.0, Verdict::Jump),
        ] {
            assert_eq!(classify(text, a, c), expected, "{text}");
        }
    }

    /// Os mesmos casos em que a bisseção acha a raiz: aqui são curvas contínuas.
    #[test]
    fn what_bisection_keeps_is_continuous_here() {
        for (text, a, c) in [
            ("x^2 - 2", 0.0, 2.0),
            ("x^3 - 2*x - 5", 2.0, 3.0),
            ("cos(x)", 1.0, 2.0),
            ("100000000000000000000*(x-0.1)", 0.0, 1.0),
            ("0.000000000000000000001*(x-0.1)", 0.0, 1.0),
            ("(x-1)^9", 0.0, 3.0),
            ("abs(x-0.3)^0.5*(x-0.3)/abs(x-0.3)", 0.0, 1.0),
        ] {
            assert_eq!(classify(text, a, c), Verdict::Continuous, "{text}");
        }
    }

    #[test]
    fn other_singularities() {
        assert_eq!(classify("1/x^2", -0.3, 0.2), Verdict::Pole);
        assert_eq!(classify("1/x", -1.0, 1.0), Verdict::Pole);
        assert_eq!(classify("ln(abs(x))", -0.3, 0.2), Verdict::Pole);
        assert_eq!(classify("abs(x)/x", -1.0, 1.0), Verdict::Jump);
        assert_eq!(classify("sqrt(x)", -1.0, 1.0), Verdict::Undefined);
        assert_eq!(classify("5", 0.0, 1.0), Verdict::Continuous);
    }

    #[test]
    fn detail_finer_than_the_numbers_is_not_called_a_jump() {
        let verdict = classify("sin(1/x)", 1e-150, 1.0000001e-150);
        assert!(
            !matches!(verdict, Verdict::Jump | Verdict::Pole),
            "{verdict:?}"
        );
    }

    #[test]
    fn the_suspect_is_narrowed_down() {
        let e = parse("1/(x-0.3)").unwrap();
        let f = |x: f64| Ok(eval(&e, &Vars::from([("x".to_owned(), x)])).ok());
        let found = classify_bracket(f, 0.0, 1.0).unwrap();
        assert!(found.from <= 0.3 && 0.3 <= found.to, "{found:?}");
        assert!(found.to - found.from < 1e-6);
    }
}
