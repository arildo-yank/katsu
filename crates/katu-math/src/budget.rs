//! Orçamento de uma conta: prazo e pedido de cancelamento, valendo para a thread que a executa. As contas
//! caras conferem nos seus laços e param com erro. É cooperativo: um trecho sem conferência termina antes
//! de parar, e os limites de tamanho é que garantem que esse trecho seja curto. Sem orçamento ativo, nada
//! muda: a rota `/math` e o Python seguem sem prazo.
use crate::{fail, ErrorCode, MathError};
use std::{
    cell::RefCell,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};

#[derive(Clone)]
struct Budget {
    deadline: Instant,
    cancel: Option<Arc<AtomicBool>>,
}

thread_local! {
    static CURRENT: RefCell<Option<Budget>> = const { RefCell::new(None) };
}

/// Devolve o orçamento anterior ao sair, inclusive em pânico: a thread é reaproveitada pelo pool.
struct Restore(Option<Budget>);

impl Drop for Restore {
    fn drop(&mut self) {
        let previous = self.0.take();
        CURRENT.with(|current| *current.borrow_mut() = previous);
    }
}

/// Executa `work` com prazo de `limit`, a contar de agora. `cancel` em verdadeiro também interrompe.
pub fn with_budget<T>(
    limit: Duration,
    cancel: Option<Arc<AtomicBool>>,
    work: impl FnOnce() -> T,
) -> T {
    let budget = Budget {
        deadline: Instant::now() + limit,
        cancel,
    };
    let previous = CURRENT.with(|current| current.borrow_mut().replace(budget));
    let _restore = Restore(previous);
    work()
}

/// Erro se o prazo venceu ou a conta foi cancelada. Custa uma leitura do relógio com orçamento ativo.
pub fn check() -> Result<(), MathError> {
    CURRENT.with(|current| {
        let current = current.borrow();
        let Some(budget) = current.as_ref() else {
            return Ok(());
        };
        if budget
            .cancel
            .as_ref()
            .is_some_and(|flag| flag.load(Ordering::Relaxed))
        {
            return fail(ErrorCode::Cancelled, "a conta foi cancelada");
        }
        if Instant::now() >= budget.deadline {
            return fail(
                ErrorCode::LimitTime,
                "a conta passou do tempo máximo e foi interrompida",
            );
        }
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{exact_eval, integrate, parse, rk4, Vars};

    const HEAVY: &str = "3^165000 / 7^93000 + 5^112000 / 11^75000";

    #[test]
    fn nothing_changes_without_a_budget() {
        assert!(check().is_ok());
        assert_eq!(exact_eval("1/3 + 1/6").unwrap().to_string(), "1/2");
    }

    #[test]
    fn exact_arithmetic_stops_at_the_deadline() {
        let got = with_budget(Duration::ZERO, None, || exact_eval(HEAVY));
        assert_eq!(got.unwrap_err().code, ErrorCode::LimitTime);
        // Conta pequena, com folga de sobra, não sente o orçamento.
        let ok = with_budget(Duration::from_secs(60), None, || exact_eval("1/3 + 1/6"));
        assert_eq!(ok.unwrap().to_string(), "1/2");
    }

    #[test]
    fn numeric_loops_stop_at_the_deadline() {
        let hard = parse("sin(1 / x) / x^3").unwrap();
        let got = with_budget(Duration::ZERO, None, || {
            integrate(&hard, "x", 1e-6, 1.0, 1e-12, &Vars::new())
        });
        assert_eq!(got.unwrap_err().code, ErrorCode::LimitTime);
        let ode = parse("y").unwrap();
        let got = with_budget(Duration::ZERO, None, || {
            rk4(&ode, "t", "y", 0.0, 1.0, 1.0, 100_000, &Vars::new())
        });
        assert_eq!(got.unwrap_err().code, ErrorCode::LimitTime);
    }

    #[test]
    fn a_cancelled_flag_stops_the_work() {
        let flag = Arc::new(AtomicBool::new(true));
        let got = with_budget(Duration::from_secs(60), Some(flag), || exact_eval(HEAVY));
        assert_eq!(got.unwrap_err().code, ErrorCode::Cancelled);
    }

    #[test]
    fn the_budget_ends_with_the_call_even_after_a_panic() {
        with_budget(Duration::ZERO, None, || assert!(check().is_err()));
        assert!(check().is_ok());
        let panicked = std::panic::catch_unwind(|| {
            with_budget(Duration::ZERO, None, || panic!("de propósito"));
        });
        assert!(panicked.is_err());
        assert!(check().is_ok());
    }
}
