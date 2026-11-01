//! Recusa com código estável, além do texto: quem chama decide pelo código, e a pessoa lê a mensagem.
use std::fmt;

/// Códigos do contrato de resultados (`docs/KATU_CONTRATO_RESULTADOS.md`). Código novo não muda o sentido
/// dos antigos; mudar o sentido exige outra versão do contrato.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ErrorCode {
    ParseError,
    InvalidArgument,
    DomainError,
    DivisionByZero,
    MissingVariable,
    PoleOrJump,
    NoSignChange,
    NotConverged,
    SingularMatrix,
    NearSingularMatrix,
    NotRepresentable,
    NoClosedForm,
    ExactModeUnavailable,
    VerificationFailed,
    LimitSize,
    LimitDepth,
    LimitEvaluations,
    LimitTime,
    LimitConcurrency,
    Cancelled,
}

impl ErrorCode {
    pub fn as_str(self) -> &'static str {
        match self {
            ErrorCode::ParseError => "parse_error",
            ErrorCode::InvalidArgument => "invalid_argument",
            ErrorCode::DomainError => "domain_error",
            ErrorCode::DivisionByZero => "division_by_zero",
            ErrorCode::MissingVariable => "missing_variable",
            ErrorCode::PoleOrJump => "pole_or_jump",
            ErrorCode::NoSignChange => "no_sign_change",
            ErrorCode::NotConverged => "not_converged",
            ErrorCode::SingularMatrix => "singular_matrix",
            ErrorCode::NearSingularMatrix => "near_singular_matrix",
            ErrorCode::NotRepresentable => "not_representable",
            ErrorCode::NoClosedForm => "no_closed_form",
            ErrorCode::ExactModeUnavailable => "exact_mode_unavailable",
            ErrorCode::VerificationFailed => "verification_failed",
            ErrorCode::LimitSize => "limit_size",
            ErrorCode::LimitDepth => "limit_depth",
            ErrorCode::LimitEvaluations => "limit_evaluations",
            ErrorCode::LimitTime => "limit_time",
            ErrorCode::LimitConcurrency => "limit_concurrency",
            ErrorCode::Cancelled => "cancelled",
        }
    }

    /// A conta falhou só neste ponto, por causa do valor: outro ponto da mesma expressão pode dar certo.
    pub fn is_point_failure(self) -> bool {
        matches!(
            self,
            ErrorCode::DomainError | ErrorCode::DivisionByZero | ErrorCode::NotRepresentable
        )
    }

    /// O `status` do contrato. O método falhar em afirmar não é o mesmo que o pedido estar errado.
    pub fn status(self) -> &'static str {
        match self {
            ErrorCode::ParseError
            | ErrorCode::InvalidArgument
            | ErrorCode::DomainError
            | ErrorCode::DivisionByZero
            | ErrorCode::SingularMatrix => "invalid_input",
            ErrorCode::MissingVariable => "data_unavailable",
            ErrorCode::NoClosedForm | ErrorCode::ExactModeUnavailable => "unsupported",
            ErrorCode::PoleOrJump
            | ErrorCode::NoSignChange
            | ErrorCode::NotConverged
            | ErrorCode::NearSingularMatrix
            | ErrorCode::NotRepresentable
            | ErrorCode::VerificationFailed => "inconclusive",
            ErrorCode::LimitSize
            | ErrorCode::LimitDepth
            | ErrorCode::LimitEvaluations
            | ErrorCode::LimitTime
            | ErrorCode::LimitConcurrency
            | ErrorCode::Cancelled => "resource_limit",
        }
    }
}

impl ErrorCode {
    /// Qual limite foi atingido, para `usage.limit_hit`.
    pub fn limit(self) -> Option<&'static str> {
        match self {
            ErrorCode::LimitSize => Some("size"),
            ErrorCode::LimitDepth => Some("depth"),
            ErrorCode::LimitEvaluations => Some("evaluations"),
            ErrorCode::LimitTime => Some("time"),
            ErrorCode::LimitConcurrency => Some("concurrency"),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MathError {
    pub code: ErrorCode,
    pub message: String,
}

impl MathError {
    pub fn new(code: ErrorCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }
}

impl fmt::Display for MathError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for MathError {}

impl From<MathError> for String {
    fn from(e: MathError) -> String {
        e.message
    }
}
