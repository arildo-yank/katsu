//! Sinônimos em português para a busca: a pergunta de quem usa ("fio", "placa", "anotar") ganha as
//! palavras que um documento técnico costuma usar ("condutor", "sinalização", "registrar"). Tabela
//! escrita à mão, sem modelo: cada grupo pode ser lido, conferido e corrigido por um auditor.
//!
//! A origem de cada grupo diz por que ele existe, para que a medida seja honesta:
//! - `Regua`: saiu dos erros da régua de desenvolvimento (ARANDU/docs/regua_busca_nr12.json).
//! - `Geral`: vocabulário comum de segurança do trabalho, que qualquer técnico listaria.
//! - `Consultor`: escrito depois de ler as perguntas de ARANDU/docs/regua_busca_nr12_consultor.json.
//!   Aquela régua NÃO é cega para quem escreveu esta tabela; medir sem estes grupos mostra quanto do
//!   ganho vem deles.
//!
//! Alguns grupos `Regua` e `Geral` têm palavras que também aparecem nas perguntas do consultor
//! ("sinalização", "treinar", "risco", "maquinário"); a origem diz o motivo do grupo, não a coincidência.
use std::collections::HashMap;
use std::sync::OnceLock;

use crate::terms;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Origin {
    Regua,
    Geral,
    Consultor,
}

/// Quais grupos a busca usa (KATU_RAG_SYNONYMS no servidor).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Synonyms {
    /// Todos os grupos (padrão).
    #[default]
    All,
    /// Sem os grupos `Consultor`: para medir a régua do consultor sem a tabela escrita para ela.
    WithoutConsultor,
    /// Nenhum: a busca só por palavras da pergunta.
    Off,
}

impl Synonyms {
    /// "todos" (ou vazio), "sem-consultor", "nenhum"; outro valor é erro.
    pub fn parse(value: &str) -> Result<Self, String> {
        match value.trim() {
            "" | "todos" => Ok(Self::All),
            "sem-consultor" => Ok(Self::WithoutConsultor),
            "nenhum" => Ok(Self::Off),
            other => Err(format!(
                "KATU_RAG_SYNONYMS={other}: use todos, sem-consultor ou nenhum"
            )),
        }
    }

    fn uses(self, origin: Origin) -> bool {
        match self {
            Self::All => true,
            Self::WithoutConsultor => origin != Origin::Consultor,
            Self::Off => false,
        }
    }
}

/// Palavras equivalentes, uma por grupo. Só palavras soltas: cada uma vira o mesmo radical da busca.
pub const GROUPS: &[(Origin, &[&str])] = &[
    // Régua: "Que cuidados os fios de alimentação…" → "condutores de alimentação elétrica".
    (
        Origin::Regua,
        &[
            "fio",
            "fios",
            "fiação",
            "cabo",
            "cabos",
            "condutor",
            "condutores",
        ],
    ),
    // Régua: "placas avisando dos riscos" → "sinalização de segurança para advertir".
    (
        Origin::Regua,
        &[
            "placa",
            "placas",
            "aviso",
            "avisos",
            "avisando",
            "cartaz",
            "sinalização",
            "sinalizar",
        ],
    ),
    // Régua: "Onde devo anotar as manutenções" → "devem ser registradas em livro próprio".
    (
        Origin::Regua,
        &[
            "anotar",
            "anotação",
            "anotações",
            "registrar",
            "registro",
            "registradas",
        ],
    ),
    // Régua: "botões de ligar e desligar" → "dispositivos de partida, acionamento e parada".
    (
        Origin::Regua,
        &["botão", "botões", "dispositivo", "dispositivos"],
    ),
    (
        Origin::Regua,
        &["ligar", "liga", "acionar", "acionamento", "partida"],
    ),
    (Origin::Regua, &["desligar", "desliga", "parar", "parada"]),
    // Régua: "marcadas no chão" → "áreas de circulação devem ser demarcadas"; "piso".
    (Origin::Regua, &["chão", "piso"]),
    (
        Origin::Regua,
        &["marcar", "marcadas", "demarcar", "demarcadas", "demarcação"],
    ),
    // Régua: "para consertar" → "reparos".
    (
        Origin::Regua,
        &["consertar", "conserto", "reparo", "reparos", "reparar"],
    ),
    // Geral: vocabulário de segurança do trabalho.
    (
        Origin::Geral,
        &[
            "treinar",
            "treinamento",
            "treinamentos",
            "capacitar",
            "capacitação",
            "curso",
        ],
    ),
    (
        Origin::Geral,
        &[
            "operador",
            "operadores",
            "trabalhador",
            "trabalhadores",
            "funcionário",
            "funcionários",
            "empregado",
            "empregados",
        ],
    ),
    (
        Origin::Geral,
        &["empresa", "empresas", "empregador", "patrão"],
    ),
    (
        Origin::Geral,
        &[
            "perigo", "perigos", "perigosa", "perigoso", "risco", "riscos",
        ],
    ),
    (
        Origin::Geral,
        &[
            "maquinário",
            "máquina",
            "máquinas",
            "equipamento",
            "equipamentos",
        ],
    ),
    (Origin::Geral, &["proibido", "proibida", "vedado", "vedada"]),
    (Origin::Geral, &["acidente", "acidentes", "sinistro"]),
    // Consultor: escritos depois de ler as perguntas dele.
    (
        Origin::Consultor,
        &[
            "etiquetagem",
            "etiqueta",
            "etiquetas",
            "cartão",
            "loto",
            "tagout",
        ],
    ),
    (
        Origin::Consultor,
        &[
            "sensor",
            "sensores",
            "detector",
            "detectores",
            "sensoriamento",
        ],
    ),
    (
        Origin::Consultor,
        &[
            "desativado",
            "desativar",
            "neutralizar",
            "burlar",
            "burla",
            "anular",
        ],
    ),
    (
        Origin::Consultor,
        &["correia", "correias", "polia", "polias", "transmissão"],
    ),
];

/// Para cada radical, os radicais equivalentes (sem ele mesmo), conforme o modo.
fn table(mode: Synonyms) -> &'static HashMap<String, Vec<String>> {
    static ALL: OnceLock<HashMap<String, Vec<String>>> = OnceLock::new();
    static WITHOUT: OnceLock<HashMap<String, Vec<String>>> = OnceLock::new();
    static OFF: OnceLock<HashMap<String, Vec<String>>> = OnceLock::new();
    let cell = match mode {
        Synonyms::All => &ALL,
        Synonyms::WithoutConsultor => &WITHOUT,
        Synonyms::Off => &OFF,
    };
    cell.get_or_init(|| {
        let mut map: HashMap<String, Vec<String>> = HashMap::new();
        for (origin, words) in GROUPS {
            if !mode.uses(*origin) {
                continue;
            }
            let mut stems: Vec<String> = words.iter().flat_map(|w| terms(w)).collect();
            stems.sort();
            stems.dedup();
            for stem in &stems {
                let entry = map.entry(stem.clone()).or_default();
                for other in &stems {
                    if other != stem && !entry.contains(other) {
                        entry.push(other.clone());
                    }
                }
            }
        }
        map
    })
}

/// Os radicais equivalentes a um radical da pergunta (vazio se não há grupo para ele).
pub fn equivalents(stem: &str, mode: Synonyms) -> &'static [String] {
    table(mode).get(stem).map_or(&[], Vec::as_slice)
}
