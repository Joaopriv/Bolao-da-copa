// Bandeiras (emoji) por nome de seleção em PT-BR.
// Códigos ISO 3166-1 alpha-2 -> regional indicator symbols (U+1F1E6..U+1F1FF).
// Inglaterra/Escócia usam sequências de tag Unicode (subdivisão GB-ENG/GB-SCT).

function isoToFlag(code) {
  return code
    .toUpperCase()
    .split("")
    .map((c) => String.fromCodePoint(0x1f1e6 + (c.charCodeAt(0) - 65)))
    .join("");
}

const ISO_BY_NAME_PT = {
  Argélia: "DZ",
  Argentina: "AR",
  Austrália: "AU",
  Áustria: "AT",
  Bélgica: "BE",
  "Bósnia e Herzegovina": "BA",
  Brasil: "BR",
  Canadá: "CA",
  "Cabo Verde": "CV",
  Colômbia: "CO",
  Croácia: "HR",
  Curaçao: "CW",
  "República Tcheca": "CZ",
  "RD Congo": "CD",
  Equador: "EC",
  Egito: "EG",
  França: "FR",
  Alemanha: "DE",
  Gana: "GH",
  Haiti: "HT",
  Irã: "IR",
  Iraque: "IQ",
  "Costa do Marfim": "CI",
  Japão: "JP",
  Jordânia: "JO",
  México: "MX",
  Marrocos: "MA",
  Holanda: "NL",
  "Nova Zelândia": "NZ",
  Noruega: "NO",
  Panamá: "PA",
  Paraguai: "PY",
  Portugal: "PT",
  Catar: "QA",
  "Arábia Saudita": "SA",
  Senegal: "SN",
  "África do Sul": "ZA",
  "Coreia do Sul": "KR",
  Espanha: "ES",
  Suécia: "SE",
  Suíça: "CH",
  Tunísia: "TN",
  Turquia: "TR",
  "Estados Unidos": "US",
  Uruguai: "UY",
  Uzbequistão: "UZ",
};

// Bandeiras de subdivisão (tag sequences) — não seguem o padrão ISO alpha-2.
const SPECIAL_FLAGS = {
  Inglaterra: "🏴󠁧󠁢󠁥󠁮󠁧󠁿",
  Escócia: "🏴󠁧󠁢󠁳󠁣󠁴󠁿",
};

export const FLAG_EMOJI = {
  ...Object.fromEntries(
    Object.entries(ISO_BY_NAME_PT).map(([name, iso]) => [name, isoToFlag(iso)])
  ),
  ...SPECIAL_FLAGS,
};

export function getFlag(name) {
  return FLAG_EMOJI[name] ?? "🏳️";
}
