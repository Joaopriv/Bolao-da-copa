import { lookupScoreProb } from "./scoreLookup";
import { pickStatus } from "./status";

function csvEscape(value) {
  const str = String(value ?? "");
  if (/[",\n;]/.test(str)) {
    return `"${str.replace(/"/g, '""')}"`;
  }
  return str;
}

// Exporta as apostas salvas para CSV (UTF-8 BOM + RFC4180), colunas:
// jogo,data,aposta,prob_modelo,status
export function exportPicksCsv(matches, picks, results) {
  const header = ["jogo", "data", "aposta", "prob_modelo", "status"];
  const rows = [header];

  for (const match of matches) {
    const pick = picks[match.game];
    if (!pick) continue;

    const [home, away] = pick.score.split("-").map(Number);
    const prob = lookupScoreProb(match, home, away);
    const result = results[match.game];
    const status = pickStatus({ home, away }, result) ?? "Pendente";

    rows.push([
      match.game,
      match.date,
      pick.score,
      prob != null ? (prob * 100).toFixed(1) + "%" : "-",
      status,
    ]);
  }

  const csvContent = rows.map((row) => row.map(csvEscape).join(",")).join("\r\n");
  const blob = new Blob(["﻿" + csvContent], { type: "text/csv;charset=utf-8;" });

  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "minhas_apostas_bolao_2026.csv";
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}
