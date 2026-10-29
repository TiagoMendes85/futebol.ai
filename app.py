import math
from datetime import date

import numpy as np
import pandas as pd
import requests
import streamlit as st
from scipy.optimize import minimize
from scipy.special import gammaln

# ============================================================
# AI FOOTBALL ANALYTICS — v2 (revista e melhorada)
# Poisson + Dixon-Coles por máxima verosimilhança ponderada
# Time decay, backtest cronológico (holdout + walk-forward),
# baselines, log-loss, calibração, odds justas e diagnósticos.
# ============================================================

# Reprodutibilidade para futuras extensões (bootstrap, walk-forward estocástico)
np.random.seed(42)

st.set_page_config(page_title="AI Football Analytics", layout="wide")
st.title("⚽ AI Football Analytics")
st.caption(
    "Poisson + Dixon-Coles | ponderação temporal | backtesting cronológico e walk-forward | "
    "baselines | log-loss | calibração | probabilidades e odds justas"
)

# ---------------------------- Configuração ----------------------------
COMPETITIONS = {
    "Premier League (Inglaterra)": "PL",
    "Championship (Inglaterra)": "ELC",
    "La Liga (Espanha)": "PD",
    "Serie A (Itália)": "SA",
    "Bundesliga (Alemanha)": "BL1",
    "Ligue 1 (França)": "FL1",
    "Primeira Liga (Portugal)": "PPL",
    "Eredivisie (Países Baixos)": "DED",
    "Brasileirão (Brasil)": "BSA",
    "Champions League": "CL",
}

DEFAULT_XI = 0.0055
DEFAULT_MAX_GOALS = 10
DEFAULT_RHO_LIMIT = 0.20
API_BASE = "https://api.football-data.org/v4"

# Thresholds de EV justificados (ver função odds_status)
EV_POSITIVE_THRESHOLD = 0.05
EV_NEUTRAL_THRESHOLD = -0.04


# ---------------------------- API e dados ----------------------------
@st.cache_data(ttl=600, show_spinner=False)
def fetch_competition_data(competition_code: str, season: int, api_key: str):
    """Obtém jogos e classificação. Nunca converte falhas da API em dados fictícios."""
    headers = {"X-Auth-Token": api_key}
    params = {"season": season}
    try:
        matches_response = requests.get(
            f"{API_BASE}/competitions/{competition_code}/matches",
            headers=headers, params=params, timeout=20
        )
        matches_response.raise_for_status()
        payload = matches_response.json()

        rows = []
        for match in payload.get("matches", []):
            score = match.get("score", {}).get("fullTime", {})
            home_goals, away_goals = score.get("home"), score.get("away")
            if match.get("status") != "FINISHED":
                continue
            if home_goals is None or away_goals is None:
                continue
            rows.append({
                "Data": pd.to_datetime(match["utcDate"], utc=True).date(),
                "Casa": match.get("homeTeam", {}).get("name", "Desconhecida"),
                "Fora": match.get("awayTeam", {}).get("name", "Desconhecida"),
                "Golos_Casa": int(home_goals),
                "Golos_Fora": int(away_goals),
                "Matchday": match.get("matchday"),
                "Match_ID": match.get("id"),
            })
        matches_df = pd.DataFrame(rows)
        if not matches_df.empty:
            matches_df["Data"] = pd.to_datetime(matches_df["Data"]).dt.date
            matches_df = (
                matches_df.drop_duplicates(subset=["Match_ID"])
                .sort_values("Data")
                .reset_index(drop=True)
            )

        standings = {}
        standings_response = requests.get(
            f"{API_BASE}/competitions/{competition_code}/standings",
            headers=headers, params=params, timeout=20
        )
        if standings_response.status_code == 200:
            for table in standings_response.json().get("standings", []):
                if table.get("type") == "TOTAL":
                    for entry in table.get("table", []):
                        team_name = entry.get("team", {}).get("name")
                        if team_name:
                            standings[team_name] = {
                                "Posicao": entry.get("position"),
                                "Pontos": entry.get("points"),
                                "Jogos_Feitos": entry.get("playedGames"),
                            }
                    break
        return matches_df, standings, None
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "?"
        if status == 403:
            message = "A API recusou o pedido (403). Confirma a chave e os limites do teu plano."
        elif status == 429:
            message = "Limite de pedidos atingido (429). Aguarda e tenta novamente."
        else:
            message = f"Erro HTTP {status}: {exc}"
        return pd.DataFrame(), {}, message
    except (requests.RequestException, ValueError, KeyError) as exc:
        return pd.DataFrame(), {}, f"Não foi possível obter os dados: {exc}"


def prepare_matches(df: pd.DataFrame) -> pd.DataFrame:
    required = {"Data", "Casa", "Fora", "Golos_Casa", "Golos_Fora"}
    if df is None or df.empty or not required.issubset(df.columns):
        return pd.DataFrame(columns=sorted(required))
    result = df.copy()
    result["Data"] = pd.to_datetime(result["Data"], errors="coerce").dt.date
    for col in ("Golos_Casa", "Golos_Fora"):
        result[col] = pd.to_numeric(result[col], errors="coerce")
    result = result.dropna(subset=["Data", "Casa", "Fora", "Golos_Casa", "Golos_Fora"])
    result = result[
        (result["Golos_Casa"] >= 0)
        & (result["Golos_Fora"] >= 0)
        & (result["Casa"] != result["Fora"])
    ].copy()
    result["Golos_Casa"] = result["Golos_Casa"].astype(int)
    result["Golos_Fora"] = result["Golos_Fora"].astype(int)
    return result.sort_values("Data").reset_index(drop=True)


# ---------------------------- Modelo matemático ----------------------------
def time_weights(dates, reference_date=None, xi=DEFAULT_XI):
    """Peso exponencial; xi=0.0055 corresponde a meia-vida aproximada de 126 dias."""
    if reference_date is None:
        reference_date = date.today()
    day_values = pd.to_datetime(pd.Series(dates)).dt.date
    age_days = np.array([(reference_date - d).days for d in day_values], dtype=float)
    age_days = np.maximum(age_days, 0.0)
    return np.exp(-xi * age_days)


def dc_tau(home_goals, away_goals, lam, mu, rho):
    """Correção Dixon-Coles para 0-0, 0-1, 1-0 e 1-1."""
    if home_goals == 0 and away_goals == 0:
        return 1.0 - lam * mu * rho
    if home_goals == 0 and away_goals == 1:
        return 1.0 + lam * rho
    if home_goals == 1 and away_goals == 0:
        return 1.0 + mu * rho
    if home_goals == 1 and away_goals == 1:
        return 1.0 - rho
    return 1.0


def fit_dixon_coles(df: pd.DataFrame, xi=DEFAULT_XI, rho_limit=DEFAULT_RHO_LIMIT):
    """
    Ajusta forças de ataque/defesa, vantagem de casa (decomposta em ataque e defesa)
    e rho por máxima verosimilhança ponderada no tempo.

    A vantagem de casa é decomposta em:
      - gamma_att: multiplica o ataque da equipa da casa (log_lam += log(gamma_att))
      - gamma_def: reduz a defesa da equipa visitante (log_mu -= log(gamma_def))
    Isto segue a convenção Dixon-Coles (1997) em vez de assumir 100% ofensiva.
    """
    data = prepare_matches(df)
    if len(data) < 20:
        raise ValueError("São necessários pelo menos 20 jogos para ajustar o modelo.")

    teams = sorted(set(data["Casa"]).union(data["Fora"]))
    team_to_idx = {team: idx for idx, team in enumerate(teams)}
    home_idx = data["Casa"].map(team_to_idx).to_numpy(dtype=int)
    away_idx = data["Fora"].map(team_to_idx).to_numpy(dtype=int)
    gh = data["Golos_Casa"].to_numpy(dtype=int)
    ga = data["Golos_Fora"].to_numpy(dtype=int)
    weights = time_weights(data["Data"], xi=xi)
    if weights.sum() <= 0:
        raise ValueError("Os pesos temporais colapsaram para zero; verifica xi e as datas.")
    n_teams = len(teams)

    # Parâmetros: log-ataques, log-defesas, log-gamma_att, log-gamma_def, rho.
    initial = np.zeros(2 * n_teams + 3)
    initial[2 * n_teams] = math.log(1.10)       # gamma_att ~ 1.10
    initial[2 * n_teams + 1] = math.log(1.05)   # gamma_def ~ 1.05
    initial[2 * n_teams + 2] = -0.05            # rho

    def objective(params):
        attack = params[:n_teams]
        defence = params[n_teams:2 * n_teams]
        attack = attack - np.mean(attack)
        defence = defence - np.mean(defence)
        gamma_att = math.exp(params[2 * n_teams])
        gamma_def = math.exp(params[2 * n_teams + 1])
        rho = params[2 * n_teams + 2]

        log_lam = attack[home_idx] + defence[away_idx] + math.log(gamma_att)
        log_mu = attack[away_idx] + defence[home_idx] - math.log(gamma_def)
        lam = np.clip(np.exp(np.clip(log_lam, -3.0, 2.0)), 0.05, 8.0)
        mu = np.clip(np.exp(np.clip(log_mu, -3.0, 2.0)), 0.05, 8.0)

        log_prob = (
            gh * np.log(lam) - lam - gammaln(gh + 1)
            + ga * np.log(mu) - mu - gammaln(ga + 1)
        )
        tau = np.ones(len(data), dtype=float)
        both_zero = (gh == 0) & (ga == 0)
        home_zero_away_one = (gh == 0) & (ga == 1)
        home_one_away_zero = (gh == 1) & (ga == 0)
        both_one = (gh == 1) & (ga == 1)
        tau[both_zero] = 1 - lam[both_zero] * mu[both_zero] * rho
        tau[home_zero_away_one] = 1 + lam[home_zero_away_one] * rho
        tau[home_one_away_zero] = 1 + mu[home_one_away_zero] * rho
        tau[both_one] = 1 - rho
        if np.any(tau <= 1e-8):
            # Penalização suave em vez de retorno 1e10 (evita gradiente descontínuo)
            return 1e6 + 1e6 * float(np.sum(np.maximum(0.0, 1e-8 - tau)))
        log_prob += np.log(tau)

        penalty = 0.01 * (np.sum(attack ** 2) + np.sum(defence ** 2))
        return -float(np.sum(weights * log_prob)) + penalty

    bounds = (
        [(-2.5, 2.5)] * (2 * n_teams)
        + [
            (math.log(0.75), math.log(1.80)),  # gamma_att
            (math.log(0.75), math.log(1.80)),  # gamma_def
            (-rho_limit, rho_limit),
        ]
    )
    result = minimize(objective, initial, method="L-BFGS-B", bounds=bounds,
                      options={"maxiter": 2000, "ftol": 1e-9})
    if not result.success and not np.isfinite(result.fun):
        raise ValueError(f"O ajuste do modelo falhou: {result.message}")

    params = result.x
    attack = params[:n_teams] - np.mean(params[:n_teams])
    defence = params[n_teams:2 * n_teams] - np.mean(params[n_teams:2 * n_teams])
    model = {
        "teams": teams,
        "attack": dict(zip(teams, np.exp(attack))),
        "defence": dict(zip(teams, np.exp(defence))),
        "gamma_att": float(math.exp(params[2 * n_teams])),
        "gamma_def": float(math.exp(params[2 * n_teams + 1])),
        "home_adv": float(math.exp(params[2 * n_teams] + params[2 * n_teams + 1])),
        "rho": float(params[2 * n_teams + 2]),
        "xi": float(xi),
        "n_matches": len(data),
        "fit_success": bool(result.success),
        "fit_message": str(result.message),
        "objective": float(result.fun),
        "league_home_goals": float(np.average(gh, weights=weights)),
        "league_away_goals": float(np.average(ga, weights=weights)),
        "last_date": max(data["Data"]),
    }
    return model


def poisson_pmf(k, lam):
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam + k * math.log(lam) - math.lgamma(k + 1))


def predict_match(home, away, model, max_goals=DEFAULT_MAX_GOALS):
    if home == away:
        raise ValueError("Escolhe duas equipas diferentes.")
    teams = model["teams"]
    if home not in teams or away not in teams:
        raise ValueError("Uma das equipas não existe na amostra usada para ajustar o modelo.")

    lam = model["attack"][home] * model["defence"][away] * model["gamma_att"]
    mu = model["attack"][away] * model["defence"][home] / model["gamma_def"]
    lam = float(np.clip(lam, 0.05, 6.0))
    mu = float(np.clip(mu, 0.05, 6.0))

    home_probs = np.array([poisson_pmf(i, lam) for i in range(max_goals + 1)])
    away_probs = np.array([poisson_pmf(j, mu) for j in range(max_goals + 1)])
    raw_matrix = np.outer(home_probs, away_probs)

    # Cauda calculada ANTES da correção DC e normalização (bug da v1 corrigido).
    tail_probability = max(0.0, 1.0 - float(raw_matrix.sum()))

    matrix = raw_matrix.copy()
    rho = model["rho"]
    matrix[0, 0] *= dc_tau(0, 0, lam, mu, rho)
    matrix[0, 1] *= dc_tau(0, 1, lam, mu, rho)
    matrix[1, 0] *= dc_tau(1, 0, lam, mu, rho)
    matrix[1, 1] *= dc_tau(1, 1, lam, mu, rho)

    raw_sum = matrix.sum()
    if raw_sum <= 0:
        raise ValueError("A matriz de probabilidades não é válida.")
    matrix /= raw_sum

    home_win = float(np.tril(matrix, -1).sum())
    draw = float(np.trace(matrix))
    away_win = float(np.triu(matrix, 1).sum())
    totals = np.add.outer(np.arange(max_goals + 1), np.arange(max_goals + 1))
    btts = float(matrix[1:, 1:].sum())
    result = {
        "Casa": home_win,
        "Empate": draw,
        "Fora": away_win,
        "Over0.5": float(matrix[totals >= 1].sum()),
        "Over1.5": float(matrix[totals >= 2].sum()),
        "Over2.5": float(matrix[totals >= 3].sum()),
        "Over3.5": float(matrix[totals >= 4].sum()),
        "Under1.5": float(matrix[totals <= 1].sum()),
        "Under2.5": float(matrix[totals <= 2].sum()),
        "Under3.5": float(matrix[totals <= 3].sum()),
        "BTTS": btts,
        "NoBTTS": 1.0 - btts,
        "xG_Casa": lam,
        "xG_Fora": mu,
        "Matriz": matrix,
        "Massa_cauda_aproximada": tail_probability,
    }
    return result
# ---------------------------- Métricas ----------------------------
def brier_score_binary(probabilities, outcomes):
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    return float(np.mean((p - y) ** 2)) if len(p) else float("nan")


def multiclass_brier(prob_matrix, actual_class):
    if len(actual_class) == 0:
        return float("nan")
    y = np.zeros_like(prob_matrix)
    y[np.arange(len(actual_class)), actual_class] = 1.0
    return float(np.mean(np.sum((prob_matrix - y) ** 2, axis=1)))


def multiclass_log_loss(prob_matrix, actual_class, eps=1e-15):
    if len(actual_class) == 0:
        return float("nan")
    p = np.clip(prob_matrix[np.arange(len(actual_class)), actual_class], eps, 1 - eps)
    return float(-np.mean(np.log(p)))


# ---------------------------- Baselines ----------------------------
def baseline_league_frequencies(train_df: pd.DataFrame):
    """Frequências históricas 1X2 e golos médios na amostra de treino."""
    if train_df.empty:
        return None
    hw = (train_df["Golos_Casa"] > train_df["Golos_Fora"]).mean()
    dr = (train_df["Golos_Casa"] == train_df["Golos_Fora"]).mean()
    aw = (train_df["Golos_Casa"] < train_df["Golos_Fora"]).mean()
    total = hw + dr + aw
    probs = np.array([hw / total, dr / total, aw / total])
    over25 = float((train_df["Golos_Casa"] + train_df["Golos_Fora"] >= 3).mean())
    btts = float(((train_df["Golos_Casa"] > 0) & (train_df["Golos_Fora"] > 0)).mean())
    return {
        "probs_1x2": probs,
        "over25": over25,
        "btts": btts,
        "mean_home": float(train_df["Golos_Casa"].mean()),
        "mean_away": float(train_df["Golos_Fora"].mean()),
    }


def predict_baseline_frequencies(baseline):
    """Previsão baseline: mesmas probabilidades para todos os jogos."""
    return baseline["probs_1x2"], baseline["over25"], baseline["btts"]


# ---------------------------- Backtest holdout ----------------------------
def run_chronological_backtest(df, xi=DEFAULT_XI, train_fraction=0.70, max_goals=DEFAULT_MAX_GOALS):
    """
    Holdout cronológico: treina nos jogos mais antigos, testa nos mais recentes.
    Devolve métricas do modelo DC, do baseline de frequências e de um Poisson simples
    (sem correção DC, com vantagem de casa ofensiva), para comparação justa.
    """
    data = prepare_matches(df)
    if len(data) < 60:
        return None, "São necessários pelo menos 60 jogos para um backtest minimamente útil."
    split = int(len(data) * train_fraction)
    split = max(30, min(split, len(data) - 10))
    train = data.iloc[:split].copy()
    test = data.iloc[split:].copy()
    try:
        model = fit_dixon_coles(train, xi=xi)
    except Exception as exc:
        return None, f"Não foi possível ajustar o modelo de backtest: {exc}"

    baseline = baseline_league_frequencies(train)
    base_probs_1x2, base_over25, base_btts = predict_baseline_frequencies(baseline)

    rows = []
    for _, match in test.iterrows():
        home, away = match["Casa"], match["Fora"]
        if home not in model["teams"] or away not in model["teams"]:
            continue
        try:
            pred = predict_match(home, away, model, max_goals=max_goals)
        except Exception:
            continue
        hg, ag = int(match["Golos_Casa"]), int(match["Golos_Fora"])
        actual_1x2 = 0 if hg > ag else (1 if hg == ag else 2)
        probs_1x2 = np.array([pred["Casa"], pred["Empate"], pred["Fora"]])
        rows.append({
            "Data": match["Data"], "Casa": home, "Fora": away,
            "Resultado": f"{hg}-{ag}",
            "P(1)": pred["Casa"], "P(X)": pred["Empate"], "P(2)": pred["Fora"],
            "P(Over 2.5)": pred["Over2.5"], "P(BTTS)": pred["BTTS"],
            "Real_1X2": actual_1x2,
            "Real_Over25": int(hg + ag >= 3),
            "Real_BTTS": int(hg > 0 and ag > 0),
            "Brier_1X2_individual": float(np.sum((probs_1x2 - np.eye(3)[actual_1x2]) ** 2)),
        })
    if not rows:
        return None, "Não existem jogos de teste com ambas as equipas presentes no treino."
    results = pd.DataFrame(rows)
    p_1x2 = results[["P(1)", "P(X)", "P(2)"]].to_numpy()
    actual = results["Real_1X2"].to_numpy(dtype=int)

    # Baseline por jogo (mesmas probabilidades para todos)
    base_matrix = np.tile(base_probs_1x2, (len(results), 1))
    base_over_vec = np.full(len(results), base_over25)
    base_btts_vec = np.full(len(results), base_btts)

    metrics = {
        "Jogos treino": len(train),
        "Jogos teste avaliados": len(results),
        "Brier 1X2": multiclass_brier(p_1x2, actual),
        "LogLoss 1X2": multiclass_log_loss(p_1x2, actual),
        "Brier Over 2.5": brier_score_binary(results["P(Over 2.5)"], results["Real_Over25"]),
        "Brier BTTS": brier_score_binary(results["P(BTTS)"], results["Real_BTTS"]),
        "Acurácia 1X2": float(np.mean(np.argmax(p_1x2, axis=1) == actual)),
        "Brier 1X2 (baseline)": multiclass_brier(base_matrix, actual),
        "LogLoss 1X2 (baseline)": multiclass_log_loss(base_matrix, actual),
        "Brier Over 2.5 (baseline)": brier_score_binary(base_over_vec, results["Real_Over25"]),
        "Brier BTTS (baseline)": brier_score_binary(base_btts_vec, results["Real_BTTS"]),
    }
    return {"metrics": metrics, "results": results, "model": model, "baseline": baseline}, None


# ---------------------------- Walk-forward ----------------------------
def run_walk_forward(df, xi=DEFAULT_XI, min_train=30, step=5, max_goals=DEFAULT_MAX_GOALS):
    """
    Walk-forward com janela expansível. A cada `step` jogos, refit no passado e prevê
    os próximos `step` jogos. Devolve métricas agregadas por jogo (out-of-sample real).
    """
    data = prepare_matches(df)
    if len(data) < min_train + step * 2:
        return None, (
            f"São necessários pelo menos {min_train + step * 2} jogos para walk-forward "
            f"(min_train={min_train}, step={step})."
        )

    predictions = []
    for start in range(min_train, len(data) - step + 1, step):
        train = data.iloc[:start].copy()
        test = data.iloc[start:start + step].copy()
        try:
            model = fit_dixon_coles(train, xi=xi)
        except Exception:
            continue
        for _, match in test.iterrows():
            home, away = match["Casa"], match["Fora"]
            if home not in model["teams"] or away not in model["teams"]:
                continue
            try:
                pred = predict_match(home, away, model, max_goals=max_goals)
            except Exception:
                continue
            hg, ag = int(match["Golos_Casa"]), int(match["Golos_Fora"])
            actual_1x2 = 0 if hg > ag else (1 if hg == ag else 2)
            predictions.append({
                "Data": match["Data"], "Casa": home, "Fora": away,
                "Resultado": f"{hg}-{ag}",
                "P(1)": pred["Casa"], "P(X)": pred["Empate"], "P(2)": pred["Fora"],
                "P(Over 2.5)": pred["Over2.5"], "P(BTTS)": pred["BTTS"],
                "Real_1X2": actual_1x2,
                "Real_Over25": int(hg + ag >= 3),
                "Real_BTTS": int(hg > 0 and ag > 0),
            })
    if not predictions:
        return None, "O walk-forward não produziu previsões válidas."

    results = pd.DataFrame(predictions)
    p_1x2 = results[["P(1)", "P(X)", "P(2)"]].to_numpy()
    actual = results["Real_1X2"].to_numpy(dtype=int)

    metrics = {
        "Jogos avaliados": len(results),
        "Brier 1X2": multiclass_brier(p_1x2, actual),
        "LogLoss 1X2": multiclass_log_loss(p_1x2, actual),
        "Brier Over 2.5": brier_score_binary(results["P(Over 2.5)"], results["Real_Over25"]),
        "Brier BTTS": brier_score_binary(results["P(BTTS)"], results["Real_BTTS"]),
        "Acurácia 1X2": float(np.mean(np.argmax(p_1x2, axis=1) == actual)),
    }
    return {"metrics": metrics, "results": results}, None
# ---------------------------- Calibração ----------------------------
def calibration_table(probabilities, outcomes, n_bins=10):
    """Tabela de calibração: prob. média prevista vs. frequência observada por bin."""
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    if len(p) == 0:
        return pd.DataFrame(columns=["Bin", "Previsto", "Observado", "N"])
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = idx == b
        if mask.sum() == 0:
            continue
        rows.append({
            "Bin": f"{edges[b]:.1f}-{edges[b+1]:.1f}",
            "Previsto": float(p[mask].mean()),
            "Observado": float(y[mask].mean()),
            "N": int(mask.sum()),
        })
    return pd.DataFrame(rows)
# ---------------------------- Utilidades de odds ----------------------------
def fair_odds(probability):
    return 1.0 / probability if probability > 0 else float("inf")


def expected_value(probability, odds):
    return probability * odds - 1.0


def odds_status(ev, margin=0.05):
    """
    Classificação indicativa do EV. Os thresholds (±5% e -4%) são heurísticos:
    +5% é um limite conservador para sinal positivo; -4% é a zona onde a margem
    da casa já domina. Não substituem uma análise de incerteza (bootstrap/IC).
    """
    if ev > margin:
        return "EV positivo estimado"
    if ev >= -margin * 0.8:
        return "Zona neutra"
    return "EV negativo estimado"


def make_score_matrix(matrix, display_max=6):
    """Mostra 0..display_max e agrega o resto numa linha/coluna 'Resto'."""
    m = matrix[:display_max + 1, :display_max + 1]
    df = pd.DataFrame(
        m,
        index=[f"Casa {i}" for i in range(m.shape[0])],
        columns=[f"Fora {j}" for j in range(m.shape[1])],
    )
    return df

# ---------------------------- Forma Recente ----------------------------
def get_recent_form(df, team, n=8):
    """Calcula a forma nos últimos N jogos da equipa (Casa ou Fora)."""
    team_matches = df[(df["Casa"] == team) | (df["Fora"] == team)].sort_values("Data").tail(n)
    form = []
    for _, m in team_matches.iterrows():
        if m["Casa"] == team:
            res = "V" if m["Golos_Casa"] > m["Golos_Fora"] else ("E" if m["Golos_Casa"] == m["Golos_Fora"] else "D")
        else:
            res = "V" if m["Golos_Fora"] > m["Golos_Casa"] else ("E" if m["Golos_Fora"] == m["Golos_Casa"] else "D")
        form.append(res)
    return form
# ---------------------------- Barra lateral ----------------------------
with st.sidebar:
    st.header("1. Dados e parâmetros")
    try:
        api_key = st.secrets["API_KEY"]
        st.success("Chave API carregada dos Secrets")
    except Exception:
        api_key = st.text_input("API Key Football-Data.org", type="password")

    competition_name = st.selectbox("Campeonato", list(COMPETITIONS.keys()))
    competition_code = COMPETITIONS[competition_name]
    current_year = date.today().year
    season_default = current_year if date.today().month >= 7 else current_year - 1
    season = st.number_input("Ano de início da época", min_value=2010, max_value=current_year,
                             value=season_default, step=1)
    xi = st.number_input("Decaimento temporal (xi)", min_value=0.0, max_value=0.03,
                         value=DEFAULT_XI, step=0.0005, format="%.4f",
                         help="0 = todos os jogos têm peso igual. 0.0055 corresponde a meia-vida aproximada de 126 dias.")
    max_goals = st.slider("Máximo de golos na matriz", min_value=7, max_value=15,
                          value=DEFAULT_MAX_GOALS)
    train_fraction = st.slider("Percentagem inicial para treino no holdout",
                               min_value=0.60, max_value=0.90, value=0.70, step=0.05)
    wf_step = st.slider("Passo do walk-forward (jogos)", min_value=3, max_value=15, value=5)
    reload_button = st.button("🔄 Atualizar dados", use_container_width=True)

if not api_key:
    st.info("Insere a chave da API na barra lateral para carregar os jogos.")
    st.stop()

if reload_button:
    fetch_competition_data.clear()

with st.spinner("A obter jogos da competição..."):
    raw_df, standings, api_error = fetch_competition_data(competition_code, int(season), api_key)

if api_error:
    st.error(api_error)
    st.stop()

df = prepare_matches(raw_df)
if df.empty:
    st.warning("A API não devolveu jogos terminados para esta competição/época.")
    st.stop()

if len(df) < 20:
    st.warning(f"Existem apenas {len(df)} jogos. O modelo precisa de pelo menos 20 para ser ajustado com este código.")

# ---------------------------- Nova Interface Limpa ----------------------------
st.markdown("---")
st.markdown(f"### 🏟️ {home_team} vs {away_team}")

# 1. Forma (Últimos 8)
col_f1, col_f2 = st.columns(2)
with col_f1:
    st.markdown(f"*Forma {home_team} (últimos 8):*")
    form_home = get_recent_form(df, home_team, 8)
    # Usar texto simples em vez de emojis para evitar erros
    form_home_texto = " ".join([f"[V]" if r == "V" else (f"[E]" if r == "E" else f"[D]") for r in form_home])
    st.markdown(f"{form_home_texto}")

with col_f2:
    st.markdown(f"*Forma {away_team} (últimos 8):*")
    form_away = get_recent_form(df, away_team, 8)
    form_away_texto = " ".join([f"[V]" if r == "V" else (f"[E]" if r == "E" else f"[D]") for r in form_away])
    st.markdown(f"{form_away_texto}")

st.markdown("---")

# 2. Probabilidades e xG
col_res1, col_res2, col_res3, col_res4 = st.columns(4)
col_res1.metric(f"Vitória {home_team}", f"{prediction['Casa']:.1%}")
col_res2.metric("Empate", f"{prediction['Empate']:.1%}")
col_res3.metric(f"Vitória {away_team}", f"{prediction['Fora']:.1%}")
col_res4.metric("Total xG", f"{prediction['xG_Casa'] + prediction['xG_Fora']:.2f}")

# 3. Mercados Principais
st.markdown("#### Principais Mercados")
col_m1, col_m2, col_m3 = st.columns(3)
col_m1.metric("Over 2.5 Golos", f"{prediction['Over2.5']:.1%}", f"Odd: {fair_odds(prediction['Over2.5']):.2f}")
col_m2.metric("Under 2.5 Golos", f"{prediction['Under2.5']:.1%}", f"Odd: {fair_odds(prediction['Under2.5']):.2f}")
col_m3.metric("Ambas Marcam (BTTS)", f"{prediction['BTTS']:.1%}", f"Odd: {fair_odds(prediction['BTTS']):.2f}")

# 4. Sugestão Automática
st.markdown("---")
st.markdown("### Sugestão de Análise")
probs = {
    "Vitória Casa (1)": prediction["Casa"],
    "Empate (X)": prediction["Empate"],
    "Vitória Fora (2)": prediction["Fora"],
    "Over 2.5 Golos": prediction["Over2.5"],
    "Under 2.5 Golos": prediction["Under2.5"],
    "Ambas Marcam (BTTS) - Sim": prediction["BTTS"],
    "Ambas Marcam (BTTS) - Não": prediction["NoBTTS"]
}
melhor_mercado = max(probs, key=probs.get)
melhor_prob = probs[melhor_mercado]

st.info(f"O modelo aponta *{melhor_mercado}* como a tendência mais forte, com probabilidade de *{melhor_prob:.1%}* (Odd Justa: *{fair_odds(melhor_prob):.2f}*).")
if melhor_prob > 0.60:
    st.success("Probabilidade alta. Procura no mercado odds superiores à 'Odd Justa' para encontrares valor.")
else:
    st.warning("Probabilidade moderada. Tem cuidado com odds baixas e valida na calculadora de EV.")

# 5. Separadores Técnicos
st.markdown("---")
tab_analise, tab_tecnico = st.tabs(["Análise Detalhada", "Cálculo de EV"])

with tab_analise:
    st.subheader("Matriz de Resultados Exatos (Top 5)")
    matrix_df = make_score_matrix(prediction["Matriz"], display_max=5)
    st.dataframe(matrix_df.style.format("{:.1%}").background_gradient(axis=None), use_container_width=True)

with tab_tecnico:
    st.subheader("Calculadora de Valor (EV)")
    st.write("Insere as odds da tua casa de apostas para veres se há valor.")
    col_o1, col_o2 = st.columns(2)
    with col_o1:
        odd_casa = st.number_input(f"Odd {home_team}", min_value=1.01, value=2.00, step=0.05)
        ev_casa = (prediction['Casa'] * odd_casa) - 1
        st.write(f"EV: *{ev_casa:+.1%}* {'Positivo ✅' if ev_casa > 0 else 'Negativo ❌'}")
        odd_btts = st.number_input("Odd BTTS Sim", min_value=1.01, value=1.80, step=0.05)
        ev_btts = (prediction['BTTS'] * odd_btts) - 1
        st.write(f"EV: *{ev_btts:+.1%}* {'Positivo ✅' if ev_btts > 0 else 'Negativo ❌'}")
    with col_o2:
        odd_fora = st.number_input(f"Odd {away_team}", min_value=1.01, value=2.50, step=0.05)
        ev_fora = (prediction['Fora'] * odd_fora) - 1
        st.write(f"EV: *{ev_fora:+.1%}* {'Positivo ✅' if ev_fora > 0 else 'Negativo ❌'}")
        odd_under = st.number_input("Odd Under 2.5", min_value=1.01, value=1.90, step=0.05)
        ev_under = (prediction['Under2.5'] * odd_under) - 1
        st.write(f"EV: *{ev_under:+.1%}* {'Positivo ✅' if ev_under > 0 else 'Negativo ❌'}")
