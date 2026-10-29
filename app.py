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

# ---------------------------- Visão geral ----------------------------
st.markdown("### Resumo dos dados")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Jogos concluídos", f"{len(df)}")
m2.metric("Equipas encontradas", f"{len(set(df['Casa']) | set(df['Fora']))}")
m3.metric("Primeiro jogo", str(min(df["Data"])))
m4.metric("Último jogo", str(max(df["Data"])))
st.caption("Os resultados são os disponibilizados pela API. Confirma a cobertura da época e do teu plano de API.")

if len(df) < 20:
    st.stop()

with st.spinner("A ajustar o modelo por máxima verosimilhança..."):
    try:
        model = fit_dixon_coles(df, xi=xi)
    except Exception as exc:
        st.error(f"Falha no ajuste do modelo: {exc}")
        st.stop()

if not model["fit_success"]:
    st.warning(f"O otimizador não convergiu plenamente: {model['fit_message']}")

with st.sidebar:
    st.markdown("---")
    st.metric("Vantagem de casa estimada", f"{model['home_adv']:.3f}")
    st.caption(
        f"Decomposta: γ_att={model['gamma_att']:.3f} (ataque casa) × "
        f"γ_def={model['gamma_def']:.3f} (defesa fora)"
    )
    st.metric("Rho Dixon-Coles estimado", f"{model['rho']:.3f}")
    st.metric("Média ponderada golos casa", f"{model['league_home_goals']:.2f}")
    st.metric("Média ponderada golos fora", f"{model['league_away_goals']:.2f}")
    st.caption(f"Objetivo final (nll + reg): {model['objective']:.2f}")

teams = sorted(model["teams"])
col_home, col_away = st.columns(2)
home_team = col_home.selectbox("🏠 Equipa da casa", teams, key="home_team")
away_options = [team for team in teams if team != home_team]
away_team = col_away.selectbox("✈️ Equipa visitante", away_options, key="away_team")

if home_team not in model["teams"] or away_team not in model["teams"]:
    st.error("As duas equipas têm de estar presentes nos jogos usados para treinar o modelo.")
    st.stop()

prediction = predict_match(home_team, away_team, model, max_goals=max_goals)

# Info de standings (agora usada)
def standing_info(team):
    if team in standings and standings[team].get("Posicao") is not None:
        s = standings[team]
        return f"#{s['Posicao']} · {s.get('Pontos', '?')} pts · {s.get('Jogos_Feitos', '?')} jogos"
    return "Sem classificação disponível"

st.markdown(f"## {home_team} vs {away_team}")
st.caption(f"🏠 {standing_info(home_team)}  |  ✈️ {standing_info(away_team)}")

x1, x2, x3 = st.columns(3)
x1.metric(f"xG {home_team}", f"{prediction['xG_Casa']:.2f}")
x2.metric(f"xG {away_team}", f"{prediction['xG_Fora']:.2f}")
x3.metric("Total xG", f"{prediction['xG_Casa'] + prediction['xG_Fora']:.2f}")

st.caption(
    f"Modelo ajustado com {model['n_matches']} jogos até {model['last_date']}. "
    f"Estes xG são estimativas do modelo, não xG oficiais de fornecedores."
)

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📊 Probabilidades", "🎯 Odds e EV", "📈 Equipas e matriz", "🧪 Backtesting", "📐 Calibração"]
)

with tab1:
    st.subheader("Mercado 1X2")
    c1, c2, c3 = st.columns(3)
    c1.metric("Vitória casa (1)", f"{prediction['Casa']:.1%}",
              f"Odd justa {fair_odds(prediction['Casa']):.2f}")
    c2.metric("Empate (X)", f"{prediction['Empate']:.1%}",
              f"Odd justa {fair_odds(prediction['Empate']):.2f}")
    c3.metric("Vitória fora (2)", f"{prediction['Fora']:.1%}",
              f"Odd justa {fair_odds(prediction['Fora']):.2f}")

    st.subheader("Mercados de golos")
    market_rows = [
        ("Over 0.5", prediction["Over0.5"]),
        ("Over 1.5", prediction["Over1.5"]),
        ("Over 2.5", prediction["Over2.5"]),
        ("Over 3.5", prediction["Over3.5"]),
        ("Under 1.5", prediction["Under1.5"]),
        ("Under 2.5", prediction["Under2.5"]),
        ("Under 3.5", prediction["Under3.5"]),
        ("BTTS — Sim", prediction["BTTS"]),
        ("BTTS — Não", prediction["NoBTTS"]),
    ]
    markets_df = pd.DataFrame([
        {"Mercado": name, "Probabilidade": prob, "Odd justa": fair_odds(prob)}
        for name, prob in market_rows
    ])
    st.dataframe(
        markets_df.style.format({"Probabilidade": "{:.1%}", "Odd justa": "{:.2f}"}),
        use_container_width=True, hide_index=True
    )
    if prediction["Massa_cauda_aproximada"] > 0.005:
        st.warning(
            f"A matriz truncada deixa aproximadamente {prediction['Massa_cauda_aproximada']:.2%} "
            "de massa de probabilidade fora do intervalo antes da normalização. "
            "Aumenta o máximo de golos e verifica se o resultado estabiliza."
        )
    else:
        st.caption("A massa de probabilidade fora da matriz é pequena para este jogo.")

with tab2:
    st.write("Introduz as odds decimais disponíveis. O EV é uma estimativa bruta e não inclui a incerteza do modelo.")
    market_options = {
        "Vitória casa (1)": "Casa",
        "Empate (X)": "Empate",
        "Vitória fora (2)": "Fora",
        "Over 1.5": "Over1.5",
        "Over 2.5": "Over2.5",
        "Over 3.5": "Over3.5",
        "Under 1.5": "Under1.5",
        "Under 2.5": "Under2.5",
        "Under 3.5": "Under3.5",
        "BTTS — Sim": "BTTS",
        "BTTS — Não": "NoBTTS",
    }
    input_rows = []
    cols = st.columns(3)
    for idx, (market_name, key) in enumerate(market_options.items()):
        with cols[idx % 3]:
            odd = st.number_input(
                f"Odd — {market_name}", min_value=1.01, max_value=1000.0,
                value=2.00, step=0.01, key=f"odd_{key}"
            )
        p = prediction[key]
        ev = expected_value(p, odd)
        input_rows.append({
            "Mercado": market_name,
            "Probabilidade modelo": p,
            "Odd introduzida": odd,
            "Odd justa": fair_odds(p),
            "EV bruto": ev,
            "Estado indicativo": odds_status(ev),
        })
    odds_df = pd.DataFrame(input_rows)
    st.dataframe(
        odds_df.style.format({
            "Probabilidade modelo": "{:.1%}", "Odd introduzida": "{:.2f}",
            "Odd justa": "{:.2f}", "EV bruto": "{:+.1%}"
        }),
        use_container_width=True, hide_index=True
    )
    st.warning(
        "Um EV positivo calculado não prova que exista uma vantagem real. "
        "Valida primeiro a calibração e o desempenho histórico fora da amostra; "
        "não uses o Kelly como recomendação automática de aposta."
    )

with tab3:
    st.subheader("Forças estimadas das equipas")
    strength_rows = []
    for team in teams:
        strength_rows.append({
            "Equipa": team,
            "Ataque relativo": model["attack"][team],
            "Propensão defensiva": model["defence"][team],
            "Jogos em casa": int((df["Casa"] == team).sum()),
            "Jogos fora": int((df["Fora"] == team).sum()),
        })
    strength_df = pd.DataFrame(strength_rows).sort_values("Ataque relativo", ascending=False)
    st.dataframe(
        strength_df.style.format({
            "Ataque relativo": "{:.3f}", "Propensão defensiva": "{:.3f}"
        }),
        use_container_width=True, hide_index=True
    )
    st.caption(
        "Ataque relativo acima de 1 indica força ofensiva acima da média estimada. "
        "Propensão defensiva acima de 1 indica tendência para sofrer mais golos; "
        "abaixo de 1 indica uma defesa mais forte."
    )
    st.subheader("Matriz de resultados (0–6; o resto é agregado no cálculo)")
    matrix_df = make_score_matrix(prediction["Matriz"], display_max=6)
    st.dataframe(
        matrix_df.style.format("{:.1%}").background_gradient(axis=None),
        use_container_width=True
    )
    score_prob = []
    for i in range(prediction["Matriz"].shape[0]):
        for j in range(prediction["Matriz"].shape[1]):
            score_prob.append((i, j, prediction["Matriz"][i, j]))
    top_scores = sorted(score_prob, key=lambda x: x[2], reverse=True)[:5]
    st.write("**Cinco resultados exatos mais prováveis**")
    st.dataframe(
        pd.DataFrame([
            {"Resultado": f"{h}-{a}", "Probabilidade": p}
            for h, a, p in top_scores
        ]).style.format({"Probabilidade": "{:.1%}"}),
        use_container_width=True, hide_index=True
    )

with tab4:
    st.subheader("Teste cronológico fora da amostra")
    st.write(
        "O holdout treina na parte inicial e avalia nos jogos posteriores. "
        "O walk-forward reajusta periodicamente e avalia em todos os jogos após o período mínimo. "
        "O walk-forward é o teste mais informativo."
    )

    col_a, col_b = st.columns(2)
    if col_a.button("Executar holdout", type="primary"):
        with st.spinner("A executar holdout..."):
            backtest, backtest_error = run_chronological_backtest(
                df, xi=xi, train_fraction=train_fraction, max_goals=max_goals
            )
        if backtest_error:
            st.error(backtest_error)
        else:
            metrics = backtest["metrics"]
            st.markdown("**Holdout — modelo DC vs. baseline de frequências**")
            a, b, c, d = st.columns(4)
            a.metric("Jogos de teste", metrics["Jogos teste avaliados"])
            b.metric("Brier 1X2", f"{metrics['Brier 1X2']:.4f}",
                     f"baseline {metrics['Brier 1X2 (baseline)']:.4f}")
            c.metric("LogLoss 1X2", f"{metrics['LogLoss 1X2']:.4f}",
                     f"baseline {metrics['LogLoss 1X2 (baseline)']:.4f}")
            d.metric("Acurácia 1X2", f"{metrics['Acurácia 1X2']:.1%}")
            e, f, g = st.columns(3)
            e.metric("Brier Over 2.5", f"{metrics['Brier Over 2.5']:.4f}",
                     f"baseline {metrics['Brier Over 2.5 (baseline)']:.4f}")
            f.metric("Brier BTTS", f"{metrics['Brier BTTS']:.4f}",
                     f"baseline {metrics['Brier BTTS (baseline)']:.4f}")
            g.metric("Jogos de treino", metrics["Jogos treino"])
            st.caption(
                "Valores mais baixos são melhores. A comparação com o baseline é o que dá "
                "significado ao número absoluto: se o modelo não bater o baseline, não está a acrescentar valor."
            )
            st.dataframe(
                backtest["results"].drop(columns=["Real_1X2", "Real_Over25", "Real_BTTS",
                                                  "Brier_1X2_individual"]),
                use_container_width=True, hide_index=True
            )
            csv = backtest["results"].to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "Descarregar resultados do holdout (CSV)",
                data=csv, file_name="backtest_holdout.csv", mime="text/csv",
            )

    if col_b.button("Executar walk-forward", type="primary"):
        with st.spinner("A executar walk-forward (pode demorar)..."):
            wf, wf_error = run_walk_forward(df, xi=xi, step=wf_step, max_goals=max_goals)
        if wf_error:
            st.error(wf_error)
        else:
            metrics = wf["metrics"]
            st.markdown("**Walk-forward — janela expansível**")
            a, b, c, d = st.columns(4)
            a.metric("Jogos avaliados", metrics["Jogos avaliados"])
            b.metric("Brier 1X2", f"{metrics['Brier 1X2']:.4f}")
            c.metric("LogLoss 1X2", f"{metrics['LogLoss 1X2']:.4f}")
            d.metric("Acurácia 1X2", f"{metrics['Acurácia 1X2']:.1%}")
            e, f = st.columns(2)
            e.metric("Brier Over 2.5", f"{metrics['Brier Over 2.5']:.4f}")
            f.metric("Brier BTTS", f"{metrics['Brier BTTS']:.4f}")
            st.dataframe(
                wf["results"].drop(columns=["Real_1X2", "Real_Over25", "Real_BTTS"]),
                use_container_width=True, hide_index=True
            )
            csv = wf["results"].to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "Descarregar resultados do walk-forward (CSV)",
                data=csv, file_name="backtest_walk_forward.csv", mime="text/csv",
            )

with tab5:
    st.subheader("Curva de calibração")
    st.write(
        "A calibração mostra se as probabilidades previstas correspondem às frequências observadas. "
        "Idealmente, os pontos ficam próximos da diagonal. Corre primeiro o holdout ou o walk-forward "
        "para gerar previsões fora da amostra."
    )
    st.info(
        "A calibração usa o último conjunto de previsões gerado (holdout ou walk-forward). "
        "Executa um dos testes no separador Backtesting antes de interpretar esta curva."
    )
    if st.button("Gerar calibração a partir do walk-forward", type="primary"):
        with st.spinner("A gerar previsões walk-forward para calibração..."):
            wf, wf_error = run_walk_forward(df, xi=xi, step=wf_step, max_goals=max_goals)
        if wf_error:
            st.error(wf_error)
        else:
            res = wf["results"]
            st.markdown("**1X2 — probabilidade da classe prevista vs. frequência observada**")
            # Usar a probabilidade da classe prevista (max) vs. acerto
            p_max = res[["P(1)", "P(X)", "P(2)"]].max(axis=1).to_numpy()
            y_hit = (res[["P(1)", "P(X)", "P(2)"]].to_numpy().argmax(axis=1)
                     == res["Real_1X2"].to_numpy()).astype(float)
            cal_1x2 = calibration_table(p_max, y_hit, n_bins=8)
            st.dataframe(
                cal_1x2.style.format({"Previsto": "{:.2%}", "Observado": "{:.2%}"}),
                use_container_width=True, hide_index=True
            )
            st.markdown("**Over 2.5**")
            cal_o25 = calibration_table(res["P(Over 2.5)"], res["Real_Over25"], n_bins=8)
            st.dataframe(
                cal_o25.style.format({"Previsto": "{:.2%}", "Observado": "{:.2%}"}),
                use_container_width=True, hide_index=True
            )
            st.markdown("**BTTS**")
            cal_btts = calibration_table(res["P(BTTS)"], res["Real_BTTS"], n_bins=8)
            st.dataframe(
                cal_btts.style.format({"Previsto": "{:.2%}", "Observado": "{:.2%}"}),
                use_container_width=True, hide_index=True
            )
            st.caption(
                "Bin com poucos jogos (N pequeno) tem observações instáveis — interpreta com cautela."
            )

with st.expander("Limitações importantes e interpretação"):
    st.markdown("""
    - O modelo usa resultados finais e não inclui lesões, alinhamentos, calendário congestionado,
      xG oficiais ou informação de mercado, salvo se forem integrados separadamente.
    - O holdout é uma divisão cronológica única; o walk-forward reajusta periodicamente e é
      o teste mais informativo. Mesmo assim, uma única época tem incerteza considerável.
    - Compara sempre o modelo com o **baseline de frequências** no mesmo conjunto de teste.
      Um Brier absoluto sem baseline não diz se o modelo acrescenta valor.
    - A época/competição e a cobertura de jogos dependem do plano da Football-Data.org.
    - As odds justas são o inverso das probabilidades do modelo e não incluem margem, custos,
      limites, variação de odds ou incerteza estatística.
    - A vantagem de casa é decomposta em `γ_att` (ataque casa) e `γ_def` (defesa fora),
      seguindo Dixon-Coles (1997). O produto `γ_att × γ_def` é a vantagem total.
    - Não há garantia de lucro. Antes de qualquer decisão, valida o modelo em várias épocas,
      ligas e períodos que não foram usados na afinação.
    """)
