import math
from datetime import date

import numpy as np
import pandas as pd
import requests
import streamlit as st
from scipy.optimize import minimize
from scipy.special import gammaln

# ============================================================
# AI FOOTBALL ANALYTICS — v2 (Interface Limpa e Focada)
# ============================================================

np.random.seed(42)

st.set_page_config(page_title="AI Football Analytics", layout="wide")
st.title("⚽ AI Football Analytics")
st.caption("Poisson + Dixon-Coles | Análise Rápida e Sugestão de Valor")

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

# ---------------------------- API e dados ----------------------------
@st.cache_data(ttl=600, show_spinner=False)
def fetch_competition_data(competition_code: str, season: int, api_key: str):
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
            if match.get("status") != "FINISHED" or home_goals is None or away_goals is None:
                continue
            rows.append({
                "Data": pd.to_datetime(match["utcDate"], utc=True).date(),
                "Casa": match.get("homeTeam", {}).get("name", "Desconhecida"),
                "Fora": match.get("awayTeam", {}).get("name", "Desconhecida"),
                "Golos_Casa": int(home_goals),
                "Golos_Fora": int(away_goals),
                "Match_ID": match.get("id"),
            })
        matches_df = pd.DataFrame(rows)
        if not matches_df.empty:
            matches_df["Data"] = pd.to_datetime(matches_df["Data"]).dt.date
            matches_df = matches_df.drop_duplicates(subset=["Match_ID"]).sort_values("Data").reset_index(drop=True)
        
        standings = {}
        return matches_df, standings, None
    except Exception as exc:
        return pd.DataFrame(), {}, f"Erro ao obter dados: {exc}"

def prepare_matches(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty: return pd.DataFrame()
    result = df.copy()
    result["Data"] = pd.to_datetime(result["Data"], errors="coerce").dt.date
    for col in ("Golos_Casa", "Golos_Fora"):
        result[col] = pd.to_numeric(result[col], errors="coerce")
    result = result.dropna(subset=["Data", "Casa", "Fora", "Golos_Casa", "Golos_Fora"])
    result = result[(result["Golos_Casa"] >= 0) & (result["Golos_Fora"] >= 0) & (result["Casa"] != result["Fora"])].copy()
    result["Golos_Casa"] = result["Golos_Casa"].astype(int)
    result["Golos_Fora"] = result["Golos_Fora"].astype(int)
    return result.sort_values("Data").reset_index(drop=True)

# ---------------------------- Modelo ----------------------------
def time_weights(dates, reference_date=None, xi=DEFAULT_XI):
    if reference_date is None: reference_date = date.today()
    day_values = pd.to_datetime(pd.Series(dates)).dt.date
    age_days = np.array([(reference_date - d).days for d in day_values], dtype=float)
    age_days = np.maximum(age_days, 0.0)
    return np.exp(-xi * age_days)

def dc_tau(home_goals, away_goals, lam, mu, rho):
    if home_goals == 0 and away_goals == 0: return 1.0 - lam * mu * rho
    if home_goals == 0 and away_goals == 1: return 1.0 + lam * rho
    if home_goals == 1 and away_goals == 0: return 1.0 + mu * rho
    if home_goals == 1 and away_goals == 1: return 1.0 - rho
    return 1.0

def fit_dixon_coles(df: pd.DataFrame, xi=DEFAULT_XI, rho_limit=DEFAULT_RHO_LIMIT):
    data = prepare_matches(df)
    if len(data) < 20: raise ValueError("Precisas de pelo menos 20 jogos.")

    teams = sorted(set(data["Casa"]).union(data["Fora"]))
    team_to_idx = {team: idx for idx, team in enumerate(teams)}
    home_idx = data["Casa"].map(team_to_idx).to_numpy(dtype=int)
    away_idx = data["Fora"].map(team_to_idx).to_numpy(dtype=int)
    gh = data["Golos_Casa"].to_numpy(dtype=int)
    ga = data["Golos_Fora"].to_numpy(dtype=int)
    weights = time_weights(data["Data"], xi=xi)
    n_teams = len(teams)

    initial = np.zeros(2 * n_teams + 3)
    initial[2 * n_teams] = math.log(1.10)
    initial[2 * n_teams + 1] = math.log(1.05)
    initial[2 * n_teams + 2] = -0.05

    def objective(params):
        attack = params[:n_teams] - np.mean(params[:n_teams])
        defence = params[n_teams:2 * n_teams] - np.mean(params[n_teams:2 * n_teams])
        gamma_att = math.exp(params[2 * n_teams])
        gamma_def = math.exp(params[2 * n_teams + 1])
        rho = params[2 * n_teams + 2]

        log_lam = attack[home_idx] + defence[away_idx] + math.log(gamma_att)
        log_mu = attack[away_idx] + defence[home_idx] - math.log(gamma_def)
        lam = np.clip(np.exp(np.clip(log_lam, -3.0, 2.0)), 0.05, 8.0)
        mu = np.clip(np.exp(np.clip(log_mu, -3.0, 2.0)), 0.05, 8.0)

        log_prob = gh * np.log(lam) - lam - gammaln(gh + 1) + ga * np.log(mu) - mu - gammaln(ga + 1)
        tau = np.ones(len(data), dtype=float)
        both_zero = (gh == 0) & (ga == 0)
        home_zero_away_one = (gh == 0) & (ga == 1)
        home_one_away_zero = (gh == 1) & (ga == 0)
        both_one = (gh == 1) & (ga == 1)
        
        tau[both_zero] = 1 - lam[both_zero] * mu[both_zero] * rho
        tau[home_zero_away_one] = 1 + lam[home_zero_away_one] * rho
        tau[home_one_away_zero] = 1 + mu[home_one_away_zero] * rho
        tau[both_one] = 1 - rho
        if np.any(tau <= 1e-8): return 1e6
        log_prob += np.log(tau)
        penalty = 0.01 * (np.sum(attack * 2) + np.sum(defence * 2))
        return -float(np.sum(weights * log_prob)) + penalty

    bounds = [(-2.5, 2.5)] * (2 * n_teams) + [(math.log(0.75), math.log(1.80)), (math.log(0.75), math.log(1.80)), (-rho_limit, rho_limit)]
    result = minimize(objective, initial, method="L-BFGS-B", bounds=bounds, options={"maxiter": 2000})

    params = result.x
    attack = params[:n_teams] - np.mean(params[:n_teams])
    defence = params[n_teams:2 * n_teams] - np.mean(params[n_teams:2 * n_teams])
    return {
        "teams": teams,
        "attack": dict(zip(teams, np.exp(attack))),
        "defence": dict(zip(teams, np.exp(defence))),
        "gamma_att": float(math.exp(params[2 * n_teams])),
        "gamma_def": float(math.exp(params[2 * n_teams + 1])),
        "home_adv": float(math.exp(params[2 * n_teams] + params[2 * n_teams + 1])),
        "rho": float(params[2 * n_teams + 2]),
        "n_matches": len(data),
        "fit_success": bool(result.success)
    }

def poisson_pmf(k, lam):
    return math.exp(-lam + k * math.log(lam) - math.lgamma(k + 1)) if lam > 0 else (1.0 if k == 0 else 0.0)

def predict_match(home, away, model, max_goals=DEFAULT_MAX_GOALS):
    lam = float(np.clip(model["attack"][home] * model["defence"][away] * model["gamma_att"], 0.05, 6.0))
    mu = float(np.clip(model["attack"][away] * model["defence"][home] / model["gamma_def"], 0.05, 6.0))

    home_probs = np.array([poisson_pmf(i, lam) for i in range(max_goals + 1)])
    away_probs = np.array([poisson_pmf(j, mu) for j in range(max_goals + 1)])
    matrix = np.outer(home_probs, away_probs)

    rho = model["rho"]
    matrix[0, 0] *= dc_tau(0, 0, lam, mu, rho)
    matrix[0, 1] *= dc_tau(0, 1, lam, mu, rho)
    matrix[1, 0] *= dc_tau(1, 0, lam, mu, rho)
    matrix[1, 1] *= dc_tau(1, 1, lam, mu, rho)
    matrix /= matrix.sum()

    totals = np.add.outer(np.arange(max_goals + 1), np.arange(max_goals + 1))
    btts = float(matrix[1:, 1:].sum())
    return {
        "Casa": float(np.tril(matrix, -1).sum()),
        "Empate": float(np.trace(matrix)),
        "Fora": float(np.triu(matrix, 1).sum()),
        "Over2.5": float(matrix[totals >= 3].sum()),
        "Under2.5": float(matrix[totals <= 2].sum()),
        "BTTS": btts,
        "NoBTTS": 1.0 - btts,
        "xG_Casa": lam,
        "xG_Fora": mu,
        "Matriz": matrix
    }

def fair_odds(p): return 1.0 / p if p > 0 else float("inf")

def get_recent_form(df, team, n=8):
    team_matches = df[(df["Casa"] == team) | (df["Fora"] == team)].sort_values("Data").tail(n)
    form = []
    for _, m in team_matches.iterrows():
        if m["Casa"] == team:
            res = "V" if m["Golos_Casa"] > m["Golos_Fora"] else ("E" if m["Golos_Casa"] == m["Golos_Fora"] else "D")
        else:
            res = "V" if m["Golos_Fora"] > m["Golos_Casa"] else ("E" if m["Golos_Fora"] == m["Golos_Casa"] else "D")
        form.append(res)
    return form

def make_score_matrix(matrix, display_max=5):
    m = matrix[:display_max + 1, :display_max + 1]
    return pd.DataFrame(m, index=[f"Casa {i}" for i in range(m.shape[0])], columns=[f"Fora {j}" for j in range(m.shape[1])])

# ---------------------------- Barra Lateral ----------------------------
with st.sidebar:
    st.header("1. Dados")
    try:
        api_key = st.secrets["API_KEY"]
        st.success("API carregada")
    except Exception:
        api_key = st.text_input("API Key", type="password")

    competition_name = st.selectbox("Campeonato", list(COMPETITIONS.keys()))
    competition_code = COMPETITIONS[competition_name]
    current_year = date.today().year
    season = st.number_input("Ano", min_value=2010, max_value=current_year, value=current_year if date.today().month >= 7 else current_year - 1)
    reload_button = st.button("🔄 Atualizar dados", use_container_width=True)

if not api_key:
    st.info("Insere a chave da API.")
    st.stop()

if reload_button: fetch_competition_data.clear()

with st.spinner("A obter jogos..."):
    raw_df, standings, api_error = fetch_competition_data(competition_code, int(season), api_key)

if api_error:
    st.error(api_error)
    st.stop()

df = prepare_matches(raw_df)
if df.empty or len(df) < 20:
    st.warning("Poucos dados para ajustar o modelo.")
    st.stop()

with st.spinner("A ajustar modelo..."):
    model = fit_dixon_coles(df)

teams = sorted(model["teams"])
col_home, col_away = st.columns(2)
home_team = col_home.selectbox("🏠 Equipa da casa", teams)
away_options = [t for t in teams if t != home_team]
away_team = col_away.selectbox("✈️ Equipa visitante", away_options)

prediction = predict_match(home_team, away_team, model)

# ---------------------------- Nova Interface Limpa ----------------------------
st.markdown("---")
st.markdown(f"### 🏟️ {home_team} vs {away_team}")

# 1. Forma (Últimos 8)
col_f1, col_f2 = st.columns(2)
with col_f1:
    st.markdown(f"*Forma {home_team}:*")
    form_home = get_recent_form(df, home_team, 8)
    st.markdown(" ".join([f"🟢" if r == "V" else (f"🟡" if r == "E" else f"🔴") for r in form_home]))
with col_f2:
    st.markdown(f"*Forma {away_team}:*")
    form_away = get_recent_form(df, away_team, 8)
    st.markdown(" ".join([f"🟢" if r == "V" else (f"🟡" if r == "E" else f"🔴") for r in form_away]))

st.markdown("---")

# 2. Probabilidades e xG
col_res1, col_res2, col_res3, col_res4 = st.columns(4)
col_res1.metric(f"Vitória {home_team}", f"{prediction['Casa']:.1%}")
col_res2.metric("Empate", f"{prediction['Empate']:.1%}")
col_res3.metric(f"Vitória {away_team}", f"{prediction['Fora']:.1%}")
col_res4.metric("Total xG", f"{prediction['xG_Casa'] + prediction['xG_Fora']:.2f}")

# 3. Mercados Principais
st.markdown("#### 🎯 Principais Mercados")
col_m1, col_m2, col_m3 = st.columns(3)
col_m1.metric("Over 2.5 Golos", f"{prediction['Over2.5']:.1%}", f"Odd: {fair_odds(prediction['Over2.5']):.2f}")
col_m2.metric("Under 2.5 Golos", f"{prediction['Under2.5']:.1%}", f"Odd: {fair_odds(prediction['Under2.5']):.2f}")
col_m3.metric("Ambas Marcam (BTTS)", f"{prediction['BTTS']:.1%}", f"Odd: {fair_odds(prediction['BTTS']):.2f}")

# 4. Sugestão Automática
st.markdown("---")
st.markdown("### 💡 Sugestão de Análise")
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
tab_analise, tab_tecnico = st.tabs(["📊 Análise Detalhada", "⚙️ Cálculo de EV"])

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
        st.write(f"EV: *{ev_casa:+.1%}* {'✅ Positivo' if ev_casa > 0 else '❌ Negativo'}")
        odd_btts = st.number_input("Odd BTTS Sim", min_value=1.01, value=1.80, step=0.05)
        ev_btts = (prediction['BTTS'] * odd_btts) - 1
        st.write(f"EV: *{ev_btts:+.1%}* {'✅ Positivo' if ev_btts > 0 else '❌ Negativo'}")
    with col_o2:
        odd_fora = st.number_input(f"Odd {away_team}", min_value=1.01, value=2.50, step=0.05)
        ev_fora = (prediction['Fora'] * odd_fora) - 1
        st.write(f"EV: *{ev_fora:+.1%}* {'✅ Positivo' if ev_fora > 0 else '❌ Negativo'}")
        odd_under = st.number_input("Odd Under 2.5", min_value=1.01, value=1.90, step=0.05)
        ev_under = (prediction['Under2.5'] * odd_under) - 1
        st.write(f"EV: *{ev_under:+.1%}* {'✅ Positivo' if ev_under > 0 else '❌ Negativo'}")
