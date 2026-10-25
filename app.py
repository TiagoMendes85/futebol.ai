import streamlit as st
import requests
import pandas as pd
import numpy as np
import math

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="AI Football Analytics PRO", layout="wide")
st.title("⚽ AI Football Analytics - Previsões Avançadas")
st.markdown("Análise preditiva com Dixon-Coles, Monte Carlo, Elo, Clean Sheets, FTS e Forma.")

# --- BARRA LATERAL ---
st.sidebar.header("1. Configurações")
try:
    API_KEY = st.secrets["API_KEY"]
    st.sidebar.success("✅ Chave carregada")
except:
    API_KEY = st.sidebar.text_input("API Key (Football-Data)", type="password")

LIGAS_DISPONIVEIS = {
    "Premier League (Inglaterra)": "PL",
    "Championship (2ª Ing.)": "ELC",
    "La Liga (Espanha)": "PD",
    "Serie A (Itália)": "SA",
    "Bundesliga (Alemanha)": "BL1",
    "Ligue 1 (França)": "FL1",
    "Primeira Liga (Portugal)": "PPL",
    "Eredivisie (Holanda)": "DED",
    "Brasileirão (Brasil)": "BSA",
    "Champions League": "CL"
}

liga_nome = st.sidebar.selectbox("🏆 Escolhe o Campeonato", list(LIGAS_DISPONIVEIS.keys()))
liga_codigo = LIGAS_DISPONIVEIS[liga_nome]

# --- FUNÇÕES DE DADOS ---
@st.cache_data(ttl=600)
def obter_dados_fd(codigo_liga, api_key):
    headers = {'X-Auth-Token': api_key}
    url_matches = f'https://api.football-data.org/v4/competitions/{codigo_liga}/matches'
    url_standings = f'https://api.football-data.org/v4/competitions/{codigo_liga}/standings'
    
    res_matches = requests.get(url_matches, headers=headers)
    res_standings = requests.get(url_standings, headers=headers)
    
    if res_matches.status_code != 200:
        return None, None
        
    dados_matches = res_matches.json()
    dados_standings = res_standings.json() if res_standings.status_code == 200 else None
    
    jogos = []
    for match in dados_matches['matches']:
        if match['status'] == 'FINISHED':
            gc = match['score']['fullTime']['home']
            gf = match['score']['fullTime']['away']
            if gc is not None and gf is not None:
                jogos.append({
                    'Data': pd.to_datetime(match['utcDate']).date(),
                    'Casa': match['homeTeam']['name'],
                    'Fora': match['awayTeam']['name'],
                    'Golos_Casa': int(gc),
                    'Golos_Fora': int(gf)
                })
    df = pd.DataFrame(jogos)
    
    tabela = {}
    if dados_standings:
        for table in dados_standings['standings']:
            if table['type'] == 'TOTAL':
                for entry in table['table']:
                    tabela[entry['team']['name']] = {
                        'Posicao': entry['position'],
                        'Pontos': entry['points'],
                        'Jogos_Feitos': entry['playedGames']
                    }
    return df, tabela

# --- MOTOR MATEMÁTICO ---
def calcular_metricas(df):
    if df.empty: return None, None, None
    
    df = df.sort_values('Data')
    
    # Elo
    elo_ratings = {equipa: 1500 for equipa in set(df['Casa']).union(set(df['Fora']))}
    K = 25
    for _, jogo in df.iterrows():
        casa, fora = jogo['Casa'], jogo['Fora']
        expectativa_casa = 1 / (1 + 10**((elo_ratings[fora] - elo_ratings[casa]) / 400))
        if jogo['Golos_Casa'] > jogo['Golos_Fora']: res = 1
        elif jogo['Golos_Casa'] == jogo['Golos_Fora']: res = 0.5
        else: res = 0
        elo_ratings[casa] += K * (res - expectativa_casa)
        elo_ratings[fora] += K * ((1 - res) - (1 - expectativa_casa))
    
    df['Peso'] = df.index.map(lambda x: 2.5 if x >= len(df) - 15 else 1.0)
    
    media_golos_casa = np.average(df['Golos_Casa'], weights=df['Peso'])
    media_golos_fora = np.average(df['Golos_Fora'], weights=df['Peso'])
    
    if media_golos_casa == 0: media_golos_casa = 1.4
    if media_golos_fora == 0: media_golos_fora = 1.2
    
    stats = {}
    equipas = set(df['Casa']).union(set(df['Fora']))
    
    for equipa in equipas:
        jogos_casa = df[df['Casa'] == equipa]
        jogos_fora = df[df['Fora'] == equipa]
        
        jogos_c = len(jogos_casa)
        jogos_f = len(jogos_fora)
        peso_c = jogos_c / (jogos_c + 5.0)
        peso_f = jogos_f / (jogos_f + 5.0)
        
        ataque_casa_bruto = np.average(jogos_casa['Golos_Casa'], weights=jogos_casa['Peso']) / media_golos_casa if jogos_c > 0 else 1.0
        defesa_casa_bruto = np.average(jogos_casa['Golos_Fora'], weights=jogos_casa['Peso']) / media_golos_fora if jogos_c > 0 else 1.0
        ataque_fora_bruto = np.average(jogos_fora['Golos_Fora'], weights=jogos_fora['Peso']) / media_golos_fora if jogos_f > 0 else 1.0
        defesa_fora_bruto = np.average(jogos_fora['Golos_Casa'], weights=jogos_fora['Peso']) / media_golos_casa if jogos_f > 0 else 1.0
        
        ataque_casa = (ataque_casa_bruto * peso_c) + (1.0 * (1 - peso_c))
        defesa_casa = (defesa_casa_bruto * peso_c) + (1.0 * (1 - peso_c))
        ataque_fora = (ataque_fora_bruto * peso_f) + (1.0 * (1 - peso_f))
        defesa_fora = (defesa_fora_bruto * peso_f) + (1.0 * (1 - peso_f))
        
        # Clean Sheets e Failed to Score (calculados a partir dos resultados da Football-Data)
        cs_casa = (jogos_casa['Golos_Fora'] == 0).sum() / jogos_c if jogos_c > 0 else 0.25
        cs_fora = (jogos_fora['Golos_Casa'] == 0).sum() / jogos_f if jogos_f > 0 else 0.25
        fts_casa = (jogos_casa['Golos_Casa'] == 0).sum() / jogos_c if jogos_c > 0 else 0.25
        fts_fora = (jogos_fora['Golos_Fora'] == 0).sum() / jogos_f if jogos_f > 0 else 0.25
            
        # Forma e Fator Momento
        ultimos_jogos = df[(df['Casa'] == equipa) | (df['Fora'] == equipa)].tail(5)
        forma = ""
        pontos_forma = 0
        for _, jogo in ultimos_jogos.iterrows():
            if jogo['Casa'] == equipa:
                if jogo['Golos_Casa'] > jogo['Golos_Fora']: forma += "V "; pontos_forma += 3
                elif jogo['Golos_Casa'] == jogo['Golos_Fora']: forma += "E "; pontos_forma += 1
                else: forma += "D "
            else:
                if jogo['Golos_Fora'] > jogo['Golos_Casa']: forma += "V "; pontos_forma += 3
                elif jogo['Golos_Fora'] == jogo['Golos_Casa']: forma += "E "; pontos_forma += 1
                else: forma += "D "
        
        fator_momento = 0.8 + (pontos_forma / 15.0) * 0.4

        stats[equipa] = {
            'Ataque_Casa': ataque_casa * fator_momento, 
            'Defesa_Casa': defesa_casa / fator_momento,
            'Ataque_Fora': ataque_fora * fator_momento, 
            'Defesa_Fora': defesa_fora / fator_momento,
            'Forma': forma.strip(), 'Jogos_Casa': jogos_c, 'Jogos_Fora': jogos_f,
            'CS_Casa': cs_casa, 'CS_Fora': cs_fora,
            'FTS_Casa': fts_casa, 'FTS_Fora': fts_fora,
            'Elo': elo_ratings[equipa],
            'Fator_Momento': fator_momento
        }
    return stats, media_golos_casa, media_golos_fora

def prever_jogo_monte_carlo(casa, fora, stats, med_casa, med_fora, sims=10000):
    lambda_casa = stats[casa]['Ataque_Casa'] * stats[fora]['Defesa_Fora'] * med_casa
    lambda_fora = stats[fora]['Ataque_Fora'] * stats[casa]['Defesa_Casa'] * med_fora
    
    # Ajuste com Clean Sheets (a equipa que sofre poucos golos reduz o lambda do adversário)
    lambda_casa *= (1.0 - (stats[fora]['CS_Fora'] * 0.4))
    lambda_fora *= (1.0 - (stats[casa]['CS_Casa'] * 0.4))

    # Ajuste com Failed to Score (a equipa que falha muito a marcar tem o lambda reduzido)
    lambda_casa *= (1.0 - (stats[casa]['FTS_Casa'] * 0.3))
    lambda_fora *= (1.0 - (stats[fora]['FTS_Fora'] * 0.3))

    # Ajuste com Elo
    diff_elo = stats[casa]['Elo'] - stats[fora]['Elo']
    lambda_casa *= (1 + (diff_elo / 1000))
    lambda_fora *= (1 - (diff_elo / 1000))

    golos_casa_sim = np.random.poisson(lambda_casa, sims)
    golos_fora_sim = np.random.poisson(lambda_fora, sims)
    total_golos_sim = golos_casa_sim + golos_fora_sim
    
    p_casa = np.mean(golos_casa_sim > golos_fora_sim)
    p_empate = np.mean(golos_casa_sim == golos_fora_sim)
    p_fora = np.mean(golos_casa_sim < golos_fora_sim)
    p_over15 = np.mean(total_golos_sim > 1.5)
    p_over25 = np.mean(total_golos_sim > 2.5)
    p_over35 = np.mean(total_golos_sim > 3.5)
    p_btts = np.mean((golos_casa_sim > 0) & (golos_fora_sim > 0))
    
    return {
        'Casa': p_casa, 'Empate': p_empate, 'Fora': p_fora, 
        'Over1.5': p_over15, 'Over2.5': p_over25, 'Over3.5': p_over35, 
        'BTTS': p_btts,
        'xG_Casa': lambda_casa, 'xG_Fora': lambda_fora
    }

def gerar_matriz_resultados(lambda_casa, lambda_fora, max_golos=5):
    def calc_poisson(k, lamb):
        return (math.exp(-lamb) * (lamb**k)) / math.factorial(k)
    matriz = np.zeros((max_golos, max_golos))
    for i in range(max_golos):
        for j in range(max_golos):
            matriz[i, j] = calc_poisson(i, lambda_casa) * calc_poisson(j, lambda_fora)
    index = [f"{i} Golos (Casa)" for i in range(max_golos)]
    columns = [f"{j} Golos (Fora)" for j in range(max_golos)]
    return pd.DataFrame(matriz, index=index, columns=columns)

def gerar_narrativa(casa, fora, stats, res):
    txt = f"O **{casa}** chega a este jogo com uma forma recente de `[{stats[casa]['Forma']}]`, enquanto o **{fora}** apresenta `[{stats[fora]['Forma']}]`. "
    txt += f"O modelo projeta um xG de **{res['xG_Casa']:.2f}** para o {casa} e **{res['xG_Fora']:.2f}** para o {fora}. "
    if res['Over2.5'] > 0.55:
        txt += f"Com {res['Over2.5']:.1%} de probabilidade para Mais de 2.5 Golos, espera-se um jogo aberto. "
    else:
        txt += f"A probabilidade de Mais de 2.5 Golos é de {res['Over2.5']:.1%}, sugerindo um jogo mais tático. "
    if res['BTTS'] > 0.55:
        txt += "Ambas as equipas têm argumentos para marcar."
    else:
        txt += "O mercado de 'Ambas Marcam' não é o mais favorável."
    return txt

# --- INTERFACE PRINCIPAL ---
if not API_KEY:
    st.warning("⚠️ Insere a tua API Key na barra lateral para começar.")
else:
    with st.spinner('A processar dados...'):
        df, tabela = obter_dados_fd(liga_codigo, API_KEY)
        
    if df is not None and not df.empty:
        stats, med_casa, med_fora = calcular_metricas(df)
        
        col1, col2 = st.columns(2)
        equipas = sorted(list(stats.keys()))
        casa = col1.selectbox("🏠 Equipa da Casa", equipas)
        fora = col2.selectbox("✈️ Equipa de Fora", equipas)
        
        if st.button("🚀 Gerar Análise Completa", use_container_width=True):
            if casa != fora:
                res = prever_jogo_monte_carlo(casa, fora, stats, med_casa, med_fora)
                
                st.markdown(f"## 🏆 Análise: {casa} vs {fora}")
                tab1, tab2, tab3 = st.tabs(["📊 Resumo e Odds", "📈 Estatísticas", "⚔️ H2H"])
                
                with tab1:
                    st.info(gerar_narrativa(casa, fora, stats, res))
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Vitória Casa (1)", f"{res['Casa']:.1%}")
                    c2.metric("Empate (X)", f"{res['Empate']:.1%}")
                    c3.metric("Vitória Fora (2)", f"{res['Fora']:.1%}")
                    
                    c4, c5, c6 = st.columns(3)
                    c4.metric("Mais de 1.5 Golos", f"{res['Over1.5']:.1%}")
                    c5.metric("Mais de 2.5 Golos", f"{res['Over2.5']:.1%}")
                    c6.metric("Mais de 3.5 Golos", f"{res['Over3.5']:.1%}")
                    st.metric("Ambas Marcam (BTTS)", f"{res['BTTS']:.1%}")
                    
                    st.markdown("### 💡 Odds Justas e Value Bet")
                    col_o1, col_o2, col_o3 = st.columns(3)
                    odd_1 = col_o1.number_input("Odd Casa (1)", min_value=1.01, step=0.01, value=float(round(1/res['Casa'], 2)))
                    odd_X = col_o2.number_input("Odd Empate (X)", min_value=1.01, step=0.01, value=float(round(1/res['Empate'], 2)))
                    odd_2 = col_o3.number_input("Odd Fora (2)", min_value=1.01, step=0.01, value=float(round(1/res['Fora'], 2)))
                    
                    col_o4, col_o5 = st.columns(2)
                    odd_over25 = col_o4.number_input("Odd Mais de 2.5 Golos", min_value=1.01, step=0.01, value=float(round(1/res['Over2.5'], 2)))
                    odd_btts = col_o5.number_input("Odd Ambas Marcam (BTTS)", min_value=1.01, step=0.01, value=float(round(1/res['BTTS'], 2)))
                    
                    ev_1 = (res['Casa'] * odd_1) - 1
                    ev_X = (res['Empate'] * odd_X) - 1
                    ev_2 = (res['Fora'] * odd_2) - 1
                    ev_over25 = (res['Over2.5'] * odd_over25) - 1
                    ev_btts = (res['BTTS'] * odd_btts) - 1
                    
                    kelly_1 = max(0, ((odd_1 * res['Casa']) - 1) / (odd_1 - 1)) if odd_1 > 1.01 else 0.0
                    kelly_X = max(0, ((odd_X * res['Empate']) - 1) / (odd_X - 1)) if odd_X > 1.01 else 0.0
                    kelly_2 = max(0, ((odd_2 * res['Fora']) - 1) / (odd_2 - 1)) if odd_2 > 1.01 else 0.0
                    kelly_over25 = max(0, ((odd_over25 * res['Over2.5']) - 1) / (odd_over25 - 1)) if odd_over25 > 1.01 else 0.0
                    kelly_btts = max(0, ((odd_btts * res['BTTS']) - 1) / (odd_btts - 1)) if odd_btts > 1.01 else 0.0
                    
                    def status_ev(ev, kelly):
                        if ev > 0.05: return f"🟢 Valor (Apostar {kelly:.1%})"
                        elif ev > -0.05: return "⚪ Neutro"
                        else: return "🔴 Sem Valor"
                    
                    odds_df = pd.DataFrame({
                        'Mercado': ['Vitória Casa', 'Empate', 'Vitória Fora', 'Mais de 2.5', 'BTTS'],
                        'Probabilidade': [f"{res['Casa']:.1%}", f"{res['Empate']:.1%}", f"{res['Fora']:.1%}", f"{res['Over2.5']:.1%}", f"{res['BTTS']:.1%}"],
                        'Odd Justa': [f"{1/res['Casa']:.2f}", f"{1/res['Empate']:.2f}", f"{1/res['Fora']:.2f}", f"{1/res['Over2.5']:.2f}", f"{1/res['BTTS']:.2f}"],
                        'Odd Inserida': [odd_1, odd_X, odd_2, odd_over25, odd_btts],
                        'EV': [f"{ev_1:.1%}", f"{ev_X:.1%}", f"{ev_2:.1%}", f"{ev_over25:.1%}", f"{ev_btts:.1%}"],
                        'Recomendação': [status_ev(ev_1, kelly_1), status_ev(ev_X, kelly_X), status_ev(ev_2, kelly_2), status_ev(ev_over25, kelly_over25), status_ev(ev_btts, kelly_btts)]
                    })
                    st.table(odds_df)

                with tab2:
                    st.markdown("### 📊 Comparação de Força (xG Esperado)")
                    chart_data = pd.DataFrame({'Equipa': [casa, fora], 'xG': [res['xG_Casa'], res['xG_Fora']]}).set_index('Equipa')
                    st.bar_chart(chart_data, color="#00a86b")
                    
                    st.markdown("### 🛡️ Análise Defensiva (Football-Data)")
                    c_cs1, c_cs2 = st.columns(2)
                    c_cs1.metric(f"Clean Sheets em Casa ({casa})", f"{stats[casa]['CS_Casa']:.1%}")
                    c_cs2.metric(f"Clean Sheets Fora ({fora})", f"{stats[fora]['CS_Fora']:.1%}")
                    
                    c_fts1, c_fts2 = st.columns(2)
                    c_fts1.metric(f"Failed to Score ({casa})", f"{stats[casa]['FTS_Casa']:.1%}")
                    c_fts2.metric(f"Failed to Score ({fora})", f"{stats[fora]['FTS_Fora']:.1%}")
                    
                    st.markdown("### 🧠 Rating Elo e Momento")
                    c_elo1, c_elo2 = st.columns(2)
                    c_elo1.metric(f"Elo ({casa})", f"{stats[casa]['Elo']:.0f}")
                    c_elo2.metric(f"Elo ({fora})", f"{stats[fora]['Elo']:.0f}")
                    
                    c_mom1, c_mom2 = st.columns(2)
                    c_mom1.metric(f"Fator Momento ({casa})", f"{stats[casa]['Fator_Momento']:.2f}x")
                    c_mom2.metric(f"Fator Momento ({fora})", f"{stats[fora]['Fator_Momento']:.2f}x")

                    st.markdown("### 🏆 Classificação")
                    if tabela and casa in tabela and fora in tabela:
                        class_df = pd.DataFrame({
                            'Equipa': [casa, fora],
                            'Posição': [f"{tabela[casa]['Posicao']}º", f"{tabela[fora]['Posicao']}º"],
                            'Pontos': [tabela[casa]['Pontos'], tabela[fora]['Pontos']]
                        })
                        st.dataframe(class_df, hide_index=True, use_container_width=True)
                    
                    st.markdown("### 🎯 Resultados Exatos")
                    matriz = gerar_matriz_resultados(res['xG_Casa'], res['xG_Fora'])
                    st.dataframe(matriz.style.background_gradient(cmap='Greens', axis=None).format("{:.2%}"), use_container_width=True)

                with tab3:
                    st.markdown("### 🤝 Histórico H2H (Últimos 5)")
                    h2h = df[((df['Casa'] == casa) & (df['Fora'] == fora)) | ((df['Casa'] == fora) & (df['Fora'] == casa))].tail(5)
                    if not h2h.empty:
                        h2h_display = h2h[['Data', 'Casa', 'Golos_Casa', 'Golos_Fora', 'Fora']].copy()
                        h2h_display['Resultado'] = h2h_display['Golos_Casa'].astype(str) + " - " + h2h_display['Golos_Fora'].astype(str)
                        st.dataframe(h2h_display[['Data', 'Casa', 'Resultado', 'Fora']], hide_index=True, use_container_width=True)
                    else:
                        st.info("Sem dados de H2H recentes.")

            else:
                st.error("Seleciona equipas diferentes.")
                
    else:
        st.error("⚠️ Não foi possível carregar os dados. Verifica a API Key ou espera 1 minuto (limite de pedidos).")
