# interface.py

import streamlit as st
from modelo import ConstrutorModelo
from dados import TratamentoDados
from variaveis import colunas_utilizadas, variavel_alvo
from analise import AnaliseDados
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

st.set_page_config(page_title="Previsão de Inadimplência", layout="wide")

menu = st.sidebar.selectbox("Navegação", ["Análise Exploratória (EDA)", "Avaliação do Modelo", "Sobre o Projeto"])

dados = TratamentoDados("bruto.csv")
df_original = dados.carregarDados()
df_limpo = dados.limparDados()
df = dados.preprocessarDados()

map_sex = {1: "Homem", 2: "Mulher"}
map_education = {
    1: "Pós-graduação",
    2: "Universidade",
    3: "Ensino Médio",
    4: "Outros",
    5: "Desconhecido",
    6: "Desconhecido"
}
map_marriage = {1: "Casado(a)", 2: "Solteiro(a)", 3: "Outro"}

df_original_plot = df_original.copy()
df_original_plot['sexo'] = df_original_plot['sexo'].map(map_sex)
df_original_plot['educação'] = df_original_plot['educação'].map(map_education)
df_original_plot['estado civil'] = df_original_plot['estado civil'].map(map_marriage)

modelo = ConstrutorModelo(df, colunas_utilizadas, alvo=variavel_alvo)

if menu == "Análise Exploratória (EDA)":
    st.title("Análise Exploratória de Dados")
    analise = AnaliseDados(df_original)

    st.subheader("Estatísticas Descritivas")
    st.dataframe(analise.estatisticas_descritivas())

    st.subheader("Histogramas")
    for coluna in colunas_utilizadas:
        fig = px.histogram(df_original_plot, x=coluna, nbins=50, title=f"Distribuição de {coluna}")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Boxplot Interativo")
    coluna = st.selectbox("Escolha uma variável:", colunas_utilizadas)
    fig_box = px.box(df_original_plot, y=coluna, title=f"Boxplot de {coluna}")
    st.plotly_chart(fig_box, use_container_width=True)

    st.subheader("Matriz de Correlação (Interativa)")
    corr = df_original[colunas_utilizadas].corr()
    fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu', title="Matriz de Correlação")
    st.plotly_chart(fig_corr, use_container_width=True)

elif menu == "Avaliação do Modelo":
    st.sidebar.header("Selecione o Algoritmo")
    tipo_modelo = st.sidebar.selectbox("Modelo", ["XGBoost", "Random Forest"])
    st.sidebar.header("Parâmetros do Modelo")

    if tipo_modelo == "XGBoost":
        max_depth = st.sidebar.slider("Profundidade da Árvore", 1, 10, 4)
        learning_rate = st.sidebar.slider("Taxa de Aprendizado", 0.01, 0.5, 0.1, step=0.01)
        num_rounds = st.sidebar.slider("Nº de Iterações", 10, 500, 100, step=10)
        threshold = st.sidebar.slider("Limiar para Classificação", 0.0, 1.0, 0.6, step=0.01)
    else:
        n_estimators = st.sidebar.slider("Nº de Árvores", 10, 500, 100, step=10)
        max_depth = st.sidebar.slider("Profundidade Máxima", 1, 20, 10)
        threshold = st.sidebar.slider("Limiar para Classificação", 0.0, 1.0, 0.6, step=0.01)

    modelo = ConstrutorModelo(df, colunas_utilizadas, alvo=variavel_alvo, tipo_modelo=tipo_modelo)

    if tipo_modelo == "XGBoost":
        modelo.configurar_parametros(max_depth, learning_rate, num_rounds, threshold)
    else:
        modelo.configurar_parametros_rf(n_estimators, max_depth, threshold)

    modelo.dividir_dados()
    modelo.treinar_modelo()

    st.title("Avaliação do Modelo de Inadimplência")
    st.subheader("Desempenho com Dados de Teste")
    metricas = modelo.avaliacao_geral()

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Acurácia", f"{metricas['acuracia'] * 100:.2f}%")
        st.markdown("**Relatório de Classificação:**")
        st.json(metricas['relatorio'])

    with col2:
        st.markdown("**Matriz de Confusão:**")
        fig, ax = plt.subplots()
        sns.heatmap(metricas['matriz_confusao'], annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
        st.pyplot(fig)


elif menu == "Sobre o Projeto":
    st.title("Sobre o Projeto")
    st.markdown("""
    ## 🎯 Objetivo
    Este projeto visa prever o risco de inadimplência de clientes utilizando dados financeiros e comportamentais. A ferramenta foi desenvolvida com interface interativa em Streamlit, permitindo ajuste de modelos e parâmetros em tempo real.

    ## 🔍 Análise Exploratória (EDA)
    - A maioria dos clientes possui limite de crédito entre R$ 50.000 e R$ 200.000.
    - Inadimplência é mais comum entre clientes com menos de 30 anos.
    - Solteiros(as) representam maior proporção de inadimplentes.
    - Clientes com menor escolaridade têm maiores taxas de inadimplência.
    - Correlação significativa entre valores de faturas em meses consecutivos.

    ## 🤖 Modelos Utilizados

    ### XGBoost (Padrão)
    - Acurácia: **80.86%**
    - Recall (inadimplentes): **50.97%**
    - Precision (inadimplentes): **56.16%**
    - F1-score (inadimplentes): **53.44%**

    ### Random Forest (Padrão)
    - Acurácia: **81.98%**
    - Recall (inadimplentes): **25.46%**
    - Precision (inadimplentes): **73.73%**
    - F1-score (inadimplentes): **37.85%**

    ## 📈 Conclusões e Recomendações
    - O XGBoost apresenta melhor recall, importante para identificar inadimplentes.
    - Random Forest tem acurácia ligeiramente maior, mas menor sensibilidade para classe positiva.
    - A escolha do modelo depende do perfil de risco e da tolerância a falsos negativos.
    - Recomendamos uso com limiar ajustável para adaptar a estratégia da política de crédito.
    """)

    with open("PREVIC.pdf", "rb") as file:
        st.download_button("Documentação PREVIC", data=file, file_name="PREVIC.pdf")
