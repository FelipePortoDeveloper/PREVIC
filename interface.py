# interface.py

import streamlit as st
import pandas as pd
from modelo import ConstrutorModelo
from dados import TratamentoDados
from variaveis import colunas_utilizadas, variavel_alvo
from analise import AnaliseDados
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# Configuração da página
st.set_page_config(page_title="Previsão de Inadimplência", layout="wide")

# Navegação lateral com menu simplificado
menu = st.sidebar.selectbox("Navegação", ["Análise Exploratória (EDA)", "Avaliação do Modelo", "Sobre o Projeto"])

# Carregar e preparar dados
dados = TratamentoDados("bruto.csv")
df_original = dados.carregarDados()
df_limpo = dados.limparDados()
df = dados.preprocessarDados()

# Instanciar e treinar modelo
modelo = ConstrutorModelo(df, colunas_utilizadas, alvo=variavel_alvo)

if menu == "Análise Exploratória (EDA)":
    st.title("Análise Exploratória de Dados")
    analise = AnaliseDados(df_original)

    st.subheader("Estatísticas")
    st.dataframe(analise.estatisticas_descritivas())

    st.subheader("Histogramas")
    for coluna in colunas_utilizadas:
        fig = px.histogram(df_original, x=coluna, nbins=50, title=f"Distribuição de {coluna}")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Boxplot")
    coluna = st.selectbox("Escolha uma variável:", colunas_utilizadas)
    fig_box = px.box(df_original, y=coluna, title=f"Boxplot de {coluna}")
    st.plotly_chart(fig_box, use_container_width=True)

    st.subheader("Matriz de Correlação")
    corr = df_original[colunas_utilizadas].corr()
    fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu', title="Matriz de Correlação")
    st.plotly_chart(fig_corr, use_container_width=True)

elif menu == "Avaliação do Modelo":
    st.sidebar.header("Parâmetros do Modelo")
    max_depth = st.sidebar.slider("Profundidade da Árvore", 1, 10, 4)
    learning_rate = st.sidebar.slider("Taxa de Aprendizado", 0.01, 0.5, 0.1, step=0.01)
    num_rounds = st.sidebar.slider("Nº de Iterações", 10, 500, 100, step=10)
    threshold = st.sidebar.slider("Limiar para Classificação", 0.0, 1.0, 0.6, step=0.01)

    modelo.configurar_parametros(max_depth, learning_rate, num_rounds, threshold)
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
    Este projeto foi desenvolvido para analisar dados financeiros de clientes e prever a probabilidade de inadimplência
    utilizando o algoritmo **XGBoost**, um dos mais poderosos métodos de classificação.

    ### Componentes do Sistema
    - **Análise Exploratória (EDA)**: Geração de gráficos estatísticos e descritivos sobre os dados.
    - **Avaliação de Modelo**: Treinamento e ajuste interativo do modelo com avaliação automática em dados de teste.
    - **Visualização de Métricas**: Apresentação de acurácia, relatório de classificação e matriz de confusão.

    ### Metodologia
    Os dados são normalizados e tratados antes de alimentar o modelo. O classificador XGBoost é treinado com ajuste para
    desbalanceamento (`scale_pos_weight`) e permite personalização dos parâmetros diretamente na aplicação.

    O objetivo é fornecer uma ferramenta interativa e visual para apoiar decisões de crédito de forma objetiva.
    """)