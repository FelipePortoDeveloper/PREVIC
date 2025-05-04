import streamlit as st
from dados import TratamentoDados
from analise import AnaliseDados

st.set_page_config(layout="wide")
st.title("PREVIC")

Td = TratamentoDados("bruto.csv")
Td.carregarDados()
Td.limparDados()
dados_normalizados = Td.preprocessarDados()

analise = AnaliseDados(Td.dados)

st.subheader("Estatisticas Descritivas")
st.dataframe(analise.estatisticas_descritivas())

st.subheader("Histogramas")
for fig in analise.plotar_histogramas():
    st.pyplot(fig)

st.subheader("Boxplot")
coluna = st.selectbox("Escolha uma variável: ", analise.colunas_numericas())
st.pyplot(analise.plotar_boxplot(coluna))

st.subheader("Matriz de correlação")
st.pyplot(analise.matriz_correlacao())