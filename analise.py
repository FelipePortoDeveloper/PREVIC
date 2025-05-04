import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from variaveis import colunas_utilizadas

class AnaliseDados:
    def __init__(self, dados: pd.DataFrame):
        self.dados = dados

    def estatisticas_descritivas(self):
        return self.dados.describe()
    
    def colunas_numericas(self):
        return self.dados.select_dtypes(include=["int64", "float64"]).columns.tolist()

    def plotar_histogramas(self):
        colunas = colunas_utilizadas
        figuras = []

        for col in colunas:
            fig, ax = plt.subplots()
            sns.histplot(self.dados[col], kde= True, ax= ax)
            figuras.append(fig)

        return figuras
    
    def plotar_boxplot(self, coluna:str):
        fig, ax = plt.subplots()
        sns.boxplot(x=self.dados[coluna], ax=ax)
        ax.set_title(f"Boxplot - {coluna}")
        return fig
    
    def matriz_correlacao(self):
        colunas = colunas_utilizadas
        fig, ax = plt.subplots()

        sns.heatmap(self.dados[colunas].corr(), annot=True, cmap="coolwarm", ax=ax)
        ax.set_title("Matriz de correlação")

        return fig