import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class TratamentoDados:
    def __init__(self, caminho:str):
        self.caminho = caminho
        self.dados = None
        self.scaler = MinMaxScaler()

    def carregarDados(self) -> pd.DataFrame:
        self.dados = pd.read_csv(self.caminho)
        return self.dados

    def remover_outliers(self, colunas):
        for col in colunas:
            Q1 = self.dados[col].quantile(0.25)
            Q3 = self.dados[col].quantile(0.75)
            IQR = Q3 - Q1
            limite_inferior = Q1 - 1.5 * IQR
            limite_superior = Q3 + 1.5 * IQR
            self.dados = self.dados[(self.dados[col] >= limite_inferior) & (self.dados[col] <= limite_superior)]
        return self.dados

    def codificar_variaveis(self):
        self.dados = pd.get_dummies(self.dados, columns=["sexo", "educação", "estado civil"], drop_first=True)
        return self.dados
    
    def limparDados(self) -> pd.DataFrame:
        if self.dados is None:
            raise ValueError("Nenhum dado carregado. Execute carregarDados() primeiro.")
        self.dados = self.dados.dropna()
        colunas_numericas = self.dados.select_dtypes(include=["float64", "int64"]).columns
        self.remover_outliers(colunas_numericas)
        self.codificar_variaveis()
        return self.dados
    
    def preprocessarDados(self) -> pd.DataFrame:
        if self.dados is None:
            raise ValueError("Nenhum dado carregado. Execute carregarDados() primeiro.")

        colunasNumericas = self.dados.select_dtypes(include=['float64', 'int64']).columns
        self.dados[colunasNumericas] = self.scaler.fit_transform(self.dados[colunasNumericas])
        return self.dados
