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
    
    def limparDados(self) -> pd.DataFrame:
        if self.dados is None:
            raise ValueError("Nenhum dado carregado. Execute load_data() primeiro.")
        
        self.dados = self.dados.dropna()
        return self.dados
    
    def preprocessarDados(self) -> pd.DataFrame:
        if self.dados is None:
            raise ValueError("Nenhum dado carregado. Execute load_data() primeiro.")

        dados_copy = self.dados.copy()
        colunasNumericas = dados_copy.select_dtypes(include=['float64', 'int64']).columns
        dados_copy[colunasNumericas] = self.scaler.fit_transform(self.dados[colunasNumericas])

        return dados_copy