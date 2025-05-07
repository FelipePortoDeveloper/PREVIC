from modelo import ConstrutorModelo
from dados import TratamentoDados
from variaveis import colunas_utilizadas, variavel_alvo
import pandas as pd

# 1. Carregar e preparar dados
dados = TratamentoDados("bruto.csv")
df = dados.carregarDados()
df = dados.limparDados()
df = dados.preprocessarDados()

# SALVAR uma cópia com ID antes de perder essa informação
df_com_id = pd.read_csv("bruto.csv")  # assumindo que o bruto.csv tem a coluna ID

# 2. Treinar o modelo
modelo = ConstrutorModelo(df, colunas_utilizadas, alvo=variavel_alvo)
modelo.dividir_dados()
modelo.treinar_modelo()

# 3. Testar ID 2 e ID 3
ids_para_testar = [63, 64]

for id_cliente in ids_para_testar:
    amostra = df_com_id[df_com_id['ID'] == id_cliente]

    if amostra.empty:
        print(f"ID {id_cliente} não encontrado.")
        continue

    entrada = amostra[colunas_utilizadas].iloc[0]
    colunas = list(entrada.index)
    valores = list(entrada.values)

    resultado = modelo.avaliar_modelo(colunas, valores)
    print(f"ID {id_cliente} {entrada[0]} => Previsão da IA: {resultado}")
