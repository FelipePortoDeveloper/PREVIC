from dados import TratamentoDados
from variaveis import colunas_utilizadas, variavel_alvo
from modelo import ConstrutorModelo

td = TratamentoDados("bruto.csv")
df = td.carregarDados()
df = td.limparDados()
df = td.preprocessarDados()

modelo = ConstrutorModelo(df, colunas_utilizadas, variavel_alvo)

modelo.dividir_dados()

modelo.treinar_modelo()

print(modelo.avaliar_modelo())
