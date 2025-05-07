Projeto que visa prever o risco de inadimplência de clientes utilizando dados financeiros e comportamentais. A ferramenta foi desenvolvida com interface interativa em Streamlit, permitindo ajuste de modelos e parâmetros em tempo real para um trabalho na Universidade católica de Brasilia.

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

## 💾 Instalação:

1. Clone o repositório
```bash 
https://github.com/FelipePortoDeveloper/PREVIC.git
cd PREVIC
```

2. Instale as dependencias
```bash
pip install -r requirements.txt
```

3. Inicie o programa
```bash
streamlit run interface.py
```

## 📬 Contato

Se tiver dúvidas ou sugestões:

Autor: Felipe Porto

E-mail: felipeportodeveloper5@gmail.com

GitHub: [FelipePortoDeveloper](https://github.com/FelipePortoDeveloper)
