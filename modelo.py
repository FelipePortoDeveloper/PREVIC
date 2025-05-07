import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd

class ConstrutorModelo:
    def __init__(self, dados: pd.DataFrame, colunas: list, alvo: str):
        self.dados = dados
        self.colunas = colunas
        self.alvo = alvo
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.modelo = None
        self.params = {
            'objective': 'binary:logistic',
            'max_depth': 4,
            'learning_rate': 0.1,
            'eval_metric': 'logloss'
        }
        self.num_rounds = 100
        self.threshold = 0.6

    def configurar_parametros(self, max_depth: int, learning_rate: float, num_rounds: int, threshold: float):
        self.params['max_depth'] = max_depth
        self.params['learning_rate'] = learning_rate
        self.num_rounds = num_rounds
        self.threshold = threshold

    def dividir_dados(self, test_size: float = 0.3):
        x = self.dados[self.colunas]
        y = self.dados[self.alvo]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            x, y, test_size=test_size, random_state=0
        )

        self.dtrain = xgb.DMatrix(self.X_train, label=self.y_train)
        self.dtest = xgb.DMatrix(self.X_test, label=self.y_test)

    def treinar_modelo(self):
        n_neg = (self.y_train == 0).sum()
        n_pos = (self.y_train == 1).sum()
        self.params['scale_pos_weight'] = n_neg / n_pos

        self.modelo = xgb.train(self.params, self.dtrain, num_boost_round=self.num_rounds)

    def avaliar_modelo(self, colunas_usuario: list, dados: list) -> str:
        dados_completos = {col: self.X_train[col].mean() for col in self.colunas}

        for col, val in zip(colunas_usuario, dados):
            dados_completos[col] = val

        df = pd.DataFrame([dados_completos])
        dinput = xgb.DMatrix(df[self.colunas])

        prob = self.modelo.predict(dinput)[0]
        print(f"Probabilidade de inadimplência: {prob:.2%}")

        return "Alto risco de inadimplência" if prob >= self.threshold else "Baixo risco de inadimplência"

    def avaliacao_geral(self):
        preds_proba = self.modelo.predict(self.dtest)
        preds = [1 if p >= self.threshold else 0 for p in preds_proba]

        acc = accuracy_score(self.y_test, preds)
        report = classification_report(self.y_test, preds, output_dict=True)
        matrix = confusion_matrix(self.y_test, preds)

        return {
            "acuracia": acc,
            "relatorio": report,
            "matriz_confusao": matrix
        }
