import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
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
        self.resultados = None
        self.grid_search = None
    
    def dividir_dados(self, test_size: float = 0.3):
        x = self.dados[self.colunas]
        y = self.dados[self.alvo]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(x, y, test_size=test_size, random_state=0)

        self.dtrain = xgb.DMatrix(self.X_train, label=self.y_train)
        self.dtest = xgb.DMatrix(self.X_test, label=self.y_test)

    def treinar_modelo(self):
        params = {'objective': 'binary:logistic', 'max_depth': 3, 'learning_rate': 0.1, 'eval_metric': 'logloss'}
        num_rounds = 50

        self.modelo = xgb.train(params, self.dtrain, num_rounds)


    def avaliar_modelo(self) -> str:
        preds = self.modelo.predict(self.dtest)
        predictions = [round(value) for value in preds]

        accuracy = accuracy_score(self.y_test, predictions)

        return str(f"{accuracy * 100:.2f}%")
