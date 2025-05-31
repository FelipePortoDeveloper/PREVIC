import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd

class ConstrutorModelo:
    def __init__(self, dados: pd.DataFrame, colunas: list, alvo: str, tipo_modelo: str = "XGBoost"):
        self.dados = dados
        self.colunas = colunas
        self.alvo = alvo
        self.tipo_modelo = tipo_modelo
        self.modelo = None
        self.threshold = 0.5

    def configurar_parametros(self, max_depth, learning_rate, num_rounds, threshold):
        self.params_xgb = {
            'objective': 'binary:logistic',
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'eval_metric': 'logloss'
        }
        self.num_rounds = num_rounds
        self.threshold = threshold

    def configurar_parametros_rf(self, n_estimators, max_depth, threshold):
        self.params_rf = {
            'n_estimators': n_estimators,
            'max_depth': max_depth
        }
        self.threshold = threshold

    def dividir_dados(self, test_size: float = 0.3):
        x = self.dados[self.colunas]
        y = self.dados[self.alvo]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            x, y, test_size=test_size, random_state=0
        )

    def treinar_modelo(self):
        if self.tipo_modelo == "XGBoost":
            n_neg = (self.y_train == 0).sum()
            n_pos = (self.y_train == 1).sum()
            self.params_xgb['scale_pos_weight'] = n_neg / n_pos
            dtrain = xgb.DMatrix(self.X_train, label=self.y_train)
            self.modelo = xgb.train(self.params_xgb, dtrain, num_boost_round=self.num_rounds)
        elif self.tipo_modelo == "Random Forest":
            self.modelo = RandomForestClassifier(**self.params_rf)
            self.modelo.fit(self.X_train, self.y_train)

    def avaliacao_geral(self):
        if self.tipo_modelo == "XGBoost":
            dtest = xgb.DMatrix(self.X_test)
            prob = self.modelo.predict(dtest)
        else:
            prob = self.modelo.predict_proba(self.X_test)[:, 1]

        preds = [1 if p >= self.threshold else 0 for p in prob]

        acc = accuracy_score(self.y_test, preds)
        report = classification_report(self.y_test, preds, output_dict=True)
        matrix = confusion_matrix(self.y_test, preds)

        return {
            "acuracia": acc,
            "relatorio": report,
            "matriz_confusao": matrix
        }
