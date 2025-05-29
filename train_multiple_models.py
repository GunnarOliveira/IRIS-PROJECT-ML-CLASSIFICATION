from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

# Carregar dados
iris = load_iris()
X, y = iris.data, iris.target

# Dividir em treino/teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Garantir pasta models
os.makedirs("models", exist_ok=True)

# Dicionário com modelos
models = {
    "knn": KNeighborsClassifier(n_neighbors=3),
    "logreg": LogisticRegression(max_iter=200),
    "tree": DecisionTreeClassifier(max_depth=2),  # Modelo mais simples (possivelmente subótimo)
}

# Treinar e salvar
for name, model in models.items():
    model.fit(X_train, y_train)
    joblib.dump(model, f"models/{name}_model.pkl")
    print(f"{name} - Acurácia no teste: {accuracy_score(y_test, model.predict(X_test)):.2%}")