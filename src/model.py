from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def train_knn(X_train, y_train, n_neighbors=3):
    """
    Treina um modelo K-Nearest Neighbors (KNN).
    
    Args:
        X_train: Features de treino.
        y_train: Labels de treino.
        n_neighbors (int): Número de vizinhos no KNN.
    
    Returns:
        model: Modelo treinado.
    """
    # Criar e treinar o modelo
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Avalia o desempenho de um modelo de classificação.
    
    Args:
        model: Modelo treinado.
        X_test: Features de teste.
        y_test: Labels de teste.
    """
    # Fazer previsões
    y_pred = model.predict(X_test)

    # Avaliar o modelo
    print(f"Acurácia: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print("\nRelatório de Classificação:")
    print(classification_report(y_test, y_pred))
    print("Matriz de Confusão:")
    print(confusion_matrix(y_test, y_pred))