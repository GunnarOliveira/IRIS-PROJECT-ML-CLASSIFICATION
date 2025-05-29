import joblib

def save_model(model, path='models/knn_model.pkl'):
    """
    Salva um modelo treinado em um arquivo.
    
    Args:
        model: Modelo treinado.
        path (str): Caminho onde o modelo será salvo.
    """
    # Salva o modelo no caminho especificado
    joblib.dump(model, path)
    print(f"Modelo salvo em {path}")