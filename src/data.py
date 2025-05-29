import numpy as np
import pandas as pd

def augment_and_save_data(output_path="data/large_iris_dataset.parquet", noise_level=0.05, multiplier=100000):

    from sklearn.datasets import load_iris
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names

    augmented_X = []
    augmented_y = []
    for _ in range(multiplier):
        noise = np.random.normal(0, noise_level, X.shape)
        augmented_X.append(X + noise)
        augmented_y.extend(y)
    
    augmented_X = np.vstack(augmented_X)
    augmented_y = np.array(augmented_y)

    # Criar DataFrame
    df_augmented = pd.DataFrame(augmented_X, columns=feature_names)
    df_augmented['species'] = augmented_y
    df_augmented['species'] = df_augmented['species'].apply(lambda x: target_names[x])

   
    df_augmented.to_parquet(output_path, index=False)
    print(f"Dataset carregado salvo em {output_path} com {len(df_augmented)} amostras.")

def load_large_dataset(path="data/large_iris_dataset.parquet"):

    df = pd.read_parquet(path)
    X = df.drop(columns=["species"]).values
    y = df["species"].map({"setosa": 0, "versicolor": 1, "virginica": 2}).values
    return X, y