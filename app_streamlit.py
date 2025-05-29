import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Carregar o modelo treinado
model = joblib.load('models/knn_model.pkl')

# Carregar o dataset Iris
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# seleção do modelo
st.sidebar.header("Selecione o Modelo")
model_choice = st.sidebar.selectbox(
    "Escolha um modelo",
    ("KNN", "Regressão Logística", "Árvore de Decisão")
)

# Mapear escolha para caminho do modelo
model_map = {
    "KNN": "models/knn_model.pkl",
    "Regressão Logística": "models/logreg_model.pkl",
    "Árvore de Decisão": "models/tree_model.pkl"
}

# Carregar modelo escolhido
@st.cache_resource
def load_selected_model(path):
    return joblib.load(path)

model = load_selected_model(model_map[model_choice])

st.info(f"Modelo selecionado: {model_choice}")

if model_choice == "Árvore de Decisão":
    st.warning("⚠️ Este modelo tem menor capacidade de generalização (underfitting).")

# Título da aplicação
st.title("Classificador de Flores Iris com Machine Learning")

# Seção 1: Visualização do Dataset
st.header("Dataset Iris")
df = pd.DataFrame(X, columns=feature_names)
df['species'] = y
df['species'] = df['species'].apply(lambda x: target_names[x])
st.write(df)

# Seção 2: Avaliação do Modelo
st.header("Avaliação do Modelo")
y_pred = model.predict(X)
accuracy = np.mean(y_pred == y) * 100
st.write(f"Acurácia do Modelo: {accuracy:.2f}%")

st.subheader("Relatório de Classificação")
report = classification_report(y, y_pred, target_names=target_names, output_dict=True)
report_df = pd.DataFrame(report).transpose()
st.write(report_df)

st.subheader("Matriz de Confusão")
cm = confusion_matrix(y, y_pred)
cm_df = pd.DataFrame(cm, index=target_names, columns=target_names)
st.write(cm_df)

# Seção 3: Fazer Previsões
st.header("Faça uma Previsão")
st.write("Insira as medidas da flor para prever sua espécie:")

# Entradas do usuário
sepal_length = st.number_input("Comprimento da Sépala (cm)", min_value=0.0, max_value=10.0, value=5.1)
sepal_width = st.number_input("Largura da Sépala (cm)", min_value=0.0, max_value=10.0, value=3.5)
petal_length = st.number_input("Comprimento da Pétala (cm)", min_value=0.0, max_value=10.0, value=1.4)
petal_width = st.number_input("Largura da Pétala (cm)", min_value=0.0, max_value=10.0, value=0.2)

# Botão para fazer a previsão
if st.button("Prever"):
    # Criar um array com as entradas do usuário
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    
    # Fazer a previsão
    prediction = model.predict(input_data)[0]
    predicted_species = target_names[prediction]
    
    # Mostrar o resultado
    st.success(f"A espécie prevista é: **{predicted_species}**")

    # Gráfico de dispersão com a previsão
    st.subheader("Gráfico de Dispersão com a Previsão")
    plt.figure(figsize=(8, 6))
    
    # Plotar os dados do dataset Iris
    sns.scatterplot(data=df, x="sepal length (cm)", y="sepal width (cm)", hue="species", palette="Set2", alpha=0.7)
    
    # Plotar o ponto da previsão do usuário
    plt.scatter(sepal_length, sepal_width, color="red", s=100, label="Sua Previsão", edgecolor="black", zorder=3)
    
    # Adicionar título e rótulos
    plt.title("Gráfico de Dispersão com a Previsão")
    plt.xlabel("Comprimento da Sépala (cm)")
    plt.ylabel("Largura da Sépala (cm)")
    plt.legend()
    st.pyplot(plt)

# Seção 4: Gráficos de Análise de Dados
st.header("Análise Visual dos Dados")

# Gráfico de dispersão
st.subheader("Gráfico de Dispersão")
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x="sepal length (cm)", y="sepal width (cm)", hue="species", palette="Set2")
plt.title("Relação entre Comprimento e Largura da Sépala")
plt.xlabel("Comprimento da Sépala (cm)")
plt.ylabel("Largura da Sépala (cm)")
st.pyplot(plt)

# Heatmap da matriz de confusão
st.subheader("Heatmap da Matriz de Confusão")
plt.figure(figsize=(8, 6))
sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("Matriz de Confusão")
plt.xlabel("Classe Prevista")
plt.ylabel("Classe Real")
st.pyplot(plt)

# Pairplot
st.subheader("Pairplot das Características")
pairplot_fig = sns.pairplot(df, hue="species", palette="Set2", diag_kind="kde")
st.pyplot(pairplot_fig)

# Distribuição das características
st.subheader("Distribuição das Características")
for feature in feature_names:
    plt.figure(figsize=(8, 4))
    sns.histplot(data=df, x=feature, hue="species", kde=True, palette="Set2")
    plt.title(f"Distribuição de {feature}")
    plt.xlabel(feature)
    plt.ylabel("Frequência")
    st.pyplot(plt)