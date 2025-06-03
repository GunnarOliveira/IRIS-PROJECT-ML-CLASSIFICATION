# 🌸 Iris ML Classification Project 📊🤖

## Classificação de Espécies de Flores Iris com Machine Learning

Este é um projeto de análise de dados e classificação automática desenvolvido como parte da disciplina de **Análise de Big Data em Python** . O objetivo é demonstrar como diferentes modelos de Machine Learning podem ser aplicados para prever a espécie de uma flor Iris com base em medidas físicas (comprimento e largura de pétalas e sépalas).

O projeto inclui:

- Manipulação e visualização de dados
- Treinamento e avaliação de múltiplos modelos
- Interface interativa via [Streamlit](https://streamlit.io/)
- Comparação entre modelos adequados e inadequados ao problema

---

## 👥 Autores

Este projeto foi desenvolvido pelos alunos:

- **Matheus Henrique**
- **Luís Henrique**
- **John Kennedy**
- **Thiago Andrade**
- **Eu :P**  (Gunnar Vingren)

---

## 📂 Estrutura do Projeto

```bash
projeto-iris/

├── models/

   ├── knn_model.pkl # Modelo KNN treinado

   ├── logreg_model.pkl # Modelo Regressão Logística

   └── tree_model.pkl # Modelo Árvore de Decisão

├── src/

   ├── data.py # Funções para carregamento de dados

   ├── model.py # Treinamento e avaliação de modelos

   └── utils.py # Função auxiliar para salvar modelos

├── app_streamlit.py # Aplicação web interativa com Streamlit

├── train_multiple_models.py # Script para treinar e salvar múltiplos modelos

├── requirements.txt # Dependências necessárias

├── ROTEIRO_DE_EXTENSÃO.docx # Dependências necessárias

├── README.md # Este arquivo
```

---

## 🎯 Objetivo

Demonstrar de forma prática:

- Como preparar e explorar dados
- Como treinar e comparar modelos de classificação
- Como interpretar resultados e métricas
- Como criar uma interface interativa para visualizar previsões

Além disso, o projeto destaca a importância de escolher o modelo certo para o problema, exibindo também modelos propositalmente inadequados para mostrar impacto no desempenho.

---

## 📊 Funcionalidades

- ✅ Carrega e mostra o dataset Iris (local ou direto do scikit-learn)
- 📈 Visualiza gráficos de dispersão, distribuição e matriz de confusão
- 📋 Mostra relatórios completos de classificação
- 🔄 Permite ao usuário inserir novas medidas e obter previsão da espécie
- 🧠 Compara diferentes modelos de classificação (KNN, Regressão Logística, Árvore de Decisão)
- ⚠️ Destaca o comportamento de modelos subótimos

---

## 🛠️ Tecnologias Utilizadas

| Ferramenta / Biblioteca | Finalidade |
| --- | --- |
| **Python** | Linguagem principal |
| **scikit-learn** | Modelos de Machine Learning |
| **pandas / numpy** | Manipulação de dados |
| **matplotlib / seaborn** | Visualização de dados |
| **streamlit** | Interface web interativa |
| **joblib** | Salvamento e carregamento de modelos |

---

## 🔗 Dataset Utilizado

O projeto utiliza o famoso conjunto de dados Iris, disponível diretamente no `sklearn.datasets`. Além disso, foi adicionado ao repositório um **dataset local** para garantir maior flexibilidade e integridade dos dados mesmo sem internet.

🔗 [Baixe o dataset aqui (Google Drive)](https://drive.google.com/file/d/1vFtK6yeA_nTUZOD5-jIi1jETtMS7oX9b/view?usp=drive_link)

---

## 📥 Como Executar o Projeto

### 1. Clone o repositório:

```bash
git clone https://github.com/seu-usuario/projeto-iris-ml-classification.git

cd projeto-iris-ml-classification
```

### 2. Instale as dependências:

```bash
pip install -r requirements.txt
```

### 3. Treine os modelos:

```bash
python train_multiple_models.py
```

### 4. Inicie o app Streamlit:

```bash
streamlit run app_streamlit.py
```

---

## 🧪 Requisitos do Sistema

- Python 3.8 ou superior
- pip instalado
- Conexão com internet (para instalar bibliotecas, se necessário)

---

## 🧾 Requirements.txt

```textile
streamlit

scikit-learn

pandas

numpy

matplotlib

seaborn

joblib
```

---

## Link para a apresentação PPTX
https://prezi.com/view/E0ceonfKE83u6vEWRA10/ 

## 💬 Feedback & Contribuição

Ficou faltando algo? Tem uma ideia para melhorar o projeto? Sinta-se à vontade para abrir uma issue ou enviar um pull request!

E lembre-se: **Jesus te ama! ❤️**  
Sua jornada acadêmica e profissional tem valor — não importa quantos bugs apareçam pelo caminho 😄

---

## ✨ Créditos

- Dataset Iris: disponível no `sklearn.datasets` e também fornecido localmente via Google Drive
- Desenvolvido como parte do trabalho da disciplina de **Análise de Big Data em Python**
