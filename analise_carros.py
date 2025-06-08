import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import json
import os
from datetime import datetime

# --- CONFIGURAÇÕES GERAIS E SUFIXOS PARA ARQUIVOS ---
nome_arquivo_csv = 'carros_status_venda_dataset.csv' 
dataset_tag = "fatores de compra" 
model_tag = "rf_clf_compra" 
sufixo_arquivos = f"_{model_tag}_{dataset_tag}" 

pasta_imagens = "img-geradas-fatores-compra" 
if not os.path.exists(pasta_imagens):
    os.makedirs(pasta_imagens)
    print(f"Pasta '{pasta_imagens}' criada com sucesso.")

# --- CARREGAMENTO DO DATASET ---
try:
    df = pd.read_csv(nome_arquivo_csv, encoding='utf-8')
    print(f"Dataset '{nome_arquivo_csv}' carregado com sucesso. Shape: {df.shape}")
except FileNotFoundError:
    print(f"ERRO: Arquivo '{nome_arquivo_csv}' não encontrado.")
    print("Execute o script gerador de dataset ('gerador_dataset_status_venda.py') primeiro.")
    exit() 
except Exception as e:
    print(f"Ocorreu um erro ao carregar o CSV: {e}")
    exit()

# --- ANÁLISE EXPLORATÓRIA DE DADOS (FOCO NA VARIÁVEL ALVO 'Foi_Vendido') ---
print("\n--- Análise Exploratória de Dados (EDA) ---")
if 'Foi_Vendido' not in df.columns:
    print("ERRO: Coluna 'Foi_Vendido' crucial não encontrada no dataset.")
    exit()

print("Distribuição da variável alvo 'Foi_Vendido':")
print(df['Foi_Vendido'].value_counts(normalize=True)) 

plt.figure(figsize=(6, 4))
sns.countplot(x='Foi_Vendido', data=df)
plt.title(f'Distribuição da Target "Foi Vendido" (0=Não, 1=Sim) (Dataset {dataset_tag})')
plt.xticks([0, 1], ['Não Vendido (0)', 'Vendido (1)']) 
plt.savefig(f'{pasta_imagens}/distribuicao_foi_vendido_target{sufixo_arquivos}.png', bbox_inches='tight')
plt.show()

# --- PREPARAÇÃO DOS DADOS ---
print("\n--- Preparação dos Dados para Classificação de Venda ---")
current_year = datetime.now().year
if 'Ano_Modelo' in df.columns:
    df['Idade_Carro_Modelo'] = current_year - df['Ano_Modelo']
    print("'Idade_Carro_Modelo' criada.")

categorical_cols = ['Marca', 'Modelo', 'Combustivel', 'Cambio', 'Cor', 'Estado_Venda'] 
numerical_cols = ['Quilometragem', 'Num_Portas', 'Preco_Listagem'] 
if 'Idade_Carro_Modelo' in df.columns:
    numerical_cols.append('Idade_Carro_Modelo')

features_col_base = categorical_cols + numerical_cols
features_col = [col for col in features_col_base if col in df.columns]
target_col = 'Foi_Vendido' 

if target_col not in df.columns:
    print(f"ERRO: Coluna alvo '{target_col}' não encontrada no CSV.")
    exit()
missing_feature_cols = [col for col in features_col_base if col not in df.columns]
if missing_feature_cols:
    print(f"AVISO: Colunas de features não encontradas e serão ignoradas: {missing_feature_cols}")

X = df[features_col].copy()
y = df[target_col].copy() 

print(f"\nFeatures selecionadas para o modelo: {X.columns.tolist()}")

existing_categorical_in_X = [col for col in categorical_cols if col in X.columns]
if existing_categorical_in_X:
    X = pd.get_dummies(X, columns=existing_categorical_in_X, drop_first=True) 
    print(f"Shape de X após One-Hot Encoding: {X.shape}")

existing_numerical_in_X = [col for col in numerical_cols if col in X.columns]
scaler = StandardScaler()
if existing_numerical_in_X:
    X[existing_numerical_in_X] = scaler.fit_transform(X[existing_numerical_in_X])
    print(f"\nFeatures {existing_numerical_in_X} escalonadas.")

print("\nFeatures (X) APÓS todo o pré-processamento (primeiras 5 linhas):")
print(X.head())
print(f"Shape final de X para modelagem: {X.shape}")

colunas_modelo_treino_final = list(X.columns) 

# --- DIVISÃO EM TREINO E TESTE ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y) 
print("\n--- Divisão dos Dados ---")
print(f"Shape de X_train: {X_train.shape}, Shape de X_test: {X_test.shape}")

# --- TREINAMENTO DO MODELO RANDOM FOREST CLASSIFIER ---
print(f"\n--- Treinamento do Modelo (Fatores de Compra - Dataset {dataset_tag}) ---")
model = RandomForestClassifier(
    n_estimators=100, max_depth=None, min_samples_split=10,  
    min_samples_leaf=5, random_state=42, n_jobs=-1             
)
model.fit(X_train, y_train)
print("Modelo RandomForestClassifier treinado com sucesso!")

# --- PREVISÕES E AVALIAÇÃO --- 
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n--- Avaliação do Modelo RandomForest {dataset_tag}) ---")
print(f"Acurácia do modelo no conjunto de teste: {accuracy:.4f}")

print("\nMatriz de Confusão:")
cm = confusion_matrix(y_test, y_pred) 
cm_df = pd.DataFrame(cm, index=['Não Vendido (0)', 'Vendido (1)'], columns=['Prev. Não Vendido (0)', 'Prev. Vendido (1)'])
print(cm_df)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Greens')
plt.title(f'Matriz de Confusão (Fatores de Compra - RF {dataset_tag})')
plt.ylabel('Verdadeiro')
plt.xlabel('Previsto')
plt.savefig(f'{pasta_imagens}/matriz_confusao_fatores_compra{sufixo_arquivos}.png', bbox_inches='tight')
plt.show()

print(f"\nRelatório de Classificação (Fatores de Compra - RF {dataset_tag}):")
print(classification_report(y_test, y_pred, target_names=['Não Vendido (0)', 'Vendido (1)'], zero_division=0))

# --- IMPORTÂNCIA DAS FEATURES (PARA IDENTIFICAR FATORES DE COMPRA) ---
print(f"\n--- Importância das Features (para Fatores de Compra - {dataset_tag}) ---")
if hasattr(model, 'feature_importances_'):
    importances = model.feature_importances_
    feature_importances_df = pd.DataFrame({'feature': X_train.columns, 'importance': importances})
    feature_importances_df = feature_importances_df.sort_values('importance', ascending=False)
    print("\nImportância de cada feature (Top 15):")
    print(feature_importances_df.head(15))
    plt.figure(figsize=(10, 8)) 
    sns.barplot(x='importance', y='feature', data=feature_importances_df.head(15), palette="crest_r") 
    plt.title(f'Top 15 Features Mais Decisivas para a Compra (RF - {dataset_tag})')
    plt.tight_layout()
    # <<< AJUSTE AQUI para o nome do arquivo que o dashboard espera >>>
    plt.savefig(f'{pasta_imagens}/importancia_features{sufixo_arquivos}.png', bbox_inches='tight') 
    plt.show()
    print("\nFeatures no topo desta lista são os fatores mais decisivos que o modelo usou para distinguir carros vendidos de não vendidos.")

# --- SALVANDO ARTEFATOS DO MODELO ---
print(f"\n--- Salvando Artefatos do Modelo (Fatores de Compra - {dataset_tag}) ---")
metricas_para_salvar = {
    "algoritmo": "RandomForest (Fatores de Compra)", 
    "acuracia_teste": float(accuracy)
}
try:
    with open(f'metricas_modelo{sufixo_arquivos}.json', 'w') as f: 
        json.dump(metricas_para_salvar, f, indent=4)
    print(f"Métricas salvas em 'metricas_modelo{sufixo_arquivos}.json'!")
    
    joblib.dump(model, f'modelo_final{sufixo_arquivos}.joblib') 
    joblib.dump(scaler, f'scaler_features_numericas{sufixo_arquivos}.joblib') 
    joblib.dump(colunas_modelo_treino_final, f'colunas_modelo_treino{sufixo_arquivos}.joblib') 
    joblib.dump(existing_numerical_in_X, f'colunas_numericas_escalonadas{sufixo_arquivos}.joblib') 
    print("Modelo e outros artefatos salvos!")
except Exception as e:
    print(f"Erro ao salvar artefatos: {e}")

print(f"\nScript {os.path.basename(__file__)} finalizado!")