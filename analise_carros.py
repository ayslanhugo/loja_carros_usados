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
# 1. Definições Principais
nome_arquivo_csv = 'carros_status_venda_dataset.csv' 
dataset_tag = "10k"  # Identifica o tamanho do dataset
model_tag = "rf_fatores_compra"  # Identifica o modelo (Random Forest para Fatores de Compra)
sufixo_arquivos = f"_{model_tag}_{dataset_tag}" # Resultado: "_rf_fatores_compra_10k"

# 2. Pasta de Saída para Gráficos
pasta_imagens = "img-geradas-fatores-compra"
if not os.path.exists(pasta_imagens):
    os.makedirs(pasta_imagens)
    print(f"Pasta '{pasta_imagens}' criada com sucesso.")

# 3. CARREGAMENTO DO DATASET
try:
    df = pd.read_csv(nome_arquivo_csv, encoding='utf-8')
    print(f"Dataset '{nome_arquivo_csv}' carregado com sucesso. Shape: {df.shape}")
except FileNotFoundError:
    print(f"ERRO: Arquivo '{nome_arquivo_csv}' não encontrado.")
    print("Execute o script 'gerador_dataset_status_venda.py' primeiro.")
    exit()
except Exception as e:
    print(f"Ocorreu um erro ao carregar o CSV: {e}")
    exit()

# 4. ENGENHARIA DE FEATURES (se necessário)
print("\n--- Engenharia de Features ---")
if 'Ano_Modelo' in df.columns:
    df['Idade_Carro_Modelo'] = datetime.now().year - df['Ano_Modelo']
    print("'Idade_Carro_Modelo' criada.")

# 5. ANÁLISE EXPLORATÓRIA DE DADOS (EDA)
print("\n--- Análise Exploratória de Dados (EDA) ---")
if 'Foi_Vendido' not in df.columns:
    print("ERRO: Coluna alvo 'Foi_Vendido' não encontrada no dataset.")
    exit()

print("Distribuição da variável alvo 'Foi_Vendido':")
print(df['Foi_Vendido'].value_counts(normalize=True)) 

plt.figure(figsize=(6, 4))
sns.countplot(x='Foi_Vendido', data=df, palette="viridis")
plt.title(f'Distribuição da Target "Foi Vendido" (0=Não, 1=Sim)')
plt.xticks([0, 1], ['Não Vendido (0)', 'Vendido (1)']) 
plt.savefig(f'{pasta_imagens}/dist_foi_vendido_target{sufixo_arquivos}.png', bbox_inches='tight')
plt.show()

# 6. GERANDO E SALVANDO RANKING DE MODELOS MAIS VENDIDOS
print("\n--- Calculando e Salvando Modelos Mais Vendidos ---")
if 'Modelo' in df.columns and 'Marca' in df.columns:
    df_vendidos = df[df['Foi_Vendido'] == 1]
    top_geral = df_vendidos['Modelo'].value_counts().nlargest(10).to_dict()
    top_por_marca = {marca: group['Modelo'].value_counts().nlargest(5).to_dict() 
                     for marca, group in df_vendidos.groupby('Marca')}
    
    ranking_modelos = {"top_geral": top_geral, "top_por_marca": top_por_marca}
    with open('ranking_modelos_vendidos.json', 'w', encoding='utf-8') as f:
        json.dump(ranking_modelos, f, indent=4, ensure_ascii=False)
    print("Ranking de modelos mais vendidos salvo em 'ranking_modelos_vendidos.json'!")


# 7. PREPARAÇÃO DOS DADOS PARA O MODELO
print("\n--- Preparação dos Dados para Classificação ---")
categorical_cols = ['Marca', 'Modelo', 'Combustivel', 'Cambio', 'Cor', 'Estado_Venda'] 
numerical_cols = ['Quilometragem', 'Num_Portas', 'Preco_Listagem', 'Idade_Carro_Modelo']
features_col = [col for col in categorical_cols + numerical_cols if col in df.columns]
target_col = 'Foi_Vendido'

X = df[features_col].copy()
y = df[target_col].copy() 

X = pd.get_dummies(X, columns=[col for col in categorical_cols if col in X.columns], drop_first=True)

existing_numerical_in_X = [col for col in numerical_cols if col in X.columns]
scaler = StandardScaler()
if existing_numerical_in_X:
    X[existing_numerical_in_X] = scaler.fit_transform(X[existing_numerical_in_X])
    print(f"Features {existing_numerical_in_X} escalonadas.")

colunas_modelo_treino_final = list(X.columns)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
print(f"Dados divididos em treino ({X_train.shape[0]} amostras) e teste ({X_test.shape[0]} amostras).")

# 8. TREINAMENTO DO MODELO RANDOM FOREST CLASSIFIER
print(f"\n--- Treinamento do Modelo RandomForestClassifier ---")
model = RandomForestClassifier(
    n_estimators=100, max_depth=None, min_samples_split=10,  
    min_samples_leaf=5, random_state=42, n_jobs=-1             
)
model.fit(X_train, y_train)
print("Modelo treinado com sucesso!")

# 9. AVALIAÇÃO DO MODELO
print(f"\n--- Avaliação do Modelo ---")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Acurácia no conjunto de teste: {accuracy:.4f}")

print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred, target_names=['Não Vendido (0)', 'Vendido (1)']))

print("\nMatriz de Confusão:")
cm = confusion_matrix(y_test, y_pred) 
cm_df = pd.DataFrame(cm, index=['Real Não Vendido', 'Real Vendido'], columns=['Prev. Não Vendido', 'Prev. Vendido'])
print(cm_df)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Greens')
plt.title(f'Matriz de Confusão (RF - {dataset_tag})')
plt.ylabel('Verdadeiro')
plt.xlabel('Previsto')
plt.savefig(f'{pasta_imagens}/matriz_confusao{sufixo_arquivos}.png', bbox_inches='tight')
plt.show()

# 10. IMPORTÂNCIA DAS FEATURES
print(f"\n--- Importância das Features ---")
importances = model.feature_importances_
feature_importances_df = pd.DataFrame({'feature': X_train.columns, 'importance': importances})
feature_importances_df = feature_importances_df.sort_values('importance', ascending=False)
print("Top 15 Features Mais Importantes:")
print(feature_importances_df.head(15))

plt.figure(figsize=(10, 8)) 
sns.barplot(x='importance', y='feature', data=feature_importances_df.head(15), palette="crest_r") 
plt.title(f'Top 15 Features Mais Importantes para a Compra (RF - {dataset_tag})')
plt.tight_layout()
plt.savefig(f'{pasta_imagens}/importancia_features{sufixo_arquivos}.png', bbox_inches='tight') 
plt.show()

# 11. SALVANDO ARTEFATOS DO MODELO
print(f"\n--- Salvando Artefatos do Modelo ---")
metricas_para_salvar = {"algoritmo": "Random Forest (Fatores de Compra)", "acuracia_teste": float(accuracy)}
try:
    with open(f'metricas_modelo{sufixo_arquivos}.json', 'w') as f: 
        json.dump(metricas_para_salvar, f, indent=4)
    joblib.dump(model, f'modelo_final{sufixo_arquivos}.joblib') 
    joblib.dump(scaler, f'scaler_features_numericas{sufixo_arquivos}.joblib') 
    joblib.dump(colunas_modelo_treino_final, f'colunas_modelo_treino{sufixo_arquivos}.joblib') 
    joblib.dump(existing_numerical_in_X, f'colunas_numericas_escalonadas{sufixo_arquivos}.joblib') 
    print("Modelo, scaler, colunas e métricas salvos com sucesso!")
except Exception as e:
    print(f"Erro ao salvar os artefatos: {e}")

print("\nScript de análise finalizado!")
