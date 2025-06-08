import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json 
from datetime import datetime

# --- CONFIGURAÇÕES DA PÁGINA E ESTILO CSS ---
st.set_page_config(page_title="Dashboard Loja de Carros", layout="wide", initial_sidebar_state="expanded") 

# Injeção de CSS para estilização personalizada
custom_css = """
/* Tema Geral Escuro */
body {
    background-color: #0E1117;
}

/* Estilo da Barra Lateral */
[data-testid="stSidebar"] {
    background-color: #1a1a2e; /* Azul escuro */
}
[data-testid="stSidebar"] .st-emotion-cache-16txtl3 h1,
[data-testid="stSidebar"] .st-emotion-cache-16txtl3 h3,
[data-testid="stSidebar"] .st-emotion-cache-16txtl3 .st-emotion-cache-1g8i7f7 {
    color: #e0e0e0; /* Cor do texto claro */
}
[data-testid="stSidebar"] .st-emotion-cache-1g8i7f7 {
    font-size: 1.1rem;
}


/* Estilo das Métricas na Sidebar */
[data-testid="stMetric"] {
    background-color: rgba(43, 49, 79, 0.6); /* Azul-cinza transparente */
    border: 1px solid rgba(255, 255, 255, 0.2);
    border-radius: 10px;
    padding: 15px;
    margin-bottom: 10px;
}
[data-testid="stMetricLabel"] {
    font-size: 1rem;
    color: #a0a0a0; /* Cinza claro para o label */
}
[data-testid="stMetricValue"] {
    font-size: 1.8rem;
    color: #ffffff; /* Branco para o valor */
}

/* Estilo dos Botões */
div.stButton > button:first-child {
    background: linear-gradient(90deg, #6e45e2, #88d3ce);
    color: white;
    border: none;
    padding: 12px 24px;
    border-radius: 8px;
    font-weight: bold;
    font-size: 1rem;
    transition: all 0.3s ease-in-out;
}
div.stButton > button:hover {
    box-shadow: 0 0 15px rgba(110, 69, 226, 0.7);
    transform: scale(1.05);
}

/* Estilo das Abas (Tabs) */
.stTabs [data-baseweb="tab-list"] {
	gap: 24px;
}
.stTabs [data-baseweb="tab"] {
	height: 50px;
    white-space: pre-wrap;
	background-color: transparent;
	border-radius: 4px 4px 0px 0px;
	gap: 1px;
	padding-top: 10px;
	padding-bottom: 10px;
}
.stTabs [aria-selected="true"] {
  	background-color: #1a1a2e;
}

/* Estilo dos Containers para criar "cards" */
.st-emotion-cache-1r6slb0 {
    background-color: rgba(26, 26, 46, 0.7); /* Azul escuro transparente */
    border-radius: 15px;
    padding: 2rem;
    border: 1px solid rgba(255, 255, 255, 0.1);
    margin-bottom: 20px;
}
"""
st.markdown(f"<style>{custom_css}</style>", unsafe_allow_html=True)


# --- DEFINIÇÕES GLOBAIS ---
estados_disponiveis_dashboard = ['AC', 'AL', 'AM', 'AP', 'BA', 'CE', 'DF', 'ES', 'GO', 'MA', 'MG', 'MS', 'MT', 'PA', 'PB', 'PE', 'PI', 'PR', 'RJ', 'RN', 'RO', 'RR', 'RS', 'SC', 'SE', 'SP', 'TO']
marcas_disponiveis_dashboard = [
    'Volkswagen', 'Fiat', 'Chevrolet', 'Ford', 'Hyundai', 'Toyota', 'Honda', 
    'Renault', 'Jeep', 'Nissan', 'Peugeot', 'Citroën'
]

# Sufixo para arquivos específicos desta execução
sufixo_artefatos = "_rf_clf_compra_fatores_compra_10k" 
dataset_tag = sufixo_artefatos.split('_')[-1] if len(sufixo_artefatos.split('_')) > 1 else "Dataset"
pasta_imagens_dashboard = "img-geradas-fatores-compra" 

# --- FUNÇÃO DE CACHING PARA CARREGAR OS ARTEFATOS ---
@st.cache_data(show_spinner="Carregando modelo e dados...")
def carregar_artefatos():
    """Carrega o modelo, scalers e outros artefatos salvos."""
    try:
        model = joblib.load(f'modelo_final{sufixo_artefatos}.joblib')
        scaler_features_numericas = joblib.load(f'scaler_features_numericas{sufixo_artefatos}.joblib') 
        colunas_modelo_treino = joblib.load(f'colunas_modelo_treino{sufixo_artefatos}.joblib') 
        colunas_numericas_escalonadas = joblib.load(f'colunas_numericas_escalonadas{sufixo_artefatos}.joblib') 
        
        with open(f'metricas_modelo{sufixo_artefatos}.json', 'r') as f: 
            metricas = json.load(f)
        
        with open('ranking_modelos_vendidos.json', 'r', encoding='utf-8') as f:
            ranking_modelos = json.load(f)
            
        return model, scaler_features_numericas, colunas_modelo_treino, colunas_numericas_escalonadas, metricas, ranking_modelos

    except FileNotFoundError as fnf_error:
        st.error(f"Erro ao carregar arquivo: {fnf_error.filename}. "
                 "Execute o script 'analise_carros.py' completamente para gerar todos os artefatos necessários com os nomes corretos.")
        return None, None, None, None, None, None
    except Exception as e:
        st.error(f"Erro crítico ao carregar os artefatos do modelo: {e}")
        return None, None, None, None, None, None

# Carrega os artefatos usando a função com cache
model, scaler_features_numericas, colunas_modelo_treino, colunas_numericas_escalonadas, metricas, ranking_modelos_vendidos_data = carregar_artefatos()

if not model:
    st.stop()

acuracia_modelo = metricas.get("acuracia_teste", 0.0)
algoritmo_usado = metricas.get("algoritmo", "Modelo Desconhecido")

# --- FUNÇÃO DE PREDIÇÃO ---
def prever_status_venda_batch(df_input):
    """Realiza o pré-processamento e a predição para um DataFrame ou dicionário."""
    if isinstance(df_input, dict):
        df_pred = pd.DataFrame([df_input])
    else:
        df_pred = df_input.copy()
    
    if 'Ano_Modelo' in df_pred.columns and 'Idade_Carro_Modelo' in colunas_modelo_treino:
        df_pred['Idade_Carro_Modelo'] = datetime.now().year - df_pred['Ano_Modelo']
    
    if 'Mes_venda' in colunas_modelo_treino and 'Mes_venda' not in df_pred.columns:
        df_pred['Mes_venda'] = datetime.now().month
        df_pred['Dia_semana_venda'] = datetime.now().weekday()
        
    categorical_cols_base = ['Marca', 'Modelo', 'Combustivel', 'Cambio', 'Cor', 'Estado_Venda']
    existing_categorical_for_dummies = [col for col in categorical_cols_base if col in df_pred.columns]
    if existing_categorical_for_dummies:
        df_pred_encoded = pd.get_dummies(df_pred, columns=existing_categorical_for_dummies, drop_first=True)
    else:
        df_pred_encoded = df_pred.copy()
    
    df_pred_realigned = df_pred_encoded.reindex(columns=colunas_modelo_treino, fill_value=0)
    
    if colunas_numericas_escalonadas and scaler_features_numericas:
        cols_presentes_para_escalonar = [col for col in colunas_numericas_escalonadas if col in df_pred_realigned.columns]
        if cols_presentes_para_escalonar:
            df_pred_realigned[cols_presentes_para_escalonar] = scaler_features_numericas.transform(df_pred_realigned[cols_presentes_para_escalonar])
    
    predictions_num = model.predict(df_pred_realigned)
    predictions_proba = model.predict_proba(df_pred_realigned)
    
    if isinstance(df_input, dict):
        status_previsto = "Vendido" if predictions_num[0] == 1 else "Não Vendido"
        probabilidade_venda = predictions_proba[0][1] 
        return status_previsto, probabilidade_venda
    else: 
        return predictions_num, predictions_proba[:, 1]


# --- INTERFACE DO DASHBOARD ---
st.title(f"🚗 Dashboard de Análise e Predição - Loja de Carros")

# --- BARRA LATERAL ---
st.sidebar.title("Desempenho do Modelo") 
st.sidebar.info(f"**Algoritmo:** {algoritmo_usado}")
if model: 
    st.sidebar.metric(label="Acurácia no Teste", value=f"{acuracia_modelo:.2%}") 
    if "acuracia_cv_media" in metricas:
         st.sidebar.metric(label="Acurácia Média CV", value=f"{metricas.get('acuracia_cv_media', 0.0):.2%}")
st.sidebar.markdown("---")
st.sidebar.markdown(f"### Gráfico de Avaliação")
try:
    st.sidebar.image(f'{pasta_imagens_dashboard}/matriz_confusao_fatores_compra{sufixo_artefatos}.png', caption=f'Matriz de Confusão', use_container_width=True) 
except FileNotFoundError:
    st.sidebar.warning(f"Imagem 'matriz_confusao...{sufixo_artefatos}.png' não encontrada.")
except Exception as e_img_sidebar_eval:
    st.sidebar.error(f"Erro ao carregar imagem: {e_img_sidebar_eval}")
st.sidebar.markdown("---")
st.sidebar.caption("Projeto de Machine Learning")
st.sidebar.markdown("Desenvolvido por Ayslan Hugo<br>[GitHub](https://github.com/ayslanhugo)", unsafe_allow_html=True)


# <<< AJUSTE: Definindo 4 abas para corrigir o NameError >>>
tab1, tab2, tab3, tab4 = st.tabs(["📊 Predição e Fatores de Compra", "🏆 Ranking de Vendas", "📈 Visualizações EDA", "📎 Análise de Arquivo"])

with tab1:
    with st.container():
        st.header("🔍 Prever Probabilidade de Venda de um Carro")
        
        col_pred1, col_pred2 = st.columns(2)
        with col_pred1:
            marca_input = st.selectbox("Marca:", options=sorted(marcas_disponiveis_dashboard), key="pred_marca")
            ano_modelo_input = st.number_input("Ano do Modelo:", min_value=2000, max_value=datetime.now().year, value=2018, step=1, key="pred_ano_modelo")
            combustivel_input = st.selectbox("Combustível:", options=['Flex', 'Gasolina', 'Diesel', 'Etanol'], key="pred_combustivel")
            cor_input = st.selectbox("Cor:", options=['Preto', 'Branco', 'Prata', 'Cinza', 'Vermelho', 'Azul', 'Marrom', 'Outra'], key="pred_cor")
            
        with col_pred2:
            modelo_input = st.text_input("Modelo:", value="Gol", key="pred_modelo")
            km_input = st.number_input("Quilometragem:", min_value=0, value=50000, step=1000, key="pred_km")
            cambio_input = st.selectbox("Câmbio:", options=['Manual', 'Automático', 'CVT', 'Automatizado'], key="pred_cambio")
            portas_input = st.selectbox("Nº de Portas:", options=[2, 4], key="pred_portas")

        preco_input = st.slider("Preço de Listagem (R$):", min_value=10000.0, max_value=250000.0, value=50000.0, step=1000.0, format="R$ %.2f", key="pred_preco")

        if st.button("Prever Status da Venda", key="pred_botao_status", use_container_width=True):
            if modelo_input:
                input_features_dict = {
                    'Marca': marca_input, 'Modelo': modelo_input, 'Ano_Modelo': ano_modelo_input,
                    'Quilometragem': km_input, 'Combustivel': combustivel_input, 'Cambio': cambio_input,
                    'Cor': cor_input, 'Num_Portas': portas_input, 'Estado_Venda': 'SP', # Estado padrão para predição individual
                    'Preco_Listagem': preco_input
                }
                status_prev, prob_venda = prever_status_venda_batch(input_features_dict) 
                
                if status_prev == "Vendido":
                    st.success(f"**Status Previsto: {status_prev}**")
                else:
                    st.warning(f"**Status Previsto: {status_prev}**")

                st.progress(prob_venda, text=f"Probabilidade de Venda: {prob_venda:.2%}")
            else:
                st.error("Campo 'Modelo' não preenchido.")

    st.markdown("---")
    with st.container():
        st.header("🎯 Fatores Importantes para a Compra")
        st.markdown(f"Com base no modelo treinado, estes são os fatores mais importantes para determinar se um carro é vendido:")
        try:
            st.image(f'{pasta_imagens_dashboard}/importancia_features{sufixo_artefatos}.png', use_container_width=True)
        except FileNotFoundError:
            st.warning(f"Imagem 'importancia_features{sufixo_artefatos}.png' não encontrada.")

with tab2:
    st.header("🏆 Ranking de Modelos Mais Vendidos")
    st.markdown(f"Ranking baseado no dataset de {dataset_tag} (carros com status 'Vendido').")
    if ranking_modelos_vendidos_data:
        col_rank1, col_rank2 = st.columns(2)
        with col_rank1:
            st.subheader("Top 10 Modelos (Geral)")
            if ranking_modelos_vendidos_data.get("top_geral"):
                df_top_geral = pd.DataFrame(list(ranking_modelos_vendidos_data["top_geral"].items()), columns=['Modelo', 'Quantidade Vendida'])
                st.dataframe(df_top_geral, use_container_width=True, height=400)

        with col_rank2:
            st.subheader("Top 5 por Marca")
            if ranking_modelos_vendidos_data.get("top_por_marca"):
                marcas_com_ranking = sorted(list(ranking_modelos_vendidos_data["top_por_marca"].keys()))
                if marcas_com_ranking:
                    marca_selecionada_ranking = st.selectbox("Selecione uma Marca:", options=marcas_com_ranking, key="ranking_marca_select")
                    df_top_marca = pd.DataFrame(list(ranking_modelos_vendidos_data["top_por_marca"][marca_selecionada_ranking].items()), columns=['Modelo', 'Quantidade Vendida'])
                    st.dataframe(df_top_marca, use_container_width=True, height=250)

with tab3:
    st.header("📈 Visualizações EDA")
    st.markdown(f"Gráficos gerados a partir da Análise Exploratória de Dados do dataset {dataset_tag}.")
    try:
        st.image(f'{pasta_imagens_dashboard}/dist_foi_vendido_target{sufixo_artefatos}.png', caption='Distribuição da Target "Foi Vendido"', use_container_width=True)
    except FileNotFoundError:
        st.warning(f"Imagem da distribuição da target não encontrada.")
    except Exception as e_img_eda:
        st.error(f"Erro ao carregar imagens da EDA: {e_img_eda}")

with tab4:
    st.header("📎 Análise e Predição em Lote")
    st.markdown("Faça o upload de um arquivo Excel (.xlsx) ou CSV (.csv) com dados de carros para prever o status de venda de cada um.")
    st.info("O arquivo deve conter as colunas: 'Marca', 'Modelo', 'Ano_Modelo', 'Quilometragem', 'Preco_Listagem', etc.")

    uploaded_file = st.file_uploader("Escolha um arquivo", type=['csv', 'xlsx'], key="file_uploader")

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df_externo = pd.read_csv(uploaded_file, encoding='utf-8')
            else: 
                df_externo = pd.read_excel(uploaded_file)

            st.success("Arquivo carregado com sucesso!")
            st.dataframe(df_externo.head())

            if st.button("Prever Status de Venda para o arquivo", key="pred_arquivo_externo", use_container_width=True):
                required_cols_for_pred = ['Marca', 'Modelo', 'Ano_Modelo', 'Quilometragem', 'Preco_Listagem']
                if all(col in df_externo.columns for col in required_cols_for_pred):
                    with st.spinner("Realizando pré-processamento e fazendo previsões..."):
                        predictions_num, predictions_proba = prever_status_venda_batch(df_externo)
                        
                        df_resultados = df_externo.copy()
                        df_resultados['Status_Venda_Previsto'] = ['Vendido' if p == 1 else 'Não Vendido' for p in predictions_num]
                        df_resultados['Probabilidade_Venda'] = [f"{p:.2%}" for p in predictions_proba]
                    
                    st.success("Previsões concluídas!")
                    
                    colunas_para_exibir = ['Marca', 'Modelo', 'Ano_Modelo', 'Preco_Listagem', 'Status_Venda_Previsto', 'Probabilidade_Venda']
                    colunas_existentes_para_exibir = [col for col in colunas_para_exibir if col in df_resultados.columns]
                    st.dataframe(df_resultados[colunas_existentes_para_exibir])
                else:
                    cols_faltando = [col for col in required_cols_for_pred if col not in df_resultados.columns]
                    st.error(f"Colunas necessárias não encontradas no arquivo: {cols_faltando}")
        except Exception as e_upload:
            st.error(f"Erro ao processar o arquivo: {e_upload}")
            st.warning("Certifique-se de que é um arquivo CSV ou Excel válido.")
