import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json 
from datetime import datetime

# --- CONFIGURAÇÕES DA PÁGINA E ESTILO CSS ---
st.set_page_config(page_title="Dashboard Loja de Carros", layout="wide", initial_sidebar_state="expanded") 

# Injeção de CSS para um visual mais moderno
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
marcas_disponiveis_dashboard = [
    'Volkswagen', 'Fiat', 'Chevrolet', 'Ford', 'Hyundai', 'Toyota', 'Honda', 
    'Renault', 'Jeep', 'Nissan', 'Peugeot', 'Citroën'
]
estados_disponiveis_dashboard = ['AC', 'AL', 'AM', 'AP', 'BA', 'CE', 'DF', 'ES', 'GO', 'MA', 'MG', 'MS', 'MT', 'PA', 'PB', 'PE', 'PI', 'PR', 'RJ', 'RN', 'RO', 'RR', 'RS', 'SC', 'SE', 'SP', 'TO']

# Sufixo para carregar os arquivos corretos
# !!! AJUSTE ESTE SUFIXO PARA CORRESPONDER AO USADO NO SEU ÚLTIMO 'analise_carros.py' !!!
sufixo_artefatos = "_rf_fatores_compra_10k" 
dataset_tag = "10k" # Apenas a tag do tamanho do dataset
pasta_imagens_dashboard = "img-geradas-fatores-compra" 

# --- FUNÇÃO DE CACHING PARA CARREGAR OS ARTEFATOS ---
@st.cache_data(show_spinner="Carregando modelo e dados...")
def carregar_artefatos():
    """Carrega o modelo, scalers e outros artefatos salvos."""
    try:
        model = joblib.load(f'modelo_final{sufixo_artefatos}.joblib')
        scaler = joblib.load(f'scaler_features_numericas{sufixo_artefatos}.joblib') 
        colunas_treino = joblib.load(f'colunas_modelo_treino{sufixo_artefatos}.joblib') 
        colunas_escalonadas = joblib.load(f'colunas_numericas_escalonadas{sufixo_artefatos}.joblib') 
        
        with open(f'metricas_modelo{sufixo_artefatos}.json', 'r') as f: 
            metricas = json.load(f)
        
        with open('ranking_modelos_vendidos.json', 'r', encoding='utf-8') as f:
            ranking_modelos = json.load(f)
            
        return model, scaler, colunas_treino, colunas_escalonadas, metricas, ranking_modelos

    except FileNotFoundError as fnf_error:
        st.error(f"Erro ao carregar arquivo: {fnf_error.filename}. "
                 "Execute o script 'analise_carros.py' completamente para gerar todos os artefatos.")
        return None, None, None, None, None, None
    except Exception as e:
        st.error(f"Erro crítico ao carregar artefatos do modelo: {e}")
        return None, None, None, None, None, None

model, scaler, colunas_modelo_treino, colunas_numericas_escalonadas, metricas, ranking_modelos_vendidos_data = carregar_artefatos()

if not model:
    st.stop()

# Extrai as métricas
acuracia_modelo = metricas.get("acuracia_teste", 0.0)
algoritmo_usado = metricas.get("algoritmo", "Modelo Desconhecido")

# --- FUNÇÃO DE PREDIÇÃO ---
def prever_status_venda_batch(df_input):
    """Realiza o pré-processamento e a predição para um DataFrame ou dicionário."""
    # Garante que a entrada seja sempre um DataFrame
    if isinstance(df_input, dict):
        df_pred = pd.DataFrame([df_input])
    else:
        df_pred = df_input.copy()

    # Engenharia de Features Consistente com o Treinamento
    if 'Ano_Modelo' in df_pred.columns and 'Idade_Carro_Modelo' in colunas_modelo_treino:
        df_pred['Idade_Carro_Modelo'] = datetime.now().year - df_pred['Ano_Modelo']
    
    # One-Hot Encoding
    categorical_cols = ['Marca', 'Modelo', 'Combustivel', 'Cambio', 'Cor', 'Estado_Venda']
    df_pred_encoded = pd.get_dummies(df_pred, columns=[c for c in categorical_cols if c in df_pred.columns], drop_first=True)
    
    # Realinhamento de Colunas
    df_pred_realigned = df_pred_encoded.reindex(columns=colunas_modelo_treino, fill_value=0)
    
    # Escalonamento
    if colunas_numericas_escalonadas and scaler:
        cols_a_escalonar = [c for c in colunas_numericas_escalonadas if c in df_pred_realigned.columns]
        if cols_a_escalonar:
            df_pred_realigned[cols_a_escalonar] = scaler.transform(df_pred_realigned[cols_a_escalonar])
    
    # Predição
    pred_num = model.predict(df_pred_realigned)
    pred_proba = model.predict_proba(df_pred_realigned)
    
    if isinstance(df_input, dict):
        status = "Venda Provável" if pred_num[0] == 1 else "Venda Improvável"
        prob = pred_proba[0][1]
        return status, prob
    else: 
        return pred_num, pred_proba[:, 1]

# --- INTERFACE DO DASHBOARD ---
st.title(f"🚗 Dashboard Loja de Carros Usados")
st.markdown(f"Análise e predição baseadas em um modelo **{algoritmo_usado}** treinado em um dataset de **{dataset_tag}** carros.")

# BARRA LATERAL
st.sidebar.title("Desempenho do Modelo") 
st.sidebar.info(f"**Algoritmo:** {algoritmo_usado}")
if model: 
    st.sidebar.metric(label="Acurácia no Teste", value=f"{acuracia_modelo:.2%}") 
    if "acuracia_cv_media" in metricas:
         st.sidebar.metric(label="Acurácia Média CV", value=f"{metricas.get('acuracia_cv_media', 0.0):.2%}")
st.sidebar.markdown("---")
st.sidebar.markdown(f"### Gráfico de Avaliação")
try:
    st.sidebar.image(f'{pasta_imagens_dashboard}/matriz_confusao{sufixo_artefatos}.png', caption=f'Matriz de Confusão', use_container_width=True) 
except FileNotFoundError:
    st.sidebar.warning(f"Imagem 'matriz_confusao...{sufixo_artefatos}.png' não encontrada.")
except Exception as e_img_sidebar_eval:
    st.sidebar.error(f"Erro ao carregar imagem: {e_img_sidebar_eval}")
st.sidebar.markdown("---")
st.sidebar.caption("Projeto de Machine Learning")
st.sidebar.markdown("Desenvolvido por Ayslan Hugo<br>[GitHub](https://github.com/ayslanhugo)", unsafe_allow_html=True)

# <<< AJUSTE: Definindo 4 abas para corrigir o NameError >>>
tab1, tab2, tab3, tab4 = st.tabs(["📊 Predição e Fatores de Venda", "🏆 Ranking de Vendas", "📈 Visualizações EDA", "📎 Análise de Arquivo"])

with tab1:
    with st.container():
        st.header("🔍 Prever Probabilidade de Venda de um Carro")
        
        col_pred1, col_pred2 = st.columns(2)
        with col_pred1:
            marca_input = st.selectbox("Marca", options=sorted(marcas_disponiveis_dashboard))
            ano_modelo_input = st.number_input("Ano do Modelo", min_value=2000, max_value=datetime.now().year, value=2018)
            combustivel_input = st.selectbox("Combustível", options=['Flex', 'Gasolina', 'Diesel', 'Etanol'])
            cor_input = st.selectbox("Cor", options=['Preto', 'Branco', 'Prata', 'Cinza', 'Vermelho', 'Azul', 'Marrom', 'Outra'])
            
        with col_pred2:
            modelo_input = st.text_input("Modelo", value="Onix")
            km_input = st.number_input("Quilometragem", min_value=0, value=50000)
            cambio_input = st.selectbox("Câmbio", options=['Manual', 'Automático', 'CVT', 'Automatizado'])
            portas_input = st.selectbox("Nº de Portas", options=[2, 4])
        
        preco_input = st.slider("Preço de Listagem (R$)", 10000.0, 250000.0, 50000.0, 1000.0, "R$ %.2f")

        if st.button("Prever Status da Venda", use_container_width=True):
            if modelo_input:
                input_data = {'Marca': marca_input, 'Modelo': modelo_input, 'Ano_Modelo': ano_modelo_input,
                              'Quilometragem': km_input, 'Combustivel': combustivel_input, 'Cambio': cambio_input,
                              'Cor': cor_input, 'Num_Portas': portas_input, 'Estado_Venda': 'SP'} # Estado padrão para simplificar UI
                input_data['Preco_Listagem'] = preco_input
                
                status, prob = prever_status_venda_batch(input_data)
                
                if status == "Venda Provável": st.success(f"**Status Previsto:** {status}")
                else: st.warning(f"**Status Previsto:** {status}")
                st.progress(prob, text=f"Probabilidade de Venda: {prob:.1%}")
            else:
                st.error("O campo 'Modelo' é obrigatório.")

    st.markdown("---")
    with st.container():
        st.header("🎯 Fatores Decisivos para a Venda")
        st.markdown(f"Com base no modelo treinado, estes são os fatores mais importantes para determinar se um carro é vendido:")
        try:
            st.image(f'{pasta_imagens_dashboard}/importancia_features{sufixo_artefatos}.png', use_container_width=True)
        except FileNotFoundError:
            st.warning(f"Imagem 'importancia_features{sufixo_artefatos}.png' não encontrada.")

with tab2:
    st.header("🏆 Ranking de Modelos Mais Vendidos")
    if ranking_modelos_vendidos_data:
        col_rank1, col_rank2 = st.columns(2)
        with col_rank1:
            st.subheader("Top 10 Geral")
            if ranking_modelos_vendidos_data.get("top_geral"):
                df_top_geral = pd.DataFrame(list(ranking_modelos_vendidos_data["top_geral"].items()), columns=['Modelo', 'Unidades Vendidas'])
                st.dataframe(df_top_geral, use_container_width=True, height=400)
        with col_rank2:
            st.subheader("Top 5 por Marca")
            if ranking_modelos_vendidos_data.get("top_por_marca"):
                marcas_com_ranking = sorted(list(ranking_modelos_vendidos_data["top_por_marca"].keys()))
                if marcas_com_ranking:
                    marca_sel = st.selectbox("Selecione uma Marca:", options=marcas_com_ranking)
                    df_top_marca = pd.DataFrame(list(ranking_modelos_vendidos_data["top_por_marca"][marca_sel].items()), columns=['Modelo', 'Unidades Vendidas'])
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
    st.markdown("Faça o upload de um arquivo CSV ou Excel com os mesmos cabeçalhos do dataset de treino.")
    uploaded_file = st.file_uploader("Escolha um arquivo", type=['csv', 'xlsx'])
    if uploaded_file:
        try:
            df_externo = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
            st.success("Arquivo carregado!")
            st.dataframe(df_externo.head())
            if st.button("Prever para todo o arquivo", use_container_width=True):
                required_cols_for_pred = ['Marca', 'Modelo', 'Ano_Modelo', 'Quilometragem', 'Preco_Listagem']
                if all(col in df_externo.columns for col in required_cols_for_pred):
                    with st.spinner("Fazendo previsões..."):
                        pred_num, pred_proba = prever_status_venda_batch(df_externo)
                        df_externo['Status_Previsto'] = ["Vendido" if p == 1 else "Não Vendido" for p in pred_num]
                        df_externo['Probabilidade_Venda'] = [f"{p:.1%}" for p in pred_proba]
                        st.dataframe(df_externo[['Marca', 'Modelo', 'Ano_Modelo', 'Preco_Listagem', 'Status_Previsto', 'Probabilidade_Venda']])
                else:
                    cols_faltando = [col for col in required_cols_for_pred if col not in df_externo.columns]
                    st.error(f"Colunas necessárias não encontradas no arquivo: {cols_faltando}")
        except Exception as e:
            st.error(f"Erro ao processar o arquivo: {e}")

