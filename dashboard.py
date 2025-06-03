import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json 
from datetime import datetime

# --- CONFIGURAÇÕES DA PÁGINA ---
st.set_page_config(page_title="Dashboard Loja de Carros", layout="wide") 

# --- DEFINIÇÕES GLOBAIS ---
# Estas listas devem ser consistentes com os dados usados no treinamento
segmentos_cliente_placeholder = ['Consumidor', 'Corporativo', 'Home Office'] # Usado se colunas_modelo_treino não carregar
estados_disponiveis_dashboard = ['AC', 'AL', 'AM', 'AP', 'BA', 'CE', 'DF', 'ES', 'GO', 'MA', 'MG', 'MS', 'MT', 'PA', 'PB', 'PE', 'PI', 'PR', 'RJ', 'RN', 'RO', 'RR', 'RS', 'SC', 'SE', 'SP', 'TO']
marcas_disponiveis_dashboard = [
    'Volkswagen', 'Fiat', 'Chevrolet', 'Ford', 'Hyundai', 'Toyota', 'Honda', 
    'Renault', 'Jeep', 'Nissan', 'Peugeot', 'Citroën'
] # Para o filtro de ranking por marca

# Sufixo para arquivos específicos desta execução (modelo e dataset)
# !!! AJUSTE ESTE SUFIXO PARA CORRESPONDER AO USADO NO SEU ÚLTIMO 'analise_carros.py' !!!
sufixo_artefatos = "_rf_clf_compra_fatores_compra_10k" # Exemplo
dataset_tag = sufixo_artefatos.split('_')[-1] if len(sufixo_artefatos.split('_')) > 1 else "Dataset"
model_name_tag = sufixo_artefatos.split('_')[1] if len(sufixo_artefatos.split('_')) > 2 else "Modelo"


pasta_imagens_dashboard = "img-geradas-fatores-compra" 

# --- CARREGAR MODELO, ARTEFATOS, MÉTRICAS E RANKINGS ---
model = None
label_encoder_placeholder = None # Não temos mais label_encoder para o alvo (0/1)
scaler_features_numericas = None 
colunas_modelo_treino = None
colunas_numericas_escalonadas = None 
acuracia_modelo = 0.0 
algoritmo_usado = "Não Carregado"
ranking_modelos_vendidos_data = {"top_geral": {}, "top_por_marca": {}}

try:
    model = joblib.load(f'modelo_final{sufixo_artefatos}.joblib')
    scaler_features_numericas = joblib.load(f'scaler_features_numericas{sufixo_artefatos}.joblib') 
    colunas_modelo_treino = joblib.load(f'colunas_modelo_treino{sufixo_artefatos}.joblib') 
    colunas_numericas_escalonadas = joblib.load(f'colunas_numericas_escalonadas{sufixo_artefatos}.joblib') 
    
    with open(f'metricas_modelo{sufixo_artefatos}.json', 'r') as f: 
        metricas_carregadas = json.load(f)
    acuracia_modelo = metricas_carregadas.get("acuracia_teste", 0.0)
    # Adicione outras métricas se salvou, ex: acuracia_cv_media
    # acuracia_cv_media = metricas_carregadas.get("acuracia_cv_media", 0.0) 
    algoritmo_usado = metricas_carregadas.get("algoritmo", "Modelo Desconhecido")
    st.sidebar.success(f"Métricas ({algoritmo_usado}) carregadas!")

    with open('ranking_modelos_vendidos.json', 'r', encoding='utf-8') as f:
        ranking_modelos_vendidos_data = json.load(f)
    st.sidebar.success("Ranking de modelos vendidos carregado!")

except FileNotFoundError as fnf_error:
    st.error(f"Erro ao carregar arquivo: {fnf_error.filename}. "
             f"Execute o script 'analise_carros.py' (com sufixo '{sufixo_artefatos}') "
             "completamente primeiro para gerar todos os artefatos necessários.")
    st.stop()
except Exception as e:
    st.error(f"Erro crítico ao carregar os artefatos do modelo: {e}")
    st.stop()

# --- FUNÇÃO DE PREDIÇÃO ---
def prever_status_venda_dashboard(input_data_dict):
    # input_data_dict deve conter todas as features base:
    # 'Marca', 'Modelo', 'Ano_Modelo', 'Quilometragem', 'Combustivel', 
    # 'Cambio', 'Cor', 'Num_Portas', 'Estado_Venda', 'Preco_Listagem'
    # e opcionalmente 'Data_Pedido_str' para features de data

    df_entrada = pd.DataFrame([input_data_dict])

    # Engenharia de features de data (se aplicável e se o modelo foi treinado com elas)
    if 'Data_Pedido_str' in df_entrada.columns and 'Mes_venda' in colunas_modelo_treino and 'Dia_semana_venda' in colunas_modelo_treino:
        try:
            data_obj = pd.to_datetime(df_entrada['Data_Pedido_str'].iloc[0])
            df_entrada['Mes_venda'] = data_obj.month
            df_entrada['Dia_semana_venda'] = data_obj.dayofweek
        except:
            st.warning("Data inválida para predição, usando valores padrão para features de data.")
            df_entrada['Mes_venda'] = 6 
            df_entrada['Dia_semana_venda'] = 2
    elif 'Mes_venda' in colunas_modelo_treino and 'Dia_semana_venda' in colunas_modelo_treino:
        # Se o modelo espera, mas data não foi fornecida, usar defaults
        df_entrada['Mes_venda'] = 6 
        df_entrada['Dia_semana_venda'] = 2


    # Criar 'Idade_Carro_Modelo' se o modelo espera
    if 'Idade_Carro_Modelo' in colunas_modelo_treino and 'Ano_Modelo' in df_entrada.columns:
        df_entrada['Idade_Carro_Modelo'] = datetime.now().year - df_entrada['Ano_Modelo']
    elif 'Idade_Carro_Modelo' in colunas_modelo_treino:
         df_entrada['Idade_Carro_Modelo'] = 5 # Um valor padrão se Ano_Modelo não for fornecido

    # One-Hot Encoding
    # Identificar colunas categóricas base que o modelo espera (antes do _Encoded)
    # Essas são as colunas que foram passadas para get_dummies no treinamento
    categorical_cols_base = ['Marca', 'Modelo', 'Combustivel', 'Cambio', 'Cor', 'Estado_Venda']
    existing_categorical_for_dummies = [col for col in categorical_cols_base if col in df_entrada.columns]
    if existing_categorical_for_dummies:
        df_entrada_encoded = pd.get_dummies(df_entrada, columns=existing_categorical_for_dummies, drop_first=True)
    else:
        df_entrada_encoded = df_entrada.copy()
    
    df_entrada_realigned = df_entrada_encoded.reindex(columns=colunas_modelo_treino, fill_value=0)
    
    if colunas_numericas_escalonadas and scaler_features_numericas:
        cols_presentes_para_escalonar = [col for col in colunas_numericas_escalonadas if col in df_entrada_realigned.columns]
        if cols_presentes_para_escalonar:
            df_entrada_realigned[cols_presentes_para_escalonar] = scaler_features_numericas.transform(df_entrada_realigned[cols_presentes_para_escalonar])
    
    predicao_numerica = model.predict(df_entrada_realigned)[0]
    predicao_probs = model.predict_proba(df_entrada_realigned)[0]
    
    status_previsto = "Vendido" if predicao_numerica == 1 else "Não Vendido"
    probabilidade_venda = predicao_probs[1] # Probabilidade da classe 1 (Vendido)
    
    return status_previsto, probabilidade_venda

# --- INTERFACE DO DASHBOARD ---
st.title(f"🚗 Dashboard Loja de Carros Usados ({algoritmo_usado} - {dataset_tag})")

# Abas
tab1, tab2, tab3 = st.tabs(["📊 Fatores de Compra e Predição", "🏆 Modelos Mais Vendidos", "📈 Visualizações EDA"])

with tab1:
    st.header("🔍 Análise de Fatores e Predição de Venda")
    st.markdown("Insira as características de um carro para prever a probabilidade de venda e identificar os fatores que influenciam a compra.")

    st.subheader("Fatores Importantes para a Compra")
    st.markdown(f"Com base no modelo treinado ({algoritmo_usado}), os seguintes fatores foram identificados como os mais importantes para determinar se um carro é vendido ou não (usando o dataset {dataset_tag}):")
    try:
        st.image(f'{pasta_imagens_dashboard}/importancia_features{sufixo_artefatos}.png', 
                 caption=f'Importância das Features para "Foi Vendido"', 
                 use_container_width=True)
    except FileNotFoundError:
        st.warning(f"Imagem 'importancia_features{sufixo_artefatos}.png' não encontrada.")

    st.subheader("Prever Status de Venda de um Carro")
    
    # Inputs para predição
    # Idealmente, as opções para Modelo seriam dinâmicas baseadas na Marca, mas simplificamos aqui
    # e o usuário digitaria o modelo. Ou teríamos uma lista muito grande de modelos.
    
    col_pred1, col_pred2 = st.columns(2)
    with col_pred1:
        marca_input = st.selectbox("Marca:", options=sorted(marcas_disponiveis_dashboard), key="pred_marca")
        ano_modelo_input = st.number_input("Ano do Modelo:", min_value=2000, max_value=datetime.now().year, value=2018, step=1, key="pred_ano_modelo")
        combustivel_input = st.selectbox("Combustível:", options=['Flex', 'Gasolina', 'Diesel', 'Etanol'], key="pred_combustivel")
        cor_input = st.selectbox("Cor:", options=['Preto', 'Branco', 'Prata', 'Cinza', 'Vermelho', 'Azul', 'Marrom', 'Outra'], key="pred_cor")
        estado_input = st.selectbox("Estado de Venda:", options=sorted(estados_disponiveis_dashboard), key="pred_estado_venda")

    with col_pred2:
        modelo_input = st.text_input("Modelo:", value="Gol", key="pred_modelo") # Usuário digita o modelo
        km_input = st.number_input("Quilometragem:", min_value=0, value=50000, step=1000, key="pred_km")
        cambio_input = st.selectbox("Câmbio:", options=['Manual', 'Automático', 'CVT', 'Automatizado'], key="pred_cambio")
        portas_input = st.selectbox("Nº de Portas:", options=[2, 4], key="pred_portas")
        preco_input = st.number_input("Preço de Listagem (R$):", min_value=0.0, value=50000.0, step=1000.0, format="%.2f", key="pred_preco")

    if st.button("Prever Status da Venda", key="pred_botao_status"):
        if modelo_input and model:
            input_features_dict = {
                'Marca': marca_input,
                'Modelo': modelo_input,
                'Ano_Modelo': ano_modelo_input,
                'Quilometragem': km_input,
                'Combustivel': combustivel_input,
                'Cambio': cambio_input,
                'Cor': cor_input,
                'Num_Portas': portas_input,
                'Estado_Venda': estado_input,
                'Preco_Listagem': preco_input
                # Data_Pedido_str não está sendo coletada, então a função usará defaults para Mes/Dia
            }
            status_prev, prob_venda = prever_status_venda_dashboard(input_features_dict)
            
            
            st.progress(prob_venda)
            st.markdown(f"**Probabilidade Estimada de Venda:** <span style='color:DodgerBlue; font-size: 1.1em;'>{prob_venda:.2%}</span>", unsafe_allow_html=True)
        else:
            st.error("Modelo não carregado ou campo 'Modelo' não preenchido.")

with tab2:
    st.header("🏆 Modelos de Carros Mais Vendidos")
    st.markdown(f"Ranking baseado no dataset de {dataset_tag} (carros com status 'Vendido').")

    if ranking_modelos_vendidos_data:
        st.subheader("Top 10 Modelos Mais Vendidos (Geral)")
        if ranking_modelos_vendidos_data.get("top_geral"):
            df_top_geral = pd.DataFrame(list(ranking_modelos_vendidos_data["top_geral"].items()), columns=['Modelo', 'Quantidade Vendida'])
            st.dataframe(df_top_geral)
        else:
            st.info("Não há dados para o ranking geral de modelos.")

        st.subheader("Top 5 Modelos Mais Vendidos por Marca")
        if ranking_modelos_vendidos_data.get("top_por_marca"):
            marcas_com_ranking = sorted(list(ranking_modelos_vendidos_data["top_por_marca"].keys()))
            if marcas_com_ranking:
                marca_selecionada_ranking = st.selectbox("Selecione uma Marca:", options=marcas_com_ranking, key="ranking_marca_select")
                if marca_selecionada_ranking in ranking_modelos_vendidos_data["top_por_marca"]:
                    df_top_marca = pd.DataFrame(list(ranking_modelos_vendidos_data["top_por_marca"][marca_selecionada_ranking].items()), columns=['Modelo', 'Quantidade Vendida'])
                    st.dataframe(df_top_marca)
            else:
                st.info("Não há dados para o ranking de modelos por marca.")
        else:
            st.info("Não há dados para o ranking de modelos por marca.")
    else:
        st.warning("Arquivo 'ranking_modelos_vendidos.json' não encontrado ou vazio. Execute 'analise_carros.py' para gerá-lo.")


with tab3:
    st.header("📈 Outras Visualizações EDA")
    st.markdown(f"Gráficos gerados a partir da Análise Exploratória de Dados do dataset {dataset_tag}.")
    
    try:
        st.image(f'{pasta_imagens_dashboard}/distribuicao_foi_vendido_target{sufixo_artefatos}.png', caption='Distribuição da Target "Foi Vendido"', use_container_width=True)
        st.image(f'{pasta_imagens_dashboard}/distribuicao_preco_venda{sufixo_artefatos}.png', caption='Distribuição do Preço de Listagem (Todos os Carros)', use_container_width=True)
        # Adicione mais gráficos da EDA aqui se desejar
        # st.image(f'{pasta_imagens_dashboard}/boxplot_preco_por_modelo_vw{sufixo_artefatos}.png', caption='Boxplot Preço por Modelo VW', use_container_width=True)
        # st.image(f'{pasta_imagens_dashboard}/matriz_correlacao_numericas{sufixo_artefatos}.png', caption='Matriz de Correlação', use_container_width=True)

    except FileNotFoundError:
        st.warning(f"Algumas imagens de visualização EDA com sufixo '{sufixo_artefatos}.png' não foram encontradas.")
    except Exception as e_img_eda:
        st.error(f"Erro ao carregar imagens da EDA: {e_img_eda}")


# --- Informações na Barra Lateral ---
st.sidebar.title(f"Desempenho do Modelo") 
st.sidebar.info(f"**Algoritmo:** {algoritmo_usado}")
if model: 
    st.sidebar.metric(label="Acurácia no Teste", value=f"{acuracia_modelo:.2%}") 
    # Adicione a acurácia CV se estiver no JSON de métricas
    if "acuracia_cv_media" in metricas_carregadas:
         st.sidebar.metric(label="Acurácia Média CV", value=f"{metricas_carregadas['acuracia_cv_media']:.2%}")
st.sidebar.markdown("---")
st.sidebar.markdown(f"### Gráficos de Avaliação")
try:
    st.sidebar.image(f'{pasta_imagens_dashboard}/matriz_confusao_fatores_compra{sufixo_artefatos}.png', caption=f'Matriz de Confusão', use_container_width=True) 
    # A imagem de importância de features já está na Tab1
except FileNotFoundError:
    st.sidebar.warning(f"Imagem 'matriz_confusao...{sufixo_artefatos}.png' não encontrada.")
except Exception as e_img_sidebar_eval:
    st.sidebar.error(f"Erro ao carregar imagem de avaliação da sidebar: {e_img_sidebar_eval}")

st.sidebar.markdown("---")
st.sidebar.caption("Projeto de Machine Learning - Loja de Carros Usados")
