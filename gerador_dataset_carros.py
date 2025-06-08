import pandas as pd
import numpy as np
import random
from datetime import datetime

print("Iniciando a geração do dataset de carros com status de venda...")

# --- Parâmetros de Geração ---
num_amostras = 50000 # Número de carros listados

# --- Definições de Base para as Características ---
marcas_modelos = {
    'Volkswagen': ['Gol', 'Polo', 'Virtus', 'T-Cross', 'Nivus', 'Fox', 'Voyage', 'Jetta', 'Tiguan', 'Saveiro', 'Amarok', 'Up!'],
    'Fiat': ['Mobi', 'Argo', 'Cronos', 'Strada', 'Toro', 'Pulse', 'Uno', 'Palio', 'Siena', 'Punto'],
    'Chevrolet': ['Onix', 'Onix Plus', 'Tracker', 'Montana', 'S10', 'Spin', 'Cruze', 'Prisma', 'Cobalt'],
    'Ford': ['Ka', 'EcoSport', 'Ranger', 'Fiesta', 'Focus'], # Modelos comuns no mercado de usados
    'Hyundai': ['HB20', 'HB20S', 'Creta', 'Tucson', 'i30'],
    'Toyota': ['Corolla', 'Hilux', 'Yaris', 'Etios', 'SW4', 'RAV4'],
    'Honda': ['Civic', 'HR-V', 'City', 'Fit', 'WR-V', 'CR-V'],
    'Renault': ['Kwid', 'Sandero', 'Logan', 'Duster', 'Captur', 'Oroch', 'Clio'],
    'Jeep': ['Renegade', 'Compass', 'Commander'],
    'Nissan': ['Kicks', 'Versa', 'March', 'Frontier', 'Sentra'],
    'Peugeot': ['208', '2008', '3008'],
    'Citroën': ['C3', 'C4 Cactus', 'Aircross']
}
lista_marcas = list(marcas_modelos.keys())
anos_fabricacao = list(range(2005, datetime.now().year + 1)) # Carros de 2005 até o ano atual
tipos_combustivel = ['Flex', 'Gasolina', 'Diesel', 'Etanol']
pesos_combustivel = [0.78, 0.12, 0.08, 0.02]
tipos_cambio = ['Manual', 'Automático', 'CVT', 'Automatizado']
pesos_cambio = [0.5, 0.35, 0.1, 0.05]
cores_populares = ['Preto', 'Branco', 'Prata', 'Cinza', 'Vermelho', 'Azul', 'Marrom']
estados_brasil = ['SP', 'RJ', 'MG', 'RS', 'PR', 'BA', 'CE', 'PE', 'SC', 'GO', 'DF', 'ES']

# --- Geração dos Dados ---
dados_carros_status = []
current_year = datetime.now().year

for i in range(num_amostras):
    if (i + 1) % (num_amostras // 20) == 0:
        print(f"  Gerando carro listado {i+1}/{num_amostras}...")

    marca = random.choice(lista_marcas)
    modelo = random.choice(marcas_modelos[marca])
    ano_fab = random.choice(anos_fabricacao)
    ano_mod = random.choice([ano_fab, ano_fab + 1]) if ano_fab < current_year else ano_fab
    idade_carro = current_year - ano_mod + 0.1 # Adiciona 0.1 para evitar idade zero e problemas com divisão

    km_base_anual = random.uniform(7000, 20000)
    quilometragem = max(500, int(idade_carro * km_base_anual * random.uniform(0.7, 1.3)))

    combustivel = np.random.choice(tipos_combustivel, p=pesos_combustivel)
    cambio = np.random.choice(tipos_cambio, p=pesos_cambio)
    cor = random.choice(cores_populares)
    num_portas = 4 if random.random() > 0.25 else 2
    estado_venda = random.choice(estados_brasil)

    # Lógica simplificada para Preço de Listagem
    preco_listagem = 60000 
    if marca in ['Toyota', 'Honda', 'Jeep']: preco_listagem *= random.uniform(1.3, 2.0)
    elif marca in ['Hyundai', 'Nissan']: preco_listagem *= random.uniform(1.1, 1.5)
    elif marca in ['Ford', 'Peugeot', 'Citroën']: preco_listagem *= random.uniform(0.8, 1.2)
    else: preco_listagem *= random.uniform(0.6, 1.1) # VW, Fiat, Chevrolet, Renault
    
    fator_ano = (ano_mod - (min(anos_fabricacao)-1) ) / (current_year - (min(anos_fabricacao)-1) + 1e-6)
    preco_listagem *= (0.25 + 0.75 * max(0.1, fator_ano)**2.2) 
    
    fator_km = max(0.2, 1 - (quilometragem / 250000))
    preco_listagem *= fator_km
    
    if cambio == 'Automático' or cambio == 'CVT': preco_listagem *= 1.1
    preco_listagem = max(10000, int(preco_listagem * random.uniform(0.85, 1.15)))

    # Lógica para determinar se o carro Foi_Vendido (TARGET)
    # Fatores que aumentam a probabilidade de venda:
    # - Preço de listagem "bom" (não muito alto em relação às suas características)
    # - Carro mais novo
    # - Baixa KM
    # - Marca/modelo popular
    # - Câmbio automático (tendência)
    
    prob_venda_base = 0.4 # Probabilidade base de um carro ser vendido

    # Ajuste por idade (mais novo, maior chance)
    if idade_carro <= 3: prob_venda_base += 0.25
    elif idade_carro <= 7: prob_venda_base += 0.10
    else: prob_venda_base -= (idade_carro / 100) * 1.5


    # Ajuste por KM (menor KM, maior chance)
    if quilometragem <= 50000: prob_venda_base += 0.15
    elif quilometragem <= 100000: prob_venda_base += 0.05
    else: prob_venda_base -= (quilometragem / 1000000) * 1.5


    # Ajuste por Preço de Listagem (simulando um "preço de mercado" e comparando)
    # Esta é uma simplificação grosseira
    preco_mercado_estimado = preco_listagem / random.uniform(0.90, 1.10) # Simula uma estimativa de mercado
    if preco_listagem < preco_mercado_estimado * 0.95: # Preço atrativo
        prob_venda_base += 0.20
    elif preco_listagem > preco_mercado_estimado * 1.10: # Preço alto
        prob_venda_base -= 0.25
    
    # Ajuste por popularidade (simplificado)
    if marca in ['Toyota', 'Honda', 'Hyundai', 'Volkswagen', 'Fiat', 'Chevrolet'] or \
       modelo in ['Onix', 'HB20', 'Gol', 'Strada', 'Corolla', 'Compass', 'Renegade', 'Tracker', 'Creta', 'Polo']:
        prob_venda_base += 0.10
        
    if cambio == 'Automático' or cambio == 'CVT':
        prob_venda_base += 0.05

    # Limitar probabilidade entre 0.05 e 0.95
    prob_venda_final = np.clip(prob_venda_base, 0.05, 0.95)
    
    foi_vendido = 1 if random.random() < prob_venda_final else 0
    
    dados_carros_status.append({
        'Marca': marca,
        'Modelo': modelo,
        'Ano_Fabricacao': ano_fab,
        'Ano_Modelo': ano_mod,
        'Quilometragem': quilometragem,
        'Combustivel': combustivel,
        'Cambio': cambio,
        'Cor': cor,
        'Num_Portas': num_portas,
        'Estado_Venda': estado_venda,
        'Preco_Listagem': preco_listagem,
        'Foi_Vendido': foi_vendido # Nossa variável alvo
    })

df_carros_status = pd.DataFrame(dados_carros_status)

# --- Salvando em CSV ---
nome_arquivo_csv_final = 'carros_status_venda_dataset.csv'
df_carros_status.to_csv(nome_arquivo_csv_final, index=False, encoding='utf-8')

print(f"\nDataset com {num_amostras} amostras de carros com status de venda gerado e salvo como '{nome_arquivo_csv_final}'!")
print("Primeiras 5 linhas do dataset gerado:")
print(df_carros_status.head())
print("\nDistribuição da variável alvo 'Foi_Vendido':")
print(df_carros_status['Foi_Vendido'].value_counts(normalize=True))

print("\nGeração concluída com sucesso!")
