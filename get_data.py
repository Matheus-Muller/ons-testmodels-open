import pandas as pd
import requests
from datetime import datetime

def extract(URL, area, inicio, fim, endpoint="cargaverificada"):    
    data_inicio = datetime.strptime(inicio, '%Y-%m-%d')
    data_fim = datetime.strptime(fim, '%Y-%m-%d')
    
    dataframes = [] 
    
    inicio_atual = data_inicio
    while inicio_atual <= data_fim:
        fim_atual = min(inicio_atual + pd.Timedelta(days=90), data_fim)

        params = {
            'cod_areacarga': area,
            'dat_inicio': inicio_atual.strftime('%Y-%m-%d'),
            'dat_fim': fim_atual.strftime('%Y-%m-%d')
        }

        url = f"{URL}/{endpoint}"
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            dados = response.json()
            if dados: 
                df = pd.DataFrame(dados)
                dataframes.append(df)
            else:
                print(f"Aviso: Dados não encontrados para o período {params['dat_inicio']} a {params['dat_fim']}.")
        else:
            print(f"Erro na requisição para o período {params['dat_inicio']} a {params['dat_fim']}: {response.status_code} - {response.text}")

        inicio_atual = fim_atual + pd.Timedelta(days=1)

    if dataframes:
        final_df = pd.concat(dataframes, ignore_index=True)
        final_df.drop_duplicates(inplace=True) 
        return final_df
    else:
        return pd.DataFrame()
    
def transform(df, programada=False):
    coluna_carga = 'val_cargaglobal'
    if programada:
        coluna_carga = 'val_cargaglobalprogramada'

    df_transformed = df.copy()
    df_transformed['din_referenciautc'] = pd.to_datetime(df_transformed['din_referenciautc']).dt.tz_localize(None) - pd.Timedelta(hours=3)

    df_transformed.rename(columns={'din_referenciautc': 'datetime'}, inplace=True)

    df_transformed = df_transformed[['datetime', 'cod_areacarga', coluna_carga]]

    return df_transformed

if __name__ == "__main__":
    url = "https://apicarga.ons.org.br/prd/"
    data_inicio = "2024-01-01"
    data_fim = datetime.now().strftime('%Y-%m-%d')

    areas = {'SECO': None, 'S': None, 'NE': None, 'N': None}

    for key in areas:
        areas[key] = extract(URL=url, area=key, inicio=data_inicio, fim=data_fim, endpoint="cargaverificada")
        areas[key] = transform(areas[key])

    carga_verificada = pd.concat([areas['SECO'], areas['S'], areas['NE'], areas['N']])
    carga_verificada = carga_verificada.pivot(index='datetime', columns='cod_areacarga', values='val_cargaglobal')
    carga_verificada = carga_verificada[:datetime.now()].copy()
    carga_verificada['SIN'] = carga_verificada.sum(axis=1)

    for key in areas:
        areas[key] = extract(URL=url, area=key, inicio=data_inicio, fim=data_fim, endpoint="cargaprogramada")
        areas[key] = transform(areas[key], programada=True)

    carga_programada = pd.concat([areas['SECO'], areas['S'], areas['NE'], areas['N']])
    carga_programada = carga_programada.pivot(index='datetime', columns='cod_areacarga', values='val_cargaglobalprogramada')

    carga_programada['SIN'] = carga_programada.sum(axis=1)

    carga_verificada.to_csv('./ignore/carga_verificada.csv', sep=';', index=True)
    carga_programada.to_csv('./ignore/carga_programada.csv', sep=';', index=True)