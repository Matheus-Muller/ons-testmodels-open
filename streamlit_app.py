import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import locale
from datetime import datetime

locale.setlocale(locale.LC_TIME, 'Portuguese_Brazil.1252')


@st.cache_data
def carrega_dados():
    carga_verificada = pd.read_csv("./db/carga_verificada.csv", sep=';', index_col=0, parse_dates=True)
    carga_programada = pd.read_csv("./db/carga_programada.csv", sep=';', index_col=0, parse_dates=True)

    return carga_verificada, carga_programada


@st.cache_data
def set_feriados():
    feriados = {
        # 2024
        "2024-01-01": "Confraternização Universal",
        "2024-02-12": "Carnaval",
        "2024-02-13": "Carnaval",
        "2024-02-14": "Carnaval",
        "2024-03-29": "Sexta-feira Santa",
        "2024-04-21": "Tiradentes",
        "2024-05-01": "Dia do Trabalho",
        "2024-05-30": "Corpus Christi",
        "2024-05-31": "Ponte",
        "2024-09-07": "Independência do Brasil",
        "2024-10-12": "Nossa Senhora Aparecida",
        "2024-11-02": "Finados",
        "2024-11-15": "Proclamação da República",
        "2024-12-25": "Natal",

        # 2025
        "2025-01-01": "Confraternização Universal",
        "2025-03-03": "Carnaval",
        "2025-03-04": "Carnaval",
        "2025-03-05": "Carnaval",
        "2025-04-18": "Sexta-feira Santa",
        "2025-04-21": "Tiradentes",
        "2025-05-01": "Dia do Trabalho",
        "2025-05-02": "Ponte",
        "2025-06-19": "Corpus Christi",
        "2025-06-20": "Ponte",
        "2025-09-07": "Independência do Brasil",
        "2025-10-12": "Nossa Senhora Aparecida",
        "2025-11-02": "Finados",
        "2025-11-15": "Proclamação da República",
        "2025-12-25": "Natal",
        "2025-12-26": "Ponte"
    }

    feriados = {pd.to_datetime(k).date(): v for k, v in feriados.items()}

    return feriados


def imprime_feriados(feriados, ano):
    feriados_filtrados = {date: name for date, name in feriados.items() if date.year == ano}

    if not feriados_filtrados:
        st.info("Nenhum feriado disponível para o ano selecionado.")
        return

    feriados_df = pd.DataFrame(list(feriados_filtrados.items()), columns=['Data', 'Feriado'])
    feriados_df['Data'] = feriados_df['Data'].apply(lambda x: x.strftime('%d/%m/%Y (%A)'))

    st.table(feriados_df)


def nome_regiao(area):
    if area == 'N':
        return 'Norte'
    elif area == 'NE':
        return 'Nordeste'
    elif area == 'SECO':
        return 'Sudeste/Centro-Oeste'
    elif area == 'S':
        return 'Sul'
    else:
        return 'SIN'


def plot_carga(carga_verificada, carga_programada, carga_prevista, area, dia, feriados):
    fig = go.Figure()

    feriado = feriados.get(dia)

    if feriado is not None and feriado == "Ponte":
        titulo_feriado = f"- Ponte"
    elif feriado is not None:
        titulo_feriado = f"- Feriado"
    else:
        titulo_feriado = ""

    regiao = nome_regiao(area)

    plot_verificada = carga_verificada[carga_verificada.index.date == dia][area]
    plot_programada = carga_programada[carga_programada.index.date == dia][area]

    fig.add_trace(go.Scatter(
        x=plot_verificada.index, y=plot_verificada.values,
        mode='lines+markers',
        name='Carga Verificada',
        line=dict(color='blue', width=3),
        marker=dict(size=4)
    ))

    fig.add_trace(go.Scatter(
        x=plot_programada.index, y=plot_programada.values,
        mode='lines+markers',
        name='Carga Programada',
        line=dict(color='green', width=3, dash='dash'),
        marker=dict(size=4)
    ))

    if not carga_prevista.empty:
        plot_prevista = carga_prevista[(carga_prevista.index.date == dia) & (carga_prevista.area == area)]
        for modelo in plot_prevista['modelo'].unique():
            plot_modelo = plot_prevista[plot_prevista['modelo'] == modelo]
            if not plot_modelo.empty:
                fig.add_trace(go.Scatter(
                    x=plot_modelo.index, y=plot_modelo['val_cargaprevista'],
                    mode='lines+markers',
                    name=f'{modelo}',
                    line=dict(width=2),
                    marker=dict(size=4)
                ))

    fig.update_layout(
        title={
            'text': f"Carga Global do {regiao} em {dia.strftime('%d/%m/%Y')} ({dia.strftime('%A').capitalize()}) {titulo_feriado}",
            'font': {'size': 20},
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title='Hora',
        yaxis_title='Carga (MWMed)',
        legend_title='',
        template='plotly_white'
    )

    st.plotly_chart(fig, use_container_width=True)


def plot_mape(carga_verificada, carga_programada, carga_prevista, area, dia):
    verificada = carga_verificada[carga_verificada.index.date == dia][area]
    programada = carga_programada.loc[verificada.index, area]

    y_true, y_pred = verificada.values, programada.values

    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    st.metric(
        label=f"**Programada**",
        value=f"{mape:.2f}"
    )

    mapes_prevista = {'modelo': [], 'mape': []}

    if not carga_prevista.empty:
        if area in carga_prevista['area'].values:
            prevista = carga_prevista.loc[(carga_prevista.index.isin(verificada.index)) & (carga_prevista['area'] == area), ['modelo', 'val_cargaprevista']]

            for modelo in prevista['modelo'].unique():
                y_prev = prevista[prevista['modelo'] == modelo]['val_cargaprevista'].values
                if len(y_prev) == 0:
                    continue

                mape_prev = np.mean(np.abs((y_true - y_prev) / y_true)) * 100

                mapes_prevista['modelo'].append(modelo)
                mapes_prevista['mape'].append(mape_prev)

    if mapes_prevista:
        for modelo, mape in zip(mapes_prevista['modelo'], mapes_prevista['mape']):
            st.metric(
                label=f"**{modelo}**",
                value=f"{mape:.2f}"
            )


def atualiza_verificada(carga_verificada, URL="https://apicarga.ons.org.br/prd/", areas=['N', 'NE', 'S', 'SECO'], endpoint="cargaverificada"):
    inicio = carga_verificada.index[-1] + pd.Timedelta(minutes=30)
    fim = datetime.now()

    all_area_dfs = []

    for area in areas:        
        data_inicio = datetime.strptime(inicio.strftime('%Y-%m-%d'), '%Y-%m-%d')
        data_fim = datetime.strptime(fim.strftime('%Y-%m-%d'), '%Y-%m-%d')
        
        dataframes_for_current_area = [] 
        
        inicio_atual = data_inicio
        
        while inicio_atual <= data_fim:
            fim_atual = min(inicio_atual + pd.Timedelta(days=90), data_fim)

            params = {
                'cod_areacarga': area,
                'dat_inicio': inicio_atual.strftime('%Y-%m-%d'),
                'dat_fim': fim_atual.strftime('%Y-%m-%d')
            }

            url = f"{URL}/{endpoint}"
            
            try:
                response = requests.get(url, params=params, timeout=30)
                response.raise_for_status() 
                
                dados = response.json()
                if dados: 
                    df_chunk = pd.DataFrame(dados)
                    dataframes_for_current_area.append(df_chunk)
            
            except requests.exceptions.RequestException as e:
                print(f"  Erro na requisição para a área '{area}': {e}")

            inicio_atual = fim_atual + pd.Timedelta(days=1)


        if dataframes_for_current_area:
            df_area = pd.concat(dataframes_for_current_area, ignore_index=True)
            df_area.drop_duplicates(inplace=True) 
            
            df_area['cod_areacarga_ref'] = area
            
            all_area_dfs.append(df_area)

    if all_area_dfs:
        full_df = pd.concat(all_area_dfs, ignore_index=True)

        full_df['din_referenciautc'] = pd.to_datetime(full_df['din_referenciautc']).dt.tz_localize(None) - pd.Timedelta(hours=3)

        full_df.rename(columns={'din_referenciautc': 'datetime'}, inplace=True)

        full_df = full_df[['datetime', 'cod_areacarga', 'val_cargaglobal']]

        full_df = full_df.pivot(index='datetime', columns='cod_areacarga', values='val_cargaglobal')
        full_df['SIN'] = full_df.sum(axis=1)
        full_df = full_df[(full_df.index >= inicio) & (full_df['SIN'] != 0)].copy()

        df_final = pd.concat([carga_verificada, full_df], axis=0)

        return df_final
    else:
        return carga_verificada
    

def atualiza_programada(carga_programada, URL="https://apicarga.ons.org.br/prd/", areas=['N', 'NE', 'S', 'SECO'], endpoint="cargaprogramada"):
    inicio = carga_programada.index[-1] + pd.Timedelta(minutes=30)
    fim = datetime.now() + pd.Timedelta(days=1)

    if carga_programada.index.date[-1] == (fim + pd.Timedelta(days=1)).date():
        return carga_programada
    else:
        all_area_dfs = []

        for area in areas:        
            data_inicio = datetime.strptime(inicio.strftime('%Y-%m-%d'), '%Y-%m-%d')
            data_fim = datetime.strptime(fim.strftime('%Y-%m-%d'), '%Y-%m-%d')
            
            dataframes_for_current_area = [] 
            
            inicio_atual = data_inicio
            
            while inicio_atual <= data_fim:
                fim_atual = min(inicio_atual + pd.Timedelta(days=90), data_fim)

                params = {
                    'cod_areacarga': area,
                    'dat_inicio': inicio_atual.strftime('%Y-%m-%d'),
                    'dat_fim': fim_atual.strftime('%Y-%m-%d')
                }

                url = f"{URL}/{endpoint}"
                
                try:
                    response = requests.get(url, params=params, timeout=30)
                    response.raise_for_status() 
                    
                    dados = response.json()
                    if dados: 
                        df_chunk = pd.DataFrame(dados)
                        dataframes_for_current_area.append(df_chunk)
                
                except requests.exceptions.RequestException as e:
                    st.error(f"Erro na requisição para a área '{area}': {e}")

                inicio_atual = fim_atual + pd.Timedelta(days=1)


            if dataframes_for_current_area:
                df_area = pd.concat(dataframes_for_current_area, ignore_index=True)
                df_area.drop_duplicates(inplace=True) 
                
                df_area['cod_areacarga_ref'] = area
                
                all_area_dfs.append(df_area)

        if all_area_dfs:
            full_df = pd.concat(all_area_dfs, ignore_index=True)

            full_df['din_referenciautc'] = pd.to_datetime(full_df['din_referenciautc']).dt.tz_localize(None) - pd.Timedelta(hours=3)

            full_df.rename(columns={'din_referenciautc': 'datetime'}, inplace=True)

            full_df = full_df[['datetime', 'cod_areacarga', 'val_cargaglobalprogramada']]

            full_df = full_df.pivot(index='datetime', columns='cod_areacarga', values='val_cargaglobalprogramada')
            full_df['SIN'] = full_df.sum(axis=1)
            full_df = full_df[(full_df.index >= inicio) & (full_df['SIN'] != 0)].copy()

            df_final = pd.concat([carga_programada, full_df], axis=0)

            return df_final
        else:
            return carga_programada


def atualizar_dados_e_data():
    st.session_state["carga_verificada"] = atualiza_verificada(st.session_state["carga_verificada"])
    st.session_state["carga_programada"] = atualiza_programada(st.session_state["carga_programada"])
    st.session_state["dia_selecionado"] = st.session_state["carga_verificada"].index.date[-1]


def adiciona_previsao(df, uploaded_file, modelo_nome, area, verifica_freq=True):
    df_previsao = pd.read_csv(uploaded_file, sep=';')

    datetime_col = df_previsao.columns[0]
    previsao_col = df_previsao.columns[1]

    df_previsao = df_previsao[[datetime_col, previsao_col]].copy()

    try:
        df_previsao.index = pd.to_datetime(df_previsao[datetime_col]) 
        df_previsao.drop(columns=datetime_col, inplace=True)
    except Exception as e:
        st.error(f"Erro ao converter a coluna '{datetime_col}' para datetime: {e}")
        return df
    
    if df_previsao[previsao_col].dtype not in [float, int]:
        try:
            df_previsao[previsao_col] = df_previsao[previsao_col].str.replace(',', '.').astype(float)
        except Exception as e:
            st.error(f"A coluna '{previsao_col}' deve conter valores numéricos. Erro ao converter: {e}")
            return df
        

    freq = pd.infer_freq(df_previsao[df_previsao.index.date == df_previsao.index.date[0]].index)
    
    if freq is None and verifica_freq:
        st.warning("Frequência não detectada automaticamente. Verifique o intervalo dos dados.")
        return df
    
    if freq == 'h' or freq == 'H':
        dfs_semihora_por_dia = []

        for dia in df_previsao.index.normalize().unique():
            df_dia_horario = df_previsao[df_previsao.index.date == dia.date()]

            datetime_index_dia = pd.date_range(
                start=df_dia_horario.index.min(),
                end=df_dia_horario.index.max(),
                freq='30min'
            )

            df_dia_semihora = df_dia_horario.reindex(datetime_index_dia)
            
            try:
                df_dia_semihora[previsao_col] = df_dia_semihora[previsao_col].interpolate(method='linear')
                
                df_previsao_last_row = df_dia_semihora.iloc[[-1]].copy()
                df_previsao_last_row[previsao_col] *= 0.95 
                df_previsao_last_row.index += pd.Timedelta(minutes=30)
                
                df_dia_final = pd.concat([df_dia_semihora, df_previsao_last_row])

                dfs_semihora_por_dia.append(df_dia_final)

            except Exception as e:
                st.error(f"Erro na interpolação para o dia {dia.date()}: {e}")
                return df
        
        if dfs_semihora_por_dia:
            df_previsao = pd.concat(dfs_semihora_por_dia)
        else:
            st.error("Não foi possível processar nenhum dia para a conversão semi-horária.")
            return df

    df_previsao['modelo'] = modelo_nome
    df_previsao['area'] = area

    df_previsao = df_previsao.rename(columns={previsao_col: 'val_cargaprevista'})

    intersecao = pd.DataFrame()

    if not df.empty:
        existe = df[(df['modelo'] == modelo_nome) & (df['area'] == area)]
        intersecao = existe.index.intersection(df_previsao.index)

    if not intersecao.empty:
        st.warning(f"Já existem {len(intersecao)} registros do modelo '{modelo_nome}' para esses horários. Eles serão sobrescritos.")

        mask = (
            df.index.isin(intersecao) & 
            (df['modelo'] == modelo_nome) & 
            (df['area'] == area)
        )

        df = df[~mask]

    df = pd.concat([df, df_previsao], ignore_index=False)

    df['datetime'] = df.index
    df = df.sort_values(by=['datetime', 'modelo']).drop(columns='datetime')

    st.success(f"Previsões do modelo '{modelo_nome}' adicionadas com sucesso!")

    return df


def remove_previsao(df, modelo_nome, area):
    df = df.copy()

    mask = (df['modelo'] == modelo_nome) & (df['area'] == area)

    if df[mask].empty:
        st.warning(f"Nenhuma previsão encontrada para o modelo '{modelo_nome}'.")
        return df

    df = df[~mask]

    df['datetime'] = df.index
    df = df.sort_values(by=['datetime', 'modelo']).drop(columns='datetime')

    st.success(f"Previsões do modelo '{modelo_nome}' removidas com sucesso para a área '{area}'!")

    return df


def main():
    if "carga_verificada" not in st.session_state or "carga_programada" not in st.session_state:
        st.session_state["carga_verificada"], st.session_state["carga_programada"] = carrega_dados()
    if "carga_prevista" not in st.session_state:
        st.session_state["carga_prevista"] = pd.DataFrame()
    if "dia_selecionado" not in st.session_state:
        st.session_state["dia_selecionado"] = st.session_state["carga_verificada"].index.date[-1]
    if "area_selecionada" not in st.session_state:
        st.session_state["area_selecionada"] = 'SIN'
    if "feriados" not in st.session_state:
        st.session_state["feriados"] = set_feriados()

    col_11, col_12 = st.columns([3, 1])

    with col_12:
        col_12_1, col_12_2, col_12_3, col_12_4, col_12_5 = st.columns([1, 1, 1, 1, 0.9])

        with col_12_1:
            with st.popover("📆"):
                st.date_input(
                    "Selecione o dia para plotagem:",
                    key="dia_selecionado",
                    min_value=min(pd.unique(st.session_state["carga_verificada"].index.date)),
                    max_value=max(pd.unique(st.session_state["carga_verificada"].index.date)),
                )

        with col_12_2:
            with st.popover("📈"):
                st.write("### Adicionar")
                uploaded_file = st.file_uploader("Selecione o arquivo CSV com as previsões", type="csv")
                modelo_nome = st.text_input("Informe o nome do modelo")
                area_nome = st.text_input("Informe a área do modelo")
                opcao_verifica_freq = st.checkbox("Verificar frequência dos dados", value=True, help="Marque esta opção se os dados não estiverem em formato semihorário")
                add_previsao = st.button("Adicionar Previsão")

                if add_previsao:
                    if uploaded_file is not None and modelo_nome and area_nome:
                        st.session_state["carga_prevista"] = adiciona_previsao(
                            st.session_state["carga_prevista"], uploaded_file, modelo_nome, area_nome, opcao_verifica_freq
                        )

                        st.rerun()
                    else:
                        st.warning("Por favor, selecione um arquivo CSV e informe o nome do modelo.")


                st.divider()

                st.write("### Remover")

                if not st.session_state["carga_prevista"].empty:
                    modelo_nome_remover = st.selectbox("Modelo a ser removido", options=st.session_state["carga_prevista"]['modelo'].unique().tolist())
                    remover_area = st.selectbox("Área a ser removida", options=st.session_state["carga_prevista"][st.session_state["carga_prevista"]['modelo'] == modelo_nome_remover]['area'].unique().tolist())
                    
                    remover_previsao = st.button("Remover Previsão")

                    if remover_previsao:
                        if modelo_nome_remover is not None and remover_area is not None:
                            st.session_state["carga_prevista"] = remove_previsao(st.session_state["carga_prevista"], modelo_nome_remover, remover_area)
                            st.success(f"Previsões do modelo '{modelo_nome_remover}' removidas com sucesso para a área '{remover_area}'!")
                            st.rerun()
                        else:
                            st.warning("Por favor, informe o nome do modelo a ser removido.")

        with col_12_3:
            with st.popover(f"📍"):
                st.selectbox(
                    "Selecione a área para plotagem:",
                    options=st.session_state["carga_verificada"].columns.tolist(),
                    key="area_selecionada",
                )

        with col_12_4:
            with st.popover("🎉"):
                st.write("### Feriados")

                ano_feriados = st.selectbox(
                    "Selecione o ano para exibir os feriados:",
                    options=st.session_state["carga_verificada"].index.year.unique().tolist(),
                    index=len(st.session_state["carga_verificada"].index.year.unique().tolist()) - 1
                )

                imprime_feriados(st.session_state["feriados"], ano_feriados)


        with col_12_5:
            st.button("🔄", on_click=atualizar_dados_e_data, help='Atualizar')
            

        with st.container(height=444):
            _, col_mape, _ = st.columns([1, 1.5, 1])

            with col_mape:
                st.markdown("""
                    <p style='font-size: 20px;'>
                        <strong>MAPE (%)</strong>
                    </p>
                    """, unsafe_allow_html=True)

                plot_mape(st.session_state["carga_verificada"], st.session_state["carga_programada"], st.session_state["carga_prevista"], st.session_state["area_selecionada"], st.session_state["dia_selecionado"])

    with col_11:
        with st.container(height=500):
            plot_carga(st.session_state["carga_verificada"], st.session_state["carga_programada"], st.session_state["carga_prevista"], st.session_state["area_selecionada"], st.session_state["dia_selecionado"], st.session_state["feriados"])


def relatorio():
    st.write("### Página de Relatório em construção ... ⌛")


if __name__ == "__main__":
    st.set_page_config(
        page_title="ONS Test Models",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    st.markdown(
        f"""<div style="background-color: #312B2B; padding: 1px; border-radius: 10px;">
        <h3 style='text-align: center; color: #FFFFFF;'>ONS Test Models - Benchmarking de Modelos</h3>
        </div>
        """, 
        unsafe_allow_html=True  
    )

    st.write(" ")

    with st.sidebar:
        st.write("# ONS Test Models")
        st.divider()

        navigation = st.selectbox(
            "Menu",
            options=["Dashboard", "Relatório"],
            index=0
        )

    if navigation == "Dashboard":
        main()

    elif navigation == "Relatório":
        relatorio()

    else:
        st.error("Opção inválida. Por favor, escolha 'Dashboard' ou 'Relatório'.")
