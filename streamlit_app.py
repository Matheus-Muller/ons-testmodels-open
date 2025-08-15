# %%
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import requests
import locale
from datetime import datetime

locale.setlocale(locale.LC_TIME, 'Portuguese_Brazil.1252')

# %%

@st.cache_data
def carrega_dados():
    carga_verificada = pd.read_csv("./db/carga_verificada.csv", sep=';', index_col=0, parse_dates=True)
    carga_programada = pd.read_csv("./db/carga_programada.csv", sep=';', index_col=0, parse_dates=True)

    return carga_verificada, carga_programada


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


def plot_carga(carga_verificada, carga_programada, area, dia):
    fig = go.Figure()

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

    fig.update_layout(
        title={
            'text': f"Carga Global do {regiao} em {dia.strftime('%d/%m/%Y')} ({dia.strftime('%A').capitalize()})",
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
    

def main():
    if "carga_verificada" not in st.session_state or "carga_programada" not in st.session_state:
        st.session_state["carga_verificada"], st.session_state["carga_programada"] = carrega_dados()
    if "dia_selecionado" not in st.session_state:
        st.session_state["dia_selecionado"] = st.session_state["carga_verificada"].index.date[-1]
    if "area_selecionada" not in st.session_state:
        st.session_state["area_selecionada"] = 'SIN'

    col_11, col_12 = st.columns([3, 1])

    with col_12:
        col_12_1, col_12_2, col_12_3, col_12_4 = st.columns([1, 1, 1, 1])

        with col_12_1:
            with st.popover("📆"):
                st.session_state["dia_selecionado"] = st.date_input(
                    "Selecione o dia para plotagem:",
                    value=st.session_state["carga_verificada"].index.date[-1],
                    min_value=min(pd.unique(st.session_state["carga_verificada"].index.date)),
                    max_value=max(pd.unique(st.session_state["carga_verificada"].index.date))
                )

        with col_12_2:
            with st.popover("📈"):
                st.write("### Adicionar")


        with col_12_3:
            with st.popover("📍"):
                st.session_state["area_selecionada"] = st.selectbox(
                    "Selecione a área para plotagem:",
                    options=st.session_state["carga_verificada"].columns.tolist(),
                    index=0
                )

        with col_12_4:
            if st.button("🔄"):
                st.session_state["carga_verificada"] = atualiza_verificada(st.session_state["carga_verificada"], URL="https://apicarga.ons.org.br/prd/", areas=['SECO', 'S', 'NE', 'N'], endpoint="cargaverificada")
                st.session_state["dia_selecionado"] = st.session_state["carga_verificada"].index.date[-1]

        with st.container(height=424):
            st.markdown("""
                <p style='text-align: center; margin-bottom: -1px; font-size: 20px;'>
                    <strong>MAPE</strong>
                </p>
                """, unsafe_allow_html=True)
            
            st.write("")

    with col_11:
        with st.container(height=500):
            plot_carga(st.session_state["carga_verificada"], st.session_state["carga_programada"], st.session_state["area_selecionada"], st.session_state["dia_selecionado"])


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
