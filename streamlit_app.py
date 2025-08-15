# %%
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
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

    fig.add_trace(go.Scatter(x=plot_verificada.index, y=plot_verificada.values,
                             mode='lines+markers',
                             name='Carga Verificada',
                             line=dict(color='blue', width=3),
                             marker=dict(size=4)))

    fig.add_trace(go.Scatter(x=plot_programada.index, y=plot_programada.values,
                             mode='lines+markers',
                             name='Carga Programada',
                             line=dict(color='red', width=3, dash='dash'),
                             marker=dict(size=4)))

    fig.update_layout(
        title={
            'text': f"Carga Global do {regiao} em {dia.strftime('%d/%m/%Y')} ({dia.strftime('%A').capitalize()})",
            'font': {'size': 20},
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title='Data e Hora',
        yaxis_title='Carga (MWMed)',
        legend_title='',
        template='plotly_white'
    )

    st.plotly_chart(fig, use_container_width=True)


def main():
    if "carga_verificada" not in st.session_state or "carga_programada" not in st.session_state:
        st.session_state["carga_verificada"], st.session_state["carga_programada"] = carrega_dados()
    if "dia_selecionado" not in st.session_state:
        st.session_state["dia_selecionado"] = st.session_state["carga_verificada"].index.date[-1]
    if "area_selecionada" not in st.session_state:
        st.session_state["area_selecionada"] = 'SIN'

    col_11, col_12 = st.columns([3, 1])

    with col_12:
        col_12_1, col_12_2, col_12_3 = st.columns([1, 1.2, 1])

        with col_12_1:
            with st.popover("📆 **Data**"):
                st.session_state["dia_selecionado"] = st.date_input(
                    "Selecione o dia para plotagem:",
                    value=st.session_state["carga_verificada"].index.date[-1],
                    min_value=min(pd.unique(st.session_state["carga_verificada"].index.date)),
                    max_value=max(pd.unique(st.session_state["carga_verificada"].index.date))
                )

        with col_12_2:
            with st.popover("📈 **Previsão**"):
                st.write("### Adicionar")


        with col_12_3:
            with st.popover("📍 **Área**"):
                st.session_state["area_selecionada"] = st.selectbox(
                    "Selecione a área para plotagem:",
                    options=st.session_state["carga_verificada"].columns.tolist(),
                    index=0
                )


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
