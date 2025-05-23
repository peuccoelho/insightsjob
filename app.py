import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


@st.cache_data
def carregar_dados():
    df = pd.read_csv("empregos.csv")
    return df

df = carregar_dados()


df['conseguiu_emprego'] = df['conseguiu_emprego'].map({'Sim': 1, 'Não': 0})  
df['participou_programas_treinamento'] = df['participou_programas_treinamento'].map({'Sim': 1, 'Não': 0}) 
df['possui_estagio'] = df['possui_estagio'].map({'Sim': 1, 'Não': 0})  
df['sentiu_preparado'] = df['sentiu_preparado'].map({'Sim': 1, 'Não': 0})  
df['tem_cursos_extras'] = df['cursos_extras'].notna().astype(int) 


bins = [17, 24, 30, 35, 40, 100]
labels = ['18-24', '25-30', '31-35', '36-40', '41+']
df['faixa_idade'] = pd.cut(df['idade'], bins=bins, labels=labels)

# Sidebar
st.sidebar.title("Painel de Insights")
opcao = st.sidebar.selectbox("Escolha um insight:", [
    "Distribuição de Idade (Pizza)",
    "Proporção por Gênero",
    "Estados com Mais Candidatos",
    "Tempo Médio até Emprego por Tipo de Empresa",
    "Tipo de Contrato Mais Comum",
    "Previsão de Emprego"
])

st.title("Análise de Dados de Empregos na Área de TI")

if opcao == "Distribuição de Idade (Pizza)":
    st.subheader("Distribuição de Idade dos Candidatos (por Faixa Etária)")
    idade_counts = df['faixa_idade'].value_counts().reindex(labels, fill_value=0)
    fig, ax = plt.subplots()
    ax.pie(idade_counts, labels=idade_counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette("pastel"))
    ax.axis('equal')
    st.pyplot(fig)
    st.markdown("**Frequência** representa a proporção de candidatos em cada faixa etária.")

elif opcao == "Proporção por Gênero":
    st.subheader("Proporção por Gênero")
    genero_counts = df['genero'].value_counts()
    fig, ax = plt.subplots()
    ax.pie(genero_counts, labels=genero_counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette("pastel"))
    ax.axis('equal')
    st.pyplot(fig)

elif opcao == "Estados com Mais Candidatos":
    st.subheader("Estados com Mais Candidatos")
    estado_counts = df['estado'].value_counts().head(10)
    fig, ax = plt.subplots()
    sns.barplot(x=estado_counts.values, y=estado_counts.index, palette="Blues_d", ax=ax)
    ax.set_xlabel("Número de Candidatos")
    ax.set_ylabel("Estado")
    st.pyplot(fig)

elif opcao == "Tempo Médio até Emprego por Tipo de Empresa":
    st.subheader("Tempo Médio até Conseguir Emprego por Tipo de Empresa")
    tempo_medio = df.groupby("tipo_empresa")["tempo_ate_emprego"].mean().sort_values()
    fig, ax = plt.subplots()
    sns.barplot(x=tempo_medio.values, y=tempo_medio.index, palette="viridis", ax=ax)
    ax.set_xlabel("Meses até conseguir emprego")
    ax.set_ylabel("Tipo de Empresa")
    st.pyplot(fig)

elif opcao == "Tipo de Contrato Mais Comum":
    st.subheader("Tipo de Contrato Mais Comum")
    contrato_counts = df['tipo_contrato'].value_counts()
    fig, ax = plt.subplots()
    sns.barplot(x=contrato_counts.index, y=contrato_counts.values, palette="Set2", ax=ax)
    ax.set_ylabel("Número de Candidatos")
    ax.set_xlabel("Tipo de Contrato")
    st.pyplot(fig)

elif opcao == "Previsão de Emprego":
    st.subheader("Previsão de Conquista de Emprego")

    features = ["idade", "possui_estagio", "sentiu_preparado", "participou_programas_treinamento",
                "contatos_profissionais", "trabalhos_freelancer", "feedbacks_recebidos", "tem_cursos_extras"]

    dados_validos = df[features + ["conseguiu_emprego"]].dropna()

    X = dados_validos[features]
    y = dados_validos["conseguiu_emprego"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    modelo = RandomForestClassifier(n_estimators=100, random_state=42)
    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    st.write(f"Acurácia do modelo: **{acc:.2f}**")

    st.markdown("### Faça uma simulação:")
    idade = st.slider("Idade", 18, 60, 25)
    estagio = st.selectbox("Possui experiência de estágio?", ["Sim", "Não"])
    preparado = st.selectbox("Sentiu-se preparado?", ["Sim", "Não"])
    treinamento = st.selectbox("Participou de programas de treinamento?", ["Sim", "Não"])
    contatos = st.slider("Contatos profissionais feitos", 0, 50, 5)
    freelancers = st.slider("Trabalhos freelancer realizados", 0, 20, 2)
    feedbacks = st.slider("Feedbacks recebidos", 0, 20, 3)
    cursos = st.selectbox("Possui cursos extras?", ["Sim", "Não"])

    entrada = pd.DataFrame({
        "idade": [idade],
        "possui_estagio": [1 if estagio == "Sim" else 0],
        "sentiu_preparado": [1 if preparado == "Sim" else 0],
        "participou_programas_treinamento": [1 if treinamento == "Sim" else 0],
        "contatos_profissionais": [contatos],
        "trabalhos_freelancer": [freelancers],
        "feedbacks_recebidos": [feedbacks],
        "tem_cursos_extras": [1 if cursos == "Sim" else 0]
    })

    pred = modelo.predict(entrada)[0]
    st.success("✅ Provavelmente conseguirá um emprego!" if pred == 1 else "❌ Pode ter dificuldades para conseguir emprego.")
