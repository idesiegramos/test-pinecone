import streamlit as st
import pinecone
from sentence_transformers import SentenceTransformer

# Inicializar Pinecone
pinecone.init(api_key="TU_API_KEY", environment="TU_ENVIRONMENT")
index_name = "nombre-de-tu-indice"

# Cargar el modelo de embedding
model = SentenceTransformer('all-MiniLM-L6-v2')

def search_documents(query, top_k=5):
    # Convertir la consulta en un vector
    query_vector = model.encode([query])[0].tolist()
    
    # Buscar en Pinecone
    index = pinecone.Index(index_name)
    results = index.query(vector=query_vector, top_k=top_k, include_metadata=True)
    
    return results

# Interfaz de Streamlit
st.title("Buscador de Documentos Similares")

query = st.text_input("Ingresa tu consulta:")

if query:
    results = search_documents(query)
    
    st.subheader("Resultados:")
    for result in results['matches']:
        st.write(f"Documento: {result['metadata']['text']}")
        st.write(f"Puntuación de similitud: {result['score']:.4f}")
        st.write("---")

st.sidebar.header("Acerca de")
st.sidebar.info("Esta aplicación utiliza Pinecone para buscar documentos similares basados en embeddings de texto.")