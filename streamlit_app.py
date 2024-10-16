#import streamlit as st
#import pinecone
#from sentence_transformers import SentenceTransformer

# Inicializar Pinecone
#pinecone.init(api_key="TU_API_KEY", environment="TU_ENVIRONMENT")
#index_name = "nombre-de-tu-indice"

## Cargar el modelo de embedding
#model = SentenceTransformer('all-MiniLM-L6-v2')

#def search_documents(query, top_k=5):
#    # Convertir la consulta en un vector
#    query_vector = model.encode([query])[0].tolist()
    
#    # Buscar en Pinecone
#    index = pinecone.Index(index_name)
#    results = index.query(vector=query_vector, top_k=top_k, include_metadata=True)
    
#    return results

import streamlit as st
import pinecone
import numpy as np
 

OPENAI_API_KEY = st.secrets.api_openai
PINECONE_API_KEY = st.secrets.api_pinecone
LANCHAIN_API_KEY = st.secrets.api_langchain
print("'Secretos' cargados correctamente")



# Configura Pinecone
api_key = PINECONE_API_KEY  # Reemplaza con tu API key
pinecone.init(api_key=api_key, environment="us-east-1")  # Cambia el entorno si es necesario
 
# Crea un nuevo índice
index_name = "mi_index"
if index_name not in pinecone.list_indexes():
    pinecone.create_index(index_name, dimension=128)  # Ajusta la dimensión según tus necesidades
 
# Conéctate al índice
index = pinecone.Index(index_name)
 
# Añade vectores al índice
vectors = np.random.random((5, 128)).astype(np.float32)  # Ejemplo de vectores aleatorios
ids = ["id1", "id2", "id3", "id4", "id5"]
index.upsert(vectors=vectors, ids=ids)
 
# Realiza una búsqueda
query_vector = np.random.random(128).astype(np.float32)  # Vector de consulta aleatorio
response = index.query(queries=[query_vector], top_k=3)  # Ajusta top_k según tus necesidades
print(response)
 
# Borra el índice (opcional)
# pinecone.delete_index(index_name)





#####################################################


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
st.sidebar.info("Hola hola creo que se ha creado la piña 🍍")