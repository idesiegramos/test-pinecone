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
from pinecone import Pinecone, ServerlessSpec
import numpy as np


OPENAI_API_KEY = st.secrets.api_openai
PINECONE_API_KEY = st.secrets.api_pinecone
LANCHAIN_API_KEY = st.secrets.api_langchain
print("'Secretos' cargados correctamente")



# Configura Pinecone
pc = Pinecone(api_key = PINECONE_API_KEY)

# Now do stuff
if 'my_index' not in pc.list_indexes().names():
    pc.create_index(
        name='my_index', 
        dimension=1536, 
        metric='euclidean',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
            )
        )

index_name = 'my_index'
# Conéctate al índice
index = pc.Index(index_name)
 
# Añade vectores al índice
vectors = np.random.random((5, 128)).astype(np.float32)  # Ejemplo de vectores aleatorios
ids = ["id1", "id2", "id3", "id4", "id5"]
index.upsert(vectors=vectors, ids=ids)
 
# Realiza una búsqueda
query_vector = np.random.random(128).astype(np.float32)  # Vector de consulta aleatorio
response = index.query(queries=[query_vector], top_k=3)  # Ajusta top_k según tus necesidades
print(response)


#########3



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