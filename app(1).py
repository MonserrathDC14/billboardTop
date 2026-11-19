import streamlit as st
import os
import pandas as pd
import numpy as np
import joblib
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from typing import Optional

# Configuración de página de Streamlit
st.set_page_config(
    page_title="Billboard Top 1 Predictor",
    page_icon="🎵",
    layout="centered"
)

# Estructura de datos requerida por el modelo + URL del Preview
class SongInfoExtended(BaseModel):
    chart_date: int = Field(description="Estimated release or chart date in YYYYMMDD format. Must be an 8-digit integer.")
    durationSeg: float = Field(description="Duration of the song in seconds (typically between 100.0 and 500.0).")
    anioNacimiento: float = Field(description="Year of birth of the artist or the main band member (numeric float, e.g. 1990.0).")
    mood: str = Field(description="The main mood of the song. Choose from: Empowering, Cool, Energizing, Yearning, Defiant, Excited, Sensual, Urgent, Sophisticated, Rowdy, Brooding, Upbeat, Aggressive, Melancholy, Gritty, Romantic, Sentimental, Fiery, Easygoing, Peaceful, Stirring, Lively, Other.")
    tempo: str = Field(description="Tempo of the song. Must be one of: 'Medium Tempo', 'Slow Tempo', 'Fast Tempo'.")
    genre: str = Field(description="Music genre. Must be one of: Urban, Pop, Traditional, Alternative & Punk, Electronica, Rock, Other, Soundtrack, Jazz.")
    artist_type: str = Field(description="Type of artist. Must be one of: Male, Female, Mixed.")
    preview_url: Optional[str] = Field(description="A direct public URL to an audio preview (.mp3 format) of the song if available (for example, from iTunes API or Spotify preview CDN). If not found, leave it as null.")

st.title("🎵 Predictor Billboard Top 1 & Buscador Inteligente")
st.write("Usa este agente inteligente para buscar información de cualquier canción, escuchar un preview de audio y predecir si será un éxito número uno en Billboard.")

# Configuración segura de la clave de API
st.sidebar.header("🔑 Autenticación")
api_key_input = st.sidebar.text_input("Gemini API Key", type="password", help="Si dejas esto en blanco, se intentará usar la variable de entorno o secretos.")

# Columnas del modelo para asegurar el One-Hot Encoding correcto
columnas_entrenamiento = [
    'chart_date', 'durationSeg', 'anioNacimiento', 'mood_Aggressive',
    'mood_Brooding', 'mood_Cool', 'mood_Defiant', 'mood_Easygoing',
    'mood_Empowering', 'mood_Energizing', 'mood_Excited', 'mood_Fiery',
    'mood_Gritty', 'mood_Lively', 'mood_Melancholy', 'mood_Other',
    'mood_Peaceful', 'mood_Romantic', 'mood_Rowdy', 'mood_Sensual',
    'mood_Sentimental', 'mood_Sophisticated', 'mood_Stirring', 'mood_Upbeat',
    'mood_Urgent', 'mood_Yearning', 'tempo_Fast Tempo', 'tempo_Medium Tempo',
    'tempo_Slow Tempo', 'genre_Alternative & Punk', 'genre_Electronica',
    'genre_Jazz', 'genre_Other', 'genre_Pop', 'genre_Rock', 'genre_Soundtrack',
    'genre_Traditional', 'genre_Urban', 'artist_type_Female',
    'artist_type_Male', 'artist_type_Mixed'
]

cancion = st.text_input("Nombre de la Canción", placeholder="Ej: Flowers")
artista = st.text_input("Artista o Banda", placeholder="Ej: Miley Cyrus")

if st.button("🔍 Buscar y Analizar"):
    if not cancion or not artista:
        st.warning("Por favor ingresa tanto el nombre de la canción como el artista.")
    else:
        # Determinar clave API
        api_key = api_key_input if api_key_input else os.getenv("GEMINI_API_KEY")
        
        if not api_key:
            # Intentar obtener de Colab userdata si está en el backend de Colab
            try:
                from google.colab import userdata
                api_key = userdata.get('GEMINI_API_KEY')
            except:
                pass
                
        if not api_key:
            st.error("❌ No se encontró ninguna API Key de Gemini. Configúrala en la barra lateral.")
        else:
            with st.spinner("🤖 Agente de IA recopilando datos de la canción y buscando audio de preview..."):
                try:
                    client = genai.Client(api_key=api_key)
                    prompt = f"""
                    Busca información fidedigna y real de la canción '{cancion}' del artista '{artista}'.
                    Encuentra y extrae los siguientes datos:
                    - chart_date: Fecha de lanzamiento en formato numérico YYYYMMDD.
                    - durationSeg: Duración de la canción en segundos.
                    - anioNacimiento: Año de nacimiento del artista o líder.
                    - mood: El mood que más represente a la canción de la lista.
                    - tempo: Su velocidad o tempo ('Medium Tempo', 'Slow Tempo', 'Fast Tempo').
                    - genre: El género correspondiente de la lista.
                    - artist_type: Clasificación del artista.
                    - preview_url: Una URL pública directa de audio mp3 para escuchar una vista previa de la canción.
                    """
                    
                    response = client.models.generate_content(
                        model='gemini-3.5-flash-lite',
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            response_schema=SongInfoExtended,
                            temperature=0.1
                        ),
                    )
                    
                    datos_cancion = SongInfoExtended.model_validate_json(response.text)
                except Exception as e:
                    st.error(f"Ocurrió un error al buscar con el Agente de IA: {e}")
                    st.stop()

            # Vista de información de la canción
            st.subheader("📊 Características Encontradas")
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Género:** {datos_cancion.genre}")
                st.write(f"**Tempo:** {datos_cancion.tempo}")
                st.write(f"**Mood:** {datos_cancion.mood}")
            with col2:
                st.write(f"**Duración:** {datos_cancion.durationSeg} segundos")
                st.write(f"**Año Nacimiento Artista:** {int(datos_cancion.anioNacimiento)}")
                st.write(f"**Fecha Estimada Lanzamiento:** {datos_cancion.chart_date}")

            # Reproductor de Preview de Audio
            st.markdown("### 🎧 Preview de la Canción")
            if datos_cancion.preview_url:
                st.audio(datos_cancion.preview_url)
            else:
                st.info("No se pudo encontrar un preview de audio directo para esta canción, pero puedes escucharla en tu plataforma favorita.")

            # Cargar Modelo y predecir
            try:
                modelo = joblib.load('modelo_random_forest.joblib')
                
                # Construir el vector de características en base a lo retornado
                # Excluimos 'preview_url' que no va en el modelo
                datos_modelo = datos_cancion.model_dump()
                datos_modelo.pop('preview_url', None)
                
                df_nuevo = pd.DataFrame([datos_modelo])
                df_nuevo_encoded = pd.get_dummies(df_nuevo, columns=['mood', 'tempo', 'genre', 'artist_type'], dtype=int)
                
                # Alinear columnas con el modelo
                df_predict = pd.DataFrame(0, index=np.arange(len(df_nuevo_encoded)), columns=columnas_entrenamiento)
                for col in df_nuevo_encoded.columns:
                    if col in df_predict.columns:
                        df_predict[col] = df_nuevo_encoded[col]
                
                prediccion = modelo.predict(df_predict)[0]
                probabilidades = modelo.predict_proba(df_predict)[0]
                
                st.markdown("--- ")
                st.subheader("🔮 Predicción de Éxito en Billboard")
                prob_entrar = probabilidades[1] * 100
                
                if prediccion == 1:
                    st.balloons()
                    st.success(f"### ¡Éxito Asegurado! 🎉 La canción tiene un {prob_entrar:.1f}% de probabilidades de entrar al Top 1 de Billboard.")
                else:
                    st.warning(f"### Es poco probable que alcance el número 1. La probabilidad de entrar es de {prob_entrar:.1f}%.")
                    
                st.progress(int(prob_entrar))
            except Exception as e:
                st.error(f"Error al realizar la predicción con el modelo: {e}")
