
import pandas as pd
from fastapi import FastAPI, HTTPException
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

app = FastAPI()

@app.get("/")
def home():
    return {"status": "Servidor funcionando correctamente"}

# 1. CARGA DE DATOS GLOBAL (Para las 7 funciones)
d1 = pd.read_csv('d_limpio.csv')

# 2. OPTIMIZACIÓN ML (Solo para la función 7)
# Reducimos a 5000 para asegurar que corra en cualquier servidor gratuito
df_ml = d1.sort_values('popularity', ascending=False).head(5000).copy()
df_ml['combined'] = df_ml['overview'].fillna('') + " " + df_ml['genres'].fillna('')

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df_ml['combined'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix).astype('float32')

indices = pd.Series(df_ml.index, index=df_ml['title'].str.lower()).drop_duplicates()

# --- FUNCIONES 1 a 6 (Consultas sobre el df completo) ---
# 1. Cantidad de películas por país y año
@app.get('/get_country/{year}/{country}')
def get_country(year: int, country: str):
    # Filtramos por año y buscamos el país en la columna de texto
    mask = (d1['release_year'] == year) & (d1['production_countries'].str.contains(country, case=False, na=False))
    cantidad = int(len(d1[mask]))
    return {
        "pais": country, 
        "anio": year, 
        "cantidad_peliculas": cantidad
    }

# 2. Recaudación por productora y año
@app.get('/get_company_revenue/{company}/{year}')
def get_company_revenue(company: str, year: int):
    # Filtramos por año y buscamos la productora (insensible a mayúsculas)
    mask = (d1['release_year'] == year) & (d1['production_companies'].str.contains(company, case=False, na=False))
    
    # Sumamos la columna revenue de las filas que cumplen el criterio
    total_revenue = int(d1[mask]['revenue'].sum())
    
    return {
        "productora": company,
        "anio": year,
        "recaudacion_total": total_revenue
    }

# 3. Cantidad de películas por año
@app.get('/get_count_movies/{year}')
def get_count_movies(year: int):
    # Filtramos por el año y contamos las filas
    cantidad = int(len(d1[d1['release_year'] == year]))
    
    return {
        "anio": year,
        "total_peliculas": cantidad
    }

# 4. Película con mayor retorno en un año específico
@app.get('/get_return/{year}')
def get_return(year: int):
    # Filtramos por año
    df_year = d1[d1['release_year'] == year]
    
    if df_year.empty:
        return "No hay datos para ese año"
    
    # Buscamos la fila con el valor máximo en la columna 'return'
    pelicula_mayor_retorno = df_year.loc[df_year['return'].idxmax(), 'title']
    
    return pelicula_mayor_retorno

# 5. Película con el menor presupuesto en un año específico
@app.get('/get_min_budget/{year}')
def get_min_budget(year: int):
    # Filtramos por año y nos aseguramos de que el presupuesto sea mayor a 0 
    # (para evitar contar películas sin datos de presupuesto cargados)
    df_year = d1[(d1['release_year'] == year) & (d1['budget'] > 0)]
    
    if df_year.empty:
        return {"mensaje": "No hay datos de presupuesto para ese año"}
    
    # Obtenemos la fila con el presupuesto mínimo
    row = df_year.loc[df_year['budget'].idxmin()]
    
    return {
        'title': str(row['title']),
        'year': int(row['release_year']),
        'budget': float(row['budget'])
    }

# 6. Top 5 franquicias con mayor recaudación histórica
@app.get('/get_collection_revenue')
def get_collection_revenue():
    # Eliminamos las filas que no pertenecen a ninguna colección
    df_collections = d1.dropna(subset=['belongs_to_collection'])
    
    # Agrupamos por colección y sumamos la recaudación (revenue)
    top_5 = df_collections.groupby('belongs_to_collection')['revenue'].sum().sort_values(ascending=False).head(5)
    
    # Devolvemos solo la lista de nombres (el índice del grupo)
    return list(top_5.index)


# --- FUNCIÓN 7: SISTEMA DE RECOMENDACIÓN ---
@app.get('/get_recommendation/{titulo}')
def get_recommendation(titulo: str):
    titulo = titulo.lower()
    if titulo not in indices:
        return {"error": "Película no encontrada en el top de popularidad"}
    
    idx = indices[titulo]
    if isinstance(idx, pd.Series): idx = idx.iloc[0]

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]

    movie_indices = [i[0] for i in sim_scores]
    return {'recomendaciones': df_ml['title'].iloc[movie_indices].tolist()}