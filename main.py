from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# Inicializar FastAPIs
app = FastAPI()

# Cargar el modelo y el escalador
model = tf.keras.models.load_model('viable_lstm_model_extended.h5')

# Simular el ajuste del escalador con datos de entrenamiento (puedes cambiarlo si tienes un escalador preajustado)
np.random.seed(42)
num_samples = 10000
X_simulated = pd.DataFrame({
    'MONTO_VIABLE': np.random.uniform(1000, 50000, num_samples),
    'COSTO_ACTUALIZADO': np.random.uniform(2000, 60000, num_samples),
    'SITUACION': np.random.choice([0, 1], num_samples),
    'ESTADO': np.random.choice([0, 1], num_samples),
    'NIVEL': np.random.choice([0, 1, 2], num_samples),
    'FUNCION': np.random.choice([0, 1, 2], num_samples),
    'SECTOR': np.random.choice([0, 1, 2], num_samples)
})
scaler = StandardScaler()
scaler.fit(X_simulated)

# Base de datos en memoria (arreglo de proyectos)
projects = []

# Modelo de datos para entrada
class Project(BaseModel):
    codigo_unico: str
    nombre_proyecto: str
    responsable: str
    fecha_inicio: str
    departamento: str
    distrito: str
    objetivo: str
    monto_viable: float
    costo_actualizado: float
    situacion: int
    estado: int
    nivel: int
    funcion: int
    sector: int

# Ruta de bienvenida
@app.get("/")
def welcome():
    return {"message": "Hola mundo"}

# Crear un proyecto
@app.post("/projects/")
def create_project(project: Project):
    try:
        # Validar los datos de entrada
        if not (1000 <= project.monto_viable <= 50000):
            raise HTTPException(status_code=400, detail="MONTO_VIABLE debe estar entre 1000 y 50000.")
        if not (2000 <= project.costo_actualizado <= 60000):
            raise HTTPException(status_code=400, detail="COSTO_ACTUALIZADO debe estar entre 2000 y 60000.")
        if project.situacion not in [0, 1]:
            raise HTTPException(status_code=400, detail="SITUACION debe ser 0 o 1.")
        if project.estado not in [0, 1]:
            raise HTTPException(status_code=400, detail="ESTADO debe ser 0 o 1.")
        if project.nivel not in [0, 1, 2]:
            raise HTTPException(status_code=400, detail="NIVEL debe ser 0, 1, o 2.")
        if project.funcion not in [0, 1, 2]:
            raise HTTPException(status_code=400, detail="FUNCION debe ser 0, 1, o 2.")
        if project.sector not in [0, 1, 2]:
            raise HTTPException(status_code=400, detail="SECTOR debe ser 0, 1, o 2.")

        # Preparar datos para predicción
        prediction_data = pd.DataFrame([{
            'MONTO_VIABLE': project.monto_viable,
            'COSTO_ACTUALIZADO': project.costo_actualizado,
            'SITUACION': project.situacion,
            'ESTADO': project.estado,
            'NIVEL': project.nivel,
            'FUNCION': project.funcion,
            'SECTOR': project.sector
        }])

        # Añadir columna 'FINALIZADO' para coincidir con las columnas del modelo
        prediction_data['FINALIZADO'] = 0

        # Normalizar los datos
        try:
            columns_to_scale = ['MONTO_VIABLE', 'COSTO_ACTUALIZADO', 'SITUACION', 'ESTADO', 'NIVEL', 'FUNCION', 'SECTOR']
            prediction_data_scaled = prediction_data.copy()
            prediction_data_scaled[columns_to_scale] = scaler.transform(prediction_data[columns_to_scale])
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error al normalizar los datos: {str(e)}")

        prediction_lstm = prediction_data_scaled.values.reshape((prediction_data_scaled.shape[0], 1, prediction_data_scaled.shape[1]))

        # Realizar predicción
        try:
            prediction = model.predict(prediction_lstm)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error al realizar la predicción: {str(e)}")

        # Interpretar resultados
        viable_score = float(prediction[0, 0])  # Convertir a float
        devengado_score = float(prediction[0, 1])  # Convertir a float
        viable_label = "Viable" if viable_score > 0.41 else "No viable"

        # Guardar el proyecto en el arreglo
        projects.append({
            "codigo_unico": project.codigo_unico,
            "Nombre del proyecto": project.nombre_proyecto,
            "Responsable": project.responsable,
            "Fecha de inicio": project.fecha_inicio,
            "Departamento": project.departamento,
            "Distrito": project.distrito,
            "Objetivo": project.objetivo,
            "Viable": viable_label,
            "Puntaje de viabilidad": round(viable_score, 2),
            "Devengado/Costo Actualizado": round(devengado_score, 2)
        })

        # Retornar respuesta
        return {
            "Proyecto registrado": {
                "Código único": project.codigo_unico,
                "Nombre del proyecto": project.nombre_proyecto,
                "Responsable": project.responsable,
                "Fecha de inicio": project.fecha_inicio,
                "Departamento": project.departamento,
                "Distrito": project.distrito,
                "Objetivo": project.objetivo
            },
            "Predicción": {
                "Viable": viable_label,
                "Puntaje de viabilidad": round(viable_score, 2),
                "Devengado/Costo Actualizado": round(devengado_score, 2)
            }
        }

    except HTTPException as http_ex:
        raise http_ex
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error inesperado: {str(e)}")




# Leer todos los proyectos
@app.get("/projects/")
def read_projects():
    if not projects:
        raise HTTPException(status_code=404, detail="No hay proyectos registrados.")
    return projects

# Leer un proyecto por código único
@app.get("/projects/{codigo_unico}")
def read_project(codigo_unico: str):
    for project in projects:
        if project.codigo_unico == codigo_unico:
            return project
    raise HTTPException(status_code=404, detail="Proyecto no encontrado.")

# Actualizar un proyecto por código único
@app.put("/projects/{codigo_unico}")
def update_project(codigo_unico: str, updated_project: Project):
    for index, project in enumerate(projects):
        if project.codigo_unico == codigo_unico:
            projects[index] = updated_project
            return updated_project
    raise HTTPException(status_code=404, detail="Proyecto no encontrado.")

# Eliminar un proyecto por código único
@app.delete("/projects/{codigo_unico}")
def delete_project(codigo_unico: str):
    for index, project in enumerate(projects):
        if project.codigo_unico == codigo_unico:
            del projects[index]
            return {"message": "Proyecto eliminado exitosamente."}
    raise HTTPException(status_code=404, detail="Proyecto no encontrado.")
