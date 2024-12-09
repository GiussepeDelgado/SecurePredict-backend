from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from fastapi.middleware.cors import CORSMiddleware
from typing import List
# Inicializar FastAPIs
app = FastAPI()

# Configuración de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Origen permitido (frontend Vite)
    allow_credentials=True,
    allow_methods=["*"],  # Permitir todos los métodos (GET, POST, PUT, etc.)
    allow_headers=["*"],  # Permitir todos los encabezados
)

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
projects = [
    {
        "codigo_unico": "SEC-003",
        "nombre_proyecto": "Capacitación a Serenazgo Local",
        "responsable": "Luis Gutiérrez",
        "fecha_inicio": "2024-01-10",
        "departamento": "Cusco",
        "distrito": "Wanchaq",
        "objetivo": "Mejorar la capacidad de respuesta",
        "monto_viable": 3000.0,
        "costo_actualizado": 2500.0,
        "situacion": 0,
        "estado": 0,
        "nivel": 0,
        "funcion": 1,
        "sector": 1,
        "viable": "No viable",
        "score_viabilidad": 0.38,
        "costo_devengado": 94.59
    },
    {
        "codigo_unico": "SEC-004",
        "nombre_proyecto": "Implementación de Cámaras de Vigilancia",
        "responsable": "María López",
        "fecha_inicio": "2024-02-15",
        "departamento": "Lima",
        "distrito": "San Borja",
        "objetivo": "Incrementar la seguridad ciudadana",
        "monto_viable": 5000.0,
        "costo_actualizado": 4500.0,
        "situacion": 1,
        "estado": 1,
        "nivel": 1,
        "funcion": 2,
        "sector": 0,
        "viable": "Viable",
        "score_viabilidad": 0.45,
        "costo_devengado": 88.24
    },
    {
        "codigo_unico": "SEC-005",
        "nombre_proyecto": "Reparación de Equipos de Serenazgo",
        "responsable": "Jorge Sánchez",
        "fecha_inicio": "2024-03-01",
        "departamento": "Arequipa",
        "distrito": "Cayma",
        "objetivo": "Mejorar el estado operativo de los equipos",
        "monto_viable": 4000.0,
        "costo_actualizado": 4200.0,
        "situacion": 0,
        "estado": 1,
        "nivel": 2,
        "funcion": 1,
        "sector": 1,
        "viable": "No viable",
        "score_viabilidad": 0.36,
        "costo_devengado": 105.0
    },
    {
        "codigo_unico": "SEC-006",
        "nombre_proyecto": "Contratación de Personal de Seguridad",
        "responsable": "Ana Martínez",
        "fecha_inicio": "2024-04-20",
        "departamento": "Trujillo",
        "distrito": "Centro Histórico",
        "objetivo": "Reforzar la seguridad en áreas críticas",
        "monto_viable": 7000.0,
        "costo_actualizado": 6000.0,
        "situacion": 1,
        "estado": 1,
        "nivel": 0,
        "funcion": 2,
        "sector": 2,
        "viable": "Viable",
        "score_viabilidad": 0.42,
        "costo_devengado": 85.71
    }
]

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
        global projects
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
        viable_label = "Viable" if viable_score > 0.39 else "No viable"

        # Guardar el proyecto en el arreglo
        projects.append({
            "codigo_unico": project.codigo_unico,
            "nombre_proyecto": project.nombre_proyecto,
            "responsable": project.responsable,
            "fecha_inicio": project.fecha_inicio,
            "departamento": project.departamento,
            "distrito": project.distrito,
            "objetivo": project.objetivo,
            "monto_viable": project.monto_viable,
            "costo_actualizado": project.costo_actualizado,
            "situacion": project.situacion,
            "estado": project.estado,
            "nivel": project.nivel,
            "funcion": project.funcion,
            "sector": project.sector,
            "viable": viable_label,
            "score_viabilidad": round(viable_score, 2),
            "costo_devengado": round(devengado_score, 2)
        })

        # Retornar respuesta
        return {
            "codigo_unico": project.codigo_unico,
            "nombre_proyecto": project.nombre_proyecto,
            "responsable": project.responsable,
            "fecha_inicio": project.fecha_inicio,
            "departamento": project.departamento,
            "distrito": project.distrito,
            "objetivo": project.objetivo,
            "monto_viable": project.monto_viable,
            "costo_actualizado": project.costo_actualizado,
            "situacion": project.situacion,
            "estado": project.estado,
            "nivel": project.nivel,
            "funcion": project.funcion,
            "sector": project.sector,
            "viable": viable_label,
            "score_viabilidad": round(viable_score, 2),
            "costo_devengado": round(devengado_score, 2)
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
        if project["codigo_unico"] == codigo_unico:  # Cambia la notación de punto a corchetes
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

# Función auxiliar para verificar la existencia del proyecto
def project_exists(codigo_unico: str):
    return any(project["codigo_unico"] == codigo_unico for project in projects)

# Endpoint POST para predicción en lote
@app.post("/projects/bulk")
def predict_bulk_projects(projects_list: List[Project]):
    results = []
    try:
        for project in projects_list:
            # Verificar si el proyecto ya existe
            if project_exists(project.codigo_unico):
                results.append({
                    "codigo_unico": project.codigo_unico,
                    "status": "Ya existe",
                    "message": "El proyecto ya existe y fue omitido."
                })
            else:
                # Si no existe, crearlo y generar predicción
                result = create_project(project)
                results.append({
                    "codigo_unico": project.codigo_unico,
                    "status": "Procesado",
                    "data": result
                })
        return {"message": "Proyectos procesados exitosamente", "data": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al procesar proyectos: {str(e)}")