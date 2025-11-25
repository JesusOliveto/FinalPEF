# Final PEF 2025 – Sistema de Recomendación de Rutas en una Ciudad Inteligente

Este proyecto implementa un **motor de ruteo** y una **aplicación web interactiva** para
recomendar rutas dentro de una ciudad ficticia que imita el trazado de **Jesús María (Córdoba)**.

El foco está en:

- Modelar una red vial realista (calles residenciales, colectoras y primarias).
- Incluir un modelo de tráfico histórico y un límite de velocidad realista del conductor.
- Ofrecer distintos algoritmos de ruteo (A*, Dijkstra, BFS) y modos de ruta multi‑destino.
- Proveer una interfaz en Streamlit para que el usuario explore rutas y métricas.
- Incluir herramientas de **testing** y **profiling** para evaluar el rendimiento del motor.

---

## 1. Propósito del sistema

El sistema busca responder a la pregunta:

> “Dado un origen y varios puntos de interés dentro de la ciudad, ¿cuál es la mejor ruta
>  para visitarlos, bajo diferentes modos (camino abierto o circuito), horarios y niveles
>  de tráfico?”

Casos de uso típicos:

- Planificación de recorridos para servicios municipales (recolección, inspecciones, etc.).
- Rutas para logística liviana dentro de la ciudad.
- Análisis de impacto del tráfico en diferentes horarios.

El sistema permite **experimentar** con:

- Algoritmo base de ruteo (A*, Dijkstra o BFS).
- Modo de visita:
	- `visit_all_open`: recorre todos los destinos y termina en el último.
	- `visit_all_circuit`: recorre todos los destinos y vuelve al origen (circuito).
	- `fixed_order`: respeta el orden en que se entregan los destinos.
- Horario del día (afecta congestión) y velocidad máxima del conductor.

---

## 2. Arquitectura general

El proyecto se organiza en tres componentes principales:

1. `logica.py` – Motor de ruteo
2. `streamlit_app.py` – Interfaz web (front en Streamlit)
3. `tests_profiling.py` – Tests unitarios y herramientas de profiling

### 2.1 Motor de ruteo (`logica.py`)

**Modelo de grafo**

- `Graph`, `Node`, `Edge` representan la red vial.
- La red de Jesús María se construye en `Graph.build_jesus_maria_hardcoded()`, que:
	- Genera parches de grilla rotados (centro, sudeste, noroeste).
	- Modela la RN‑9 como una diagonal primaria con conectores.
	- Asigna clases viales (`PRIMARY`, `COLLECTOR`, `RESIDENTIAL`) y velocidades.
	- Define qué segmentos son doble mano y cuáles son mano única.

**Modelo de tráfico**

- `HistoricalTrafficModel` hereda de `TrafficModel` y define un método
	`travel_time_seconds(edge, hour=...)` que:
	- Aplica un **factor de congestión** según hora del día y tipo de calle.
	- Limita la velocidad efectiva por un parámetro `driver_max_kmh` (ej. 40 km/h),
		para simular el comportamiento real de un conductor.

**Routers (algoritmos de ruteo)**

- `DijkstraRouter`:
	- Implementación de Dijkstra sobre tiempos de viaje.
	- Soporta **batching** de vecinos (uso de un generador `batcher`).
	- Usa `ThreadPoolExecutor` para paralelizar el cálculo de tiempos por arista.
	- Aplica **memoización** de resultados por clave `(graph, traffic, src, dst, hour)`.

- `AStarRouter`:
	- Implementa A* con **heurística temporal** basada en distancia geodésica (haversine)
		y una velocidad máxima optimista `vmax_h`.
	- Usa el mismo modelo de tráfico para el costo real de las aristas.

- `BFSRouter`:
	- Realiza una búsqueda en anchura (útil como baseline/depuración).
	- Encuentra un camino no ponderado y luego suma tiempos reales sobre ese camino.

**Servicio par‑a‑par y caché**

- `PairwiseDistanceService`:
	- Calcula una **matriz de tiempos** y un mapa de paths para una lista de waypoints.
	- Selecciona `AStarRouter`, `DijkstraRouter` o `BFSRouter` según `Algorithm`.
	- Usa `RouteCache` (LRU) para cachear tramos individuales (src, dst, hora, algoritmo).
	- Ejecuta los pares `(i, j)` en paralelo con `ThreadPoolExecutor`.

**Solvers multi‑stop**

- `HeldKarpExact`: implementación del algoritmo Held‑Karp (DP) para TSP con origen fijo.
	- Complejidad `O(n^2 2^n)`; se usa para problemas pequeños (n ≤ ~13).
	- Soporta modo circuito agregando el costo de regreso al origen.

- `HeuristicRoute`: heurística **Nearest Neighbor + 2‑opt**, con varios reinicios.
	- Se usa para casos con mayor cantidad de destinos.

**Servicio de alto nivel**

- `RoutingService` combina todas las piezas:
	- Llama a `PairwiseDistanceService.compute_matrix` para obtener tiempos y paths.
	- Selecciona solver exacto o heurístico según el tamaño del problema.
	- Ensambla la lista de `RouteLeg` usando `RouteSplicer`.
	- Devuelve un `RouteResult` con:
		- Orden de visita (`visit_order`).
		- Lista de tramos (`legs`).
		- Tiempo total (`total_seconds`) y distancia total (`total_distance_m`).
		- Resumen del algoritmo utilizado.

**Integración con IA (Gemini)**

- La función `route_with_gemini` permite delegar la decisión del orden de visita a
	un modelo de lenguaje (Gemini), mientras que los caminos y tiempos por tramo
	siguen siendo calculados por el motor local.
- Está implementada de forma opcional y robusta: ante errores o JSON inválido,
	aplica un **fallback local** (Held‑Karp o heurística).

### 2.2 Interfaz web (`streamlit_app.py`)

La app de Streamlit ofrece una UI interactiva para configurar y visualizar rutas:

- **Configuración lateral (sidebar)**:
	- Selección de algoritmo base: A*, Dijkstra, BFS.
	- Modo de ruta: abierto, circuito o en orden fijo.
	- Opción para usar IA (Gemini) en el ordenamiento de visitas.
	- Color de calles por clase o por nivel de tráfico estimado.
	- Controles de “Hora del día” y “Velocidad del conductor (km/h)”.

- **POIs (puntos de interés)**:
	- `make_pois` genera un conjunto pequeño de POIs (Base Central, Plaza, Escuela, etc.)
		mapeados a nodos del grafo.
	- El usuario elige un **origen** y uno o varios **destinos** por nombre.

- **Visualización con PyDeck**:
	- `build_road_layer` dibuja la red vial (PathLayer) y colorea las calles según
		clase o congestión.
	- `build_route_layers` dibuja la ruta resultante y los marcadores de origen/destinos.

- **Panel de métricas**:
	- Muestra tiempo total estimado en minutos, distancia total en km y algoritmo usado.
	- Detalla cada tramo (`RouteLeg`): origen/destino (por nombre), segundos y km.

### 2.3 Tests y profiling (`tests_profiling.py`)

Este módulo concentra:

- **Tests unitarios** sobre:
	- Construcción del grafo y no vacuidad.
	- Consistencia de costos entre Dijkstra y A* cuando existe ruta.
	- Validez básica de rutas BFS.
	- Formas y contenido de la matriz par‑a‑par.
	- Comportamiento de circuito (regreso al origen).
	- Herramientas de profiling (`run_profiled`, `profile_call`).
	- Memoización en `DijkstraRouter`.

- **Utilidades CLI** para:
	- `cmd_smoke`: smoke test end‑to‑end (un tramo representativo).
	- `cmd_tests`: ejecutar toda la suite de tests.
	- `cmd_profile`: perfilar una ruta multi‑destino y ver el top de funciones
		más costosas según `pstats`.

---

## 3. Tecnologías y técnicas utilizadas

- **Lenguaje**: Python 3.
- **Front**: Streamlit + PyDeck.
- **Estructuras de datos**: grafos dirigidos, listas de adyacencia, LRU cache.
- **Algoritmos de ruteo**: Dijkstra, A*, BFS.
- **Optimización**:
	- Batching de vecinos y uso de generators (`yield`).
	- Concurrencia con `ThreadPoolExecutor` para par‑a‑par.
	- Memoización de Dijkstra y caché LRU de tramos.
- **Optimización de rutas multi‑stop**: Held‑Karp (exacto) y heurística NN + 2‑opt.
- **Profiling**: `cProfile`, `pstats`, utilidades `run_profiled`, `profile_call`,
	y `ProfileTimer` (context manager para tiempos wall‑clock).
- **Testing**: `unittest`, con casos que reflejan tanto el motor interno como
	comportamientos cercanos al front.

---

## 4. Guía de uso desde la terminal

Para los siguientes comandos se asume que estás en el directorio raíz del proyecto:

```powershell
cd C:\Users\Gsu\Documents\FinalPEF
```

Y que estás utilizando el entorno virtual configurado en `.venv`.

### 4.1 Ejecutar la aplicación Streamlit

Para levantar el front web:

```powershell
C:/Users/Gsu/Documents/FinalPEF/.venv/Scripts/streamlit.exe run streamlit_app.py
```

Esto abrirá (o actualizará) la app en tu navegador en `http://localhost:8501`.

### 4.2 Ejecutar los tests unitarios

Para correr toda la suite de tests definida en `tests_profiling.py`:

```powershell
C:/Users/Gsu/Documents/FinalPEF/.venv/Scripts/python.exe tests_profiling.py --tests
```

Opcionalmente, puedes aumentar la verbosidad:

```powershell
C:/Users/Gsu/Documents/FinalPEF/.venv/Scripts/python.exe tests_profiling.py --tests --verbosity 2
```

### 4.3 Ejecutar el smoke test

El smoke test valida rápidamente que el motor puede calcular al menos un tramo
alcanzable y muestra una métrica sencilla en consola:

```powershell
C:/Users/Gsu/Documents/FinalPEF/.venv/Scripts/python.exe tests_profiling.py --smoke
```

Salida típica:

```text
Smoke OK: 0→1 · 0.3 min · 0.12 km
```

### 4.4 Ejecutar profiling de una ruta multi‑destino

Para perfilar una ruta con varios puntos (similar al uso del front):

```powershell
C:/Users/Gsu/Documents/FinalPEF/.venv/Scripts/python.exe tests_profiling.py \
	--profile \
	--algo astar \
	--mode visit_all_circuit \
	--n 6 \
	--hour 8 \
	--driver-max-kmh 40
```

Parámetros clave:

- `--algo`: `astar`, `dijkstra` o `bfs`.
- `--mode`: `visit_all_open`, `visit_all_circuit` o `fixed_order`.
- `--n`: cantidad de waypoints tomados de la red (mínimo 2).
- `--hour`: hora del día (0–23) para el modelo de tráfico.
- `--driver-max-kmh`: velocidad máxima del conductor.

La salida incluye un resumen de la ruta (tiempo y distancia total) y el top de
funciones más costosas según `pstats` (cumtime).

---

## 5. Resumen

Este proyecto integra:

- Un **motor de ruteo configurable** con varios algoritmos y un modelo de tráfico
	realista.
- Una **UI interactiva** que permite explorar rutas, algoritmos y modos de visita
	sobre una ciudad ficticia pero coherente.
- Un set de **tests y herramientas de profiling** que facilitan validar la
	corrección y medir el rendimiento.

Este README funciona como informe técnico del sistema, cubriendo propósito,
arquitectura, tecnologías y forma de ejecución, y sirve como base para
extensiones futuras (por ejemplo, incorporar datos reales de OSM o más heurísticas
de optimización de rutas).

---

## 6. Informe de rendimiento

Para evaluar el rendimiento del motor de ruteo, se ejecutó el comando de profiling
incluido en `tests_profiling.py`:

```powershell
C:/Users/Gsu/Documents/FinalPEF/.venv/Scripts/python.exe tests_profiling.py \
	--profile \
	--algo astar \
	--mode visit_all_circuit \
	--n 6 \
	--hour 8 \
	--driver-max-kmh 40
```

Configuración del escenario:

- Algoritmo base de tramos: `astar`.
- Modo de ruta: `visit_all_circuit` (circuito que vuelve al origen).
- Número de nodos considerados: `n = 6`.
- Hora del día: `8` (horario pico de la mañana, mayor congestión).
- Velocidad máxima del conductor: `40 km/h`.

### 6.1 Resultados numéricos (A* vs Dijkstra)

**Escenario 1 – A***

- Comando:
	- `--algo astar --mode visit_all_circuit --n 6 --hour 8 --driver-max-kmh 40`.
- Resumen de la ruta:
	- `Total: 155.5s · 1.15 km · algoritmo: astar + Held-Karp`.
- Medidas de rendimiento:
	- `Wall time: 0.0210 s`.
	- `Peak memory: 0.134 MiB`.
	- Llamadas totales: `7807` (7738 primitivas).
	- Top por `cumtime`: `RoutingService.route`, `PairwiseDistanceService.compute_matrix`,
		funciones internas de `concurrent.futures`/`threading`, `route_pair` y
		`AStarRouter.route`.

**Escenario 2 – Dijkstra**

- Comando:
	- `--algo dijkstra --mode visit_all_circuit --n 6 --hour 8 --driver-max-kmh 40`.
- Resumen de la ruta:
	- `Total: 155.5s · 1.15 km · algoritmo: dijkstra + Held-Karp`.
- Medidas de rendimiento:
	- `Wall time: 0.1769 s`.
	- `Peak memory: 0.250 MiB`.
	- Llamadas totales: `48419` (43261 primitivas).
	- Top por `cumtime`: dominado por utilidades de concurrencia
		(`concurrent.futures._base`, `threading`, colas internas) y por
		`RoutingService.route`, `PairwiseDistanceService.compute_matrix`,
		`route_pair` y `DijkstraRouter.route`.

En ambos casos, el costo de la ruta (155.5 s y 1.15 km) es idéntico, ya que
tanto A* como Dijkstra operan sobre la misma red y el mismo modelo de tráfico.

### 6.2 Interpretación

**Calidad de la solución (tiempo de viaje de la ruta, 155.5 s)**

- Representa el tiempo estimado sobre la red vial y el modelo de tráfico.
- Para un circuito de ~1.15 km en hora pico con un límite de 40 km/h y
	factores de congestión, un tiempo del orden de 2–3 minutos es coherente
	con lo esperado para una ciudad compacta.

**Rendimiento comparativo A* vs Dijkstra**

- **A***:
	- Wall time ~0.021 s; pico de memoria ~0.134 MiB.
	- Usa una heurística geográfica de tiempo (`est`) para priorizar nodos que
		están “más cerca en tiempo” del destino.
	- Esto reduce la cantidad de nodos expandidos por tramo y, en consecuencia,
		la cantidad de llamadas a funciones y trabajo del `ThreadPoolExecutor`.

- **Dijkstra**:
	- Wall time ~0.177 s (≈8 veces más lento en este escenario); pico de memoria
		~0.250 MiB (casi el doble).
	- No usa heurística: explora el grafo de manera más exhaustiva, lo que
		incrementa significativamente el número de llamadas y trabajo por hilo.
	- Las trazas de `concurrent.futures` y `threading` dominan más el perfil
		debido a la mayor cantidad de tareas procesadas.

**Técnicas que impactan en el rendimiento**

- **Heurística informada (A*)**:
	- La heurística de tiempo geográfico hace que A* expanda menos nodos que
		Dijkstra para llegar al mismo resultado, reduciendo el coste total.
- **Batching y concurrencia en Dijkstra**:
	- El ruteador Dijkstra está optimizado con batching y `ThreadPoolExecutor`,
		pero aun así, al no contar con heurística, realiza más trabajo global en
		este tipo de escenario.
- **Caché de tramos (RouteCache) y memoización**:
	- Ambas técnicas benefician tanto a A* como a Dijkstra, evitando recomputar
		tramos repetidos y reduciendo tiempo y llamadas a funciones.
- **Modelo de tráfico y límite de velocidad**:
	- Afectan los valores de costo, pero no la complejidad; ambos algoritmos
		trabajan sobre los mismos costos y producen resultados coherentes.
	- Representa el tiempo estimado sobre la red vial y el modelo de tráfico.
	- Para un circuito de ~1.15 km en hora pico con un límite de 40 km/h y
		factores de congestión, un tiempo del orden de 2–3 minutos es coherente
		con lo esperado para una ciudad compacta.

- **Wall time del motor (0.021 s)**:
	- La ejecución completa del caso de prueba (cálculo de matriz, Held‑Karp,
		ensamblado de tramos y reporting) tarda alrededor de 21 milisegundos en la
		máquina de prueba.
	- Esto indica que, para tamaños de problema pequeños/medios (como el uso típico
		desde la UI), el motor responde prácticamente en tiempo real.

- **Uso de memoria (pico ~0.134 MiB)**:
	- El perfil de memoria muestra un pico muy bajo (del orden de décimas de MiB),
		lo que confirma que la estructura de datos (grafo, matrices y cachés) es
		ligera para el tamaño actual de la red y del problema.
	- Esto deja margen para aumentar el número de destinos o la densidad de la
		red sin comprometer la memoria en entornos de escritorio.

- **Distribución del costo en CPU**:
	- Las funciones con mayor tiempo acumulado son:
		- `RoutingService.route` y `PairwiseDistanceService.compute_matrix`, que
			orquestan el cálculo de la matriz de tiempos y el orden de visita.
		- Lógicas internas de `ThreadPoolExecutor` y `threading`, resultado esperado
			dado que se utiliza concurrencia para los pares par-a-par.
		- Las funciones de cálculo de heurística (`est`) y de ruteo (`route_pair`,
			`DijkstraRouter.route`/`AStarRouter.route`).
	- No se observan cuellos de botella inesperados fuera de estas zonas críticas,
		lo que sugiere que la mayor parte del coste se invierte efectivamente en
		cómputo de rutas, y no en overhead innecesario.

En resumen, el motor cumple con los objetivos de:

- Mantener tiempos de respuesta muy bajos para escenarios típicos usados en la UI.
- Utilizar memoria de forma eficiente.
- Concentrar el coste de CPU en las partes esperadas del pipeline de cálculo
	(matriz par-a-par, ruteo y solver TSP).
- Demostrar que **A***, gracias a su heurística, resulta más eficiente que
	Dijkstra en este contexto, manteniendo la misma calidad de solución.



## Historial resumido del proyecto "Final PEF 2025"

Autor: Ignacio Jesús Olariaga Oliveto y Facundo Lopez Moreno

**Herramientas utilizadas para el desarrollo:**

- Git para control de versiones.
- Streamlit para la creación de la aplicación web.
- ChatGPT para asistencia en diseño inicial y generación de código.
- Copilot con Visual Studio Code para autocompletado y sugerencias de código.
- Copilot Agents con modelo GPT-5 para generación avanzada de código y documentación.
- PlantUML para diagramas UML.
- PyDeck para visualización de mapas interactivos.
- OSMnx para integración con datos de OpenStreetMap.
- PEP8 para estandarización de estilo de código.


### 1. Inicio del repositorio

- `3f6aac4e30a2` — first commit.
- `756122871c6a` — todos los archivos base.

### 2. Creación de la app "Pueblito: Rutas Inteligentes"

- `daafe24bcb6f` — feat: Add Streamlit app for intelligent routing in "Pueblito":
	- Se crea la primera versión de la app Streamlit (`streamlit_app.py`).
	- Integración con PyDeck para mapas interactivos (calles, tráfico, POIs).
	- Se agregan diagramas UML (`DER.puml`, `DER_operacional.puml`, `diagrama.puml`).
	- Se actualiza `requirements.txt` con dependencias de Streamlit y testing.

- `4dbc1df3d511`, `fcd6ea4dcb87` — Fixes en referencias a `graph` y ajustes en `make_pois`.

### 3. Integración con OSM y mejoras de visualización

- `10cbf34607a8` — feat: Enhance "Pueblito" app with OSM integration and new features:
	- Toggle entre grilla sintética y grafo OSM de Jesús María.
	- Carga y caché de grafo OSM (GraphML opcional).
	- Mejoras en estilos visuales y capas de mapa.

- `f0b1cf1c22c3` — fix: Añadir `osmnx` a los requisitos.
- `2da98fb397e5` — feat: Añadir opciones de fallback en la carga de OSM.
- `c590ced1519f` — Refactor de la app para simplificar integración OSM y UI.
- `e46d1f91152f`, `e8fddbfa0577` — Ajustes de coordenadas y rotación de puntos.

### 4. Ajustes de grilla, centrado y modelo de grafo

- `86f8206b1096` — Actualizar coordenadas base en `Graph` y POIs.
- `b15436e64a8f` — Refactor de estructura de código para legibilidad.
- `6c5e7bc5ad39`, `a9478a1e85fe`, `d76217a33884e`, `c327ce20f20e`, `12342e1a1cd2`,
	`422567486add`, `d7bdb69c6bef`, `898900e4af3a` — Ajustes sucesivos de centrado y grilla.
- `60e552715980`, `2e49273d98ae` — Fixes de estilo del mapa (`none`) en PyDeck.

### 5. Hacia un modelo de ciudad más realista

- `e1d6710fd5e0` — Implementar política de doble mano y ajustes en aristas del grafo.
- `35b8ca1e2768` — Refactorización general.
- `7f9d52884e29` — Mejorar documentación y refactorizar funciones en `logica.py` y `streamlit_app.py`.
- `2c00a67a861e` — Agregar límite de velocidad al modelo de tráfico histórico y ajustar carga de servicios.

### 6. Transición a "Jesús María: Rutas Inteligentes" e integración IA

- `ef71292137b5` — Eliminar heurísticas del API público y ajustar inicialización de servicios.
- `e87aeb0e987d` — Añadir regreso al origen en modo circuito.
- `e88cb18ef225` — Eliminar entradas manuales y simplificar selección de destinos.
- `8243bf592e04` — Ajustar slider de hora del día en la barra lateral.

- `3b3a07cbc6aa` — Refactor grande de `streamlit_app.py`:
	- Eliminar `SSSPMemo` y banderas de dirección de aristas en la UI.
	- Mejorar botón "Clear Destinations", capas de mapa y generación de POIs.
	- Limpiar configuración de la barra lateral.
	- Añadir primera implementación del cliente Google GenAI (Gemini).

- `005eb188929f`, `1c791c18cb9a` — Limpieza de código muerto y pequeños arreglos.
- `4df909b83b23` — "integracion de gemini": refuerzo de la integración IA.

### 7. Ajustes de contexto "Jesús María" y mejoras de UI

- `64a5073c9fcd` — Actualizar nombres/participantes en diagramas; agregar `legs` a `RouteResult`;
	mejorar UI para "Jesús María: Rutas Inteligentes".

- `42105c7fb03e` — Eliminar `SSSPMemo` no utilizado en `logica.py`; mejorar etiquetas de POI
	y textos de botones en `streamlit_app.py`.

- `c869e9f0fc6c` — Agregar nota descriptiva al Router en `DER_operacional.puml`.

### 8. Optimización del motor y documentación

- `2c3b734a3d9a` — Batching y paralelización en Dijkstra; memoización de resultados.
	("Requerimientos técnicos cumplidos").

- `7894d1eee148` — Documentación detallada en `logica.py` y `streamlit_app.py`.
- `9c7a75b04e57` — Normalización de estilo PEP8.
- `57653182128c` — Funciones de perfilado/temporización en `logica.py` y CLI de utilidades.

### 9. Afinado de experiencia de usuario

- `bdfc0d12b097` — Tiempo estimado en minutos en la interfaz Streamlit.
- `67d7051e1c13`, `c1f4de3fc685`, `1f89e0f0ae0f` — Ajustes sucesivos de posición y filas
	en los parches de grilla en `Graph`.
- `7be59d3ce793` — Agregar nombres de POI y mejorar visualización de rutas en la UI.

### 10. Separación de responsabilidades, tests y profiling

- `4e59307cc92a` — Refactor routing engine tests and profiling utilities:
	- Eliminación de código temporal y `temp.txt`.
	- Creación de `tests_profiling.py` para concentrar tests y profiling.
	- Implementación de casos de prueba para Dijkstra, A* y servicios auxiliares.
	- Introducción de smoke tests y comandos CLI para `--tests`, `--smoke` y `--profile`.
