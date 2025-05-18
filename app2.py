import joblib
from shiny import App, ui, render, reactive
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates



# Cargar el modelo previamente guardado
modelo = joblib.load('modelos/modelo_autos.pkl')

future = modelo.make_future_dataframe(periods=11, freq='MS')  # 'MS' = mes de inicio
forecast = modelo.predict(future)

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Dividir forecast en parte observada y parte futura
last_date = modelo.history['ds'].max()
forecast_hist = forecast[forecast['ds'] <= last_date]
forecast_future = forecast[forecast['ds'] >= last_date]

# Crear la figura
fig, ax = plt.subplots(figsize=(12, 6))

# Datos reales
#ax.plot(modelo.history['ds'], modelo.history['y'], label='Datos reales', color='#3498DB', linewidth=2)

# Predicción en periodo histórico (opcional)
ax.plot(forecast_hist['ds'], forecast_hist['yhat'], label='Ajuste histórico', color='#3498DB')

# Predicción futura (aquí se aplica el color naranja)
ax.plot(forecast_future['ds'], forecast_future['yhat'], label='Predicción futura', color='#E67E22', linewidth=2.5)

# Banda de incertidumbre para predicción futura
ax.fill_between(forecast_future['ds'],
                forecast_future['yhat_lower'],
                forecast_future['yhat_upper'],
                color='#FAD7A0', alpha=0.4, label='Intervalo de confianza (futuro)')

# Estética del eje X
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.setp(ax.get_xticklabels(), rotation=90)

# Títulos y leyenda
ax.set_title("Predicción de Prophet con colores personalizados", pad=15)
ax.set_xlabel("Fecha")
ax.set_ylabel("Triciclos")
ax.grid(True, linestyle='--', alpha=0.3)
ax.legend()
plt.tight_layout()
#plt.show()

# Lista de vehículos y sus nombres de archivo de modelos (ajusta nombres)
modelos_disponibles = {
    'AUTOS': 'modelos/modelo_autos.pkl',
    'MOTOS': 'modelos/modelo_motos.pkl',
    'AUTOBUS DE 2 EJES': 'modelos/modelo_autobus_2_ejes.pkl',
    'AUTOBUS DE 3 EJES': 'modelos/modelo_autobus_3_ejes.pkl',
    'AUTOBUS DE 4 EJES': 'modelos/modelo_autobus_4_ejes.pkl',
    'CAMIONES DE 2 EJES': 'modelos/modelo_camiones_2_ejes.pkl',
    'CAMIONES DE 3 EJES': 'modelos/modelo_camiones_3_ejes.pkl',
    'CAMIONES DE 4 EJES': 'modelos/modelo_camiones_4_ejes.pkl',
    'CAMIONES DE 5 EJES': 'modelos/modelo_camiones_5_ejes.pkl',
    'CAMIONES DE 6 EJES': 'modelos/modelo_camiones_6_ejes.pkl',
    'CAMIONES DE 7 EJES': 'modelos/modelo_camiones_7_ejes.pkl',
    'CAMIONES DE 8 EJES': 'modelos/modelo_camiones_8_ejes.pkl',
    'CAMIONES DE 9 EJES': 'modelos/modelo_camiones_9_ejes.pkl',
    'TRICICLOS': 'modelos/modelo_triciclos.pkl',
    'EJE EXTRA AUTOBUS': 'modelos/modelo_eje_extra_autobus.pkl',
    'EJE EXTRA CAMION': 'modelos/modelo_eje_extra_camion.pkl',
    'PEATONES': 'modelos/modelo_peatones.pkl'    
    # agrega los demás...
}

# Datos para selección rango años, ejemplo estático, tú puedes usar df['AÑO'] u otro
anos_disponibles = list(range(2021, 2026))
meses_disponibles = list(range(1, 13))


app_ui = ui.page_fluid(
    ui.card(
        ui.card_header("Movimientos mensuales por tipo de vehículo en la red CAPUFEx 2021-2025"),
        ui.layout_sidebar(
            ui.sidebar(
                ui.h4("Controles de Pronóstico"),
                ui.input_slider(
                    "anio_pronostico",
                    "Rango de años",
                    min=anos_disponibles[0], max=anos_disponibles[-1],
                    value=(anos_disponibles[0], anos_disponibles[-1])
                ),
                ui.input_select(
                    "tipo_vehiculo",
                    "Tipo de vehículo",
                    choices=list(modelos_disponibles.keys()),
                    selected="AUTOS"
                ),
                ui.input_select(
                    "mes_pronostico",
                    "Mes para pronóstico",
                    choices=[str(m) for m in meses_disponibles],
                    selected="1"
                ),
                ui.hr(),
                ui.h4("Controles de Visualización"),
                ui.input_select(
                    "anio_visualizacion", 
                    "Año para visualización", 
                    choices = {str(year): year for year in anos_disponibles}
                ),
                ui.input_checkbox_group(
                    "tipos_visualizacion", 
                    "Tipos de vehículos a visualizar", 
                    choices=[
                        'AUTOS', 'MOTOS', 'AUTOBUS DE 2 EJES', 'AUTOBUS DE 3 EJES',
                        'AUTOBUS DE 4 EJES', 'CAMIONES DE 2 EJES', 'CAMIONES DE 3 EJES',
                        'CAMIONES DE 4 EJES', 'CAMIONES DE 5 EJES', 'CAMIONES DE 6 EJES',
                        'CAMIONES DE 7 EJES', 'CAMIONES DE 8 EJES', 'CAMIONES DE 9 EJES',
                        'TRICICLOS'
                    ],
                    selected=['AUTOS', 'MOTOS']
                ),
                width=300,
            ),
            ui.navset_card_tab(
                ui.nav_panel("Pronóstico",
                    ui.layout_columns(  
                        ui.card(
                            #ui.card_header("Card 1 header"),
                            ui.layout_columns(
                                 ui.output_image("image"),
                                 ui.output_ui("total_card_text"),
                            ),max_height='130px',fill=False ,full_screen=False
                            
                        ),  
                        ui.card(  
                            #ui.card_header("Card 2 header"),
                            ui.layout_columns(
                                 ui.output_image("image2"),
                                 ui.p("FRECUENCIA",ui.p("1 MES")),    
                            ),max_height='130px',fill=False ,full_screen=False
                        ),
                        ui.card(  
                            #ui.card_header("Card 3 header"),
                            ui.layout_columns(
                                 ui.output_image("image3"),
                                 ui.output_ui("pronostico_card_text"), 
                            ),max_height='130px',fill=False ,full_screen=False
                        ),    
                    ),
                    ui.layout_columns(
                        ui.card(
                            ui.strong("TABLA DE DATOS HISTORICOS"),
                            ui.output_data_frame("penguins_df")
                        ),
                        ui.card(
                            #ui.p("TABLA DE PREDICCIONES"),
                            #ui.p("DATOS HISTORICOS"),
                            ui.output_plot(
                                "plot_prediccion",width='100%',height='600px',
                                click=False,  
                                dblclick=False,  
                                hover=False,  
                                brush=False,  
                            ),
                        )
                    ),
                    ui.layout_columns(
                        ui.card(
                            ui.strong("FRECUENCIA DE TIPOS DE VEHICULOS POR AÑOS"),
                            ui.output_plot("distribucion_plot")
                        ),
                        ui.layout_columns(
                            ui.card(
                                ui.strong("CANTIDAD DE VEHICULOS"),
                                ui.output_ui("cantidad_vehiculos_card")
                            ),
                            ui.card(
                                ui.strong("ESTADISTICAS"),
                                ui.output_ui("estadisticas_card")
                            )
                        )
                    )),
                ui.nav_panel("Distribución", ui.output_text_verbatim("respuestas_text"))
            )
        )
    )
)


def server(input, output, session):

    @reactive.Calc
    def modelo_cargado():
        tipo = input.tipo_vehiculo()
        ruta_modelo = modelos_disponibles.get(tipo)
        if ruta_modelo is None:
            return None
        modelo = joblib.load(ruta_modelo)
        return modelo

    @reactive.Calc
    def rango_fechas():
        anio_min, anio_max = input.anio_pronostico()
        mes = int(input.mes_pronostico())
        return anio_min, anio_max, mes
    
    @render.image  
    def image():
        tipo = input.tipo_vehiculo()
    
        # Mapear tipo de vehículo a nombre de imagen
        if tipo == "AUTOS":
            ruta = "www/img/car.png"
        elif tipo == "EJE EXTRA AUTOBUS":
            ruta = "www/img/autoRemolcado.png"
        elif "AUTOBUS" in tipo:
            ruta = "www/img/autobus.png"
        elif "CAMIONES" in tipo:
            ruta = "www/img/camion.png"
        elif "MOTOS" in tipo:
            ruta = "www/img/moto.png"
        elif "TRICICLOS" in tipo:
            ruta = "www/img/mototriciclo.png"
        elif "PEATONES" in tipo:
            ruta = "www/img/peaton.png"
        elif "EJE EXTRA CAMION" in tipo:
            ruta = "www/img/camionRemolcado.png"    
        else:
            ruta = "www/img/car.png"  # una imagen genérica por si acaso

        return {"src": ruta, "width": "130px"}
    
    @render.image  
    def image2():
        img2 = {"src":"www/img/calendar.png", "width": "100px"}  
        return img2
    
    @render.image  
    def image3():
        tipo = input.tipo_vehiculo()
    
        # Mapear tipo de vehículo a nombre de imagen
        if tipo == "AUTOS":
            ruta = "www/img/car.png"
        elif tipo == "EJE EXTRA AUTOBUS":
            ruta = "www/img/autoRemolcado.png"
        elif "AUTOBUS" in tipo:
            ruta = "www/img/autobus.png"
        elif "CAMIONES" in tipo:
            ruta = "www/img/camion.png"
        elif "MOTOS" in tipo:
            ruta = "www/img/moto.png"
        elif "TRICICLOS" in tipo:
            ruta = "www/img/mototriciclo.png"
        elif "PEATONES" in tipo:
            ruta = "www/img/peaton.png"
        elif "EJE EXTRA CAMION" in tipo:
            ruta = "www/img/camionRemolcado.png"
        else:
            ruta = "www/img/car.png"  # una imagen genérica por si acaso

        return {"src": ruta, "width": "130px"}
    
    @reactive.Calc
    def total_pronosticado_actual():
        modelo = modelo_cargado()
        if modelo is None:
            return "Modelo no disponible", "N/A"

        anio_min, anio_max, mes = rango_fechas()
        
        fecha_inicio = pd.to_datetime("2021-01-01")
        fecha_fin = pd.to_datetime(f"{anio_max}-{mes:02d}-01")

        fecha_ultima_real = modelo.history['ds'].max()

        # 1. Sumamos datos reales hasta el límite entre enero 2021 y fecha_fin
        df_real = modelo.history[
            (modelo.history['ds'] >= fecha_inicio) &
            (modelo.history['ds'] <= fecha_fin)
        ]
        total = df_real['y'].sum() if not df_real.empty else 0

        # 2. Si la fecha seleccionada supera la última real, predecimos lo que falta
        if fecha_fin > fecha_ultima_real:
            fechas_faltantes = pd.date_range(
                start=fecha_ultima_real + pd.DateOffset(months=1),
                end=fecha_fin,
                freq='MS'
            )
            if not fechas_faltantes.empty:
                df_futuro = pd.DataFrame({'ds': fechas_faltantes})
                forecast = modelo.predict(df_futuro)
                total += forecast['yhat'].sum()

        # 3. Formateo de resultado
        tipo = input.tipo_vehiculo()
        texto = f"TOTAL DE {tipo} HASTA {anio_max}"
        valor_formateado = f"{int(total):,}".replace(",", ".")
        return texto, valor_formateado



    @render.ui
    def pronostico_card_text():
        modelo = modelo_cargado()
        if modelo is None:
            return ui.p("PRONÓSTICO", ui.p("N/A"))

        anio_min, anio_max, mes = rango_fechas()
        anio = anio_max
        fecha_objetivo = pd.to_datetime(f"{anio}-{mes:02d}-01")
        future_df = pd.DataFrame({'ds': [fecha_objetivo]})
        prediccion = modelo.predict(future_df)

        if not prediccion.empty:
            valor = int(prediccion.iloc[0]['yhat'])
            valor_formateado = f"{valor:,}".replace(",", ".")
        else:
            valor_formateado = "Sin datos"

        return ui.p("PRONÓSTICO", ui.p(valor_formateado))


    @render.ui
    def total_card_text():
        titulo, valor = total_pronosticado_actual()
        return ui.p(titulo, ui.p(valor))


    @reactive.Calc
    def tabla_completa():
        ruta_csv = "dataclean.csv"  # ajusta si está en una carpeta: "data/datos.csv"
        df = pd.read_csv(ruta_csv)

        # Asegura que la columna de fecha sea tipo datetime y esté ordenada
        df['FECHA'] = pd.to_datetime(df['FECHA'])
        df = df.sort_values('FECHA')

        return df



    @render.data_frame
    def penguins_df():
        return render.DataGrid(tabla_completa())
    

    @render.plot
    def distribucion_plot():
        df = tabla_completa()
        anio = int(input.anio_visualizacion())
        tipos = input.tipos_visualizacion()

        # Filtrar por año
        df['AÑO'] = df['FECHA'].dt.year
        df_filtrado = df[df['AÑO'] == anio]

        # Sumar por cada tipo seleccionado
        totales = {tipo: df_filtrado[tipo].sum() for tipo in tipos if tipo in df_filtrado.columns}

        # Definir una paleta de colores para cada tipo de vehículo
        colores = {
            'AUTOS': '#3498DB',        # Azul
            'MOTOS': '#E74C3C',        # Rojo
            'TRICICLOS': '#F39C12',    # Naranja
            'AUTOBUS': '#2ECC71',      # Verde
            'CAMIONES': '#9B59B6',     # Morado
            'PEATONES': '#1ABC9C'      # Turquesa
        }
        
        # Crear lista de colores para cada barra
        colores_barras = [colores.get(tipo, '#95A5A6') for tipo in totales.keys()]  # Gris por defecto

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(totales.keys(), totales.values(), color=colores_barras)
        ax.set_title(f"Frecuencia de tipos de vehículos en {anio}")
        ax.set_ylabel("Cantidad total")
        ax.set_xticklabels(totales.keys(), rotation=45)
        plt.tight_layout()
        return fig
    
    @render.ui
    def estadisticas_card():
        df = tabla_completa()
        anio = int(input.anio_visualizacion())
        tipos = input.tipos_visualizacion()
        df['AÑO'] = df['FECHA'].dt.year
        df_filtrado = df[df['AÑO'] == anio]

        totales = {tipo: df_filtrado[tipo].sum() for tipo in tipos if tipo in df_filtrado.columns}
        
        if not totales:
            return ui.p("No hay datos disponibles para los tipos seleccionados.")

        tipo_max = max(totales, key=totales.get)
        tipo_min = min(totales, key=totales.get)

        return ui.div(
        ui.p(ui.strong("Vehículo con mayor movimiento:"), f" {tipo_max} ({int(totales[tipo_max]):,})"),
        ui.p(ui.strong("Vehículo con menor movimiento:"), f" {tipo_min} ({int(totales[tipo_min]):,})")
        )


    
    @render.ui
    def cantidad_vehiculos_card():
        df = tabla_completa()
        anio = int(input.anio_visualizacion())
        tipos = input.tipos_visualizacion()
        df['AÑO'] = df['FECHA'].dt.year
        df_filtrado = df[df['AÑO'] == anio]

        cards = []
        for tipo in tipos:
            if tipo not in df_filtrado.columns:
                continue
            total = int(df_filtrado[tipo].sum())
            total_txt = f"{total:,}".replace(",", ".")
            
            # Rutas consistentes usando solo 'www/img/'
            if tipo == "AUTOS":
                ruta = "www/img/car.png"  # Corregido: agregado 'www/'
            elif tipo == "MOTOS":
                ruta = "www/img/moto.png"
            elif tipo == "TRICICLOS":
                ruta = "www/img/mototriciclo.png"
            elif "CAMIONES" in tipo:
                ruta = "www/img/camion.png"
            elif "AUTOBUS" in tipo:
                ruta = "www/img/autobus.png"
            elif "PEATONES" in tipo:
                ruta = "www/img/peaton.png"
            else:
                ruta = "www/img/car.png"

            cards.append(
                ui.layout_columns(
                    #img problemas de rutas
                    #ui.img(src=ruta, width="60px"),
                    ui.p(f"{tipo}: {total_txt}")
                )
            )
        return ui.div(*cards)

    @output
    @render.plot
    def plot_prediccion():
        modelo = modelo_cargado()
        if modelo is None:
            return None

        anio_min, anio_max, mes = rango_fechas()

        # Crear dataframe futuro: desde anio_min/mes hasta anio_max/mes, frecuencia mensual inicio de mes
        start_date = f"{anio_min}-{mes:02d}-01"
        end_date = f"{anio_max}-{mes:02d}-01"
        fechas_futuras = pd.date_range(start=start_date, end=end_date, freq='MS')

        # Para prophet necesitamos df con columna 'ds'
        future = pd.DataFrame({'ds': fechas_futuras})

        # Predecir con el modelo
        forecast = modelo.predict(future)

        # Fecha límite datos históricos para separar
        last_date = modelo.history['ds'].max()

        # Dividir forecast en histórico y futuro (en caso de que futuro incluya histórico)
        forecast_hist = forecast[forecast['ds'] <= last_date]
        forecast_future = forecast[forecast['ds'] >= last_date]

        fig, ax = plt.subplots(figsize=(12, 6))

        # Gráfico ajuste histórico
        if not forecast_hist.empty:
            ax.plot(forecast_hist['ds'], forecast_hist['yhat'], label='Ajuste histórico', color='#3498DB')

        # Gráfico predicción futura
        if not forecast_future.empty:
            ax.plot(forecast_future['ds'], forecast_future['yhat'], label='Predicción futura', color='#E67E22', linewidth=2.5)
            ax.fill_between(forecast_future['ds'],
                            forecast_future['yhat_lower'],
                            forecast_future['yhat_upper'],
                            color='#FAD7A0', alpha=0.4, label='Intervalo de confianza (futuro)')

        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.setp(ax.get_xticklabels(), rotation=90)

        ax.set_title(f"Predicción {input.tipo_vehiculo()} desde {anio_min} hasta {anio_max}, mes {mes}", pad=15)
        ax.set_xlabel("Fecha")
        ax.set_ylabel("Cantidad")
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.legend()
        plt.tight_layout()

        return fig
    


app = App(app_ui, server)
app.run()

