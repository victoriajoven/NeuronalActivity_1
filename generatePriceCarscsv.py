import csv
import random

# Definir opciones posibles
manufacturers = ["BMV", "SEAT", "OPEL", "MERCEDES"]
colors = ["red", "sky blue", "black", "white", "yellow", "orange", "navy blue", "pearl white"]
fuel_types = ["electric", "hybrid", "gasoline"]
doors_options = [2, 4]

# Nombre del archivo CSV
filename = "PriceCars.csv"

# Número de registros
num_records = 1000

# Abrir archivo para escritura
with open(filename, mode='w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    
    # Escribir cabecera
    writer.writerow([
        "idRegistro", "manufacturer", "color", "fuel_type", "year",
        "consumptionliters", "doors", "base_price", "gps_price",
        "leather_seats_price", "sport_package_price", "safety_package_price",
        "CO_Emissions"
    ])
    
    # Generar registros
    for i in range(1, num_records + 1):
        manufacturer = random.choice(manufacturers)
        color = random.choice(colors)
        fuel_type = random.choice(fuel_types)        
       
        # Consumo
        if fuel_type == "electric":
            consumption = 0.0
            year = random.randint(2019, 2025)
        
        elif fuel_type == "hybrid":
            consumption = round(random.uniform(5, 7), 2)
            year = random.randint(2019, 2025)
        else:
            consumption = round(random.uniform(7, 12), 2)
            year = random.randint(2010, 2025)

        doors = random.choice(doors_options)
        
        # Precios opcionales
        base_price = round(random.uniform(15000, 40000), 2)
        gps_price = round(random.uniform(500, 1500), 2)        
        sport_package_price = round(random.uniform(500, 1500), 2)
        safety_package_price = round(random.uniform(500, 1500), 2)
        
        # Emisiones CO2
        if fuel_type == "electric":
            co_emissions = 0.0
        elif fuel_type == "hybrid":
             #if consumption  menor o igual 6:
            co_emissions = round(random.uniform(20, 65), 4)
            #if consumption igual o mayor a 6
            co_emissions = round(random.uniform(66, 90), 4)
        else:  # gasoline
            #if consumption  menor o igual 7.5:
            co_emissions = round(random.uniform(100, 170), 4)
             #if consumption  mayor a  7.5:
            co_emissions = round(random.uniform(171, 240), 4)
        
        # Escribir fila
        writer.writerow([
            i, manufacturer, color, fuel_type, year,
            consumption, doors, base_price, gps_price,
            sport_package_price, safety_package_price,
            co_emissions
        ])

print(f"Archivo '{filename}' creado con {num_records} registros.")
