import torch
import serial
import time
import numpy as np
from model import NavigationNet
import joblib
import os
import socket

# ==============================
# Rutas absolutas
# ==============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Carpeta IAccd
MODEL_PATH = os.path.join(BASE_DIR, "robot_navigation_model.pth")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

print("Buscando archivos en:", BASE_DIR)
print("Existe modelo:", os.path.exists(MODEL_PATH))
print("Existe escalador:", os.path.exists(SCALER_PATH))

if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    raise FileNotFoundError("Modelo o escalador no encontrados en la ruta especificada.")

# ==============================
# Configuración del puerto serial
# ==============================
try:
    ser = serial.Serial('COM5', 9600, timeout=1)
    time.sleep(2)
except serial.SerialException as e:
    print("Error al abrir el puerto serial:", e)
    exit(1)

# ==============================
# Configuración UDP para Webots
# ==============================
UDP_IP = "127.0.0.1"
UDP_PORT = 5005
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# ==============================
# Cargar modelo entrenado
# ==============================
input_size = 2
output_size = 4
model = NavigationNet(input_features=input_size, num_classes=output_size)
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()

# ==============================
# Cargar escalador
# ==============================
scaler = joblib.load(SCALER_PATH)

# ==============================
# Funciones auxiliares
# ==============================
def leer_sensores():
    """Lee datos de sensores IR desde Arduino"""
    if ser.in_waiting > 0:
        linea = ser.readline().decode('utf-8').strip()
        try:
            datos = [float(x) for x in linea.split(',')]
            if len(datos) == 2:
                return np.array(datos).reshape(1, -1)
        except ValueError:
            print("Lectura inválida:", linea)
            return None
    return None

def enviar_comando_arduino(clase_predicha):
    """Envía comando al Arduino según la predicción"""
    comando_dict = {
        0: "AVANZAR\n",
        1: "REV_DERECHA\n",
        2: "REV_IZQUIERDA\n",
        3: "CENTRO\n"
    }
    comando = comando_dict.get(clase_predicha, "STOP\n")
    ser.write(comando.encode('utf-8'))
    print(f"[Arduino] Acción enviada: {comando.strip()}")

def enviar_comando_webots(clase_predicha):
    """Envía la acción a Webots vía UDP"""
    sock.sendto(str(clase_predicha).encode(), (UDP_IP, UDP_PORT))
    print(f"[Webots] Acción enviada: {clase_predicha}")

def fallback(sensores):
    """
    Función fallback de seguridad: si los sensores detectan obstáculo cercano,
    detener o retroceder, cambiar dirección según la lectura de los sensores.
    """
    dist_left, dist_right = sensores.flatten()
    umbral_cercano = 50  # Ajusta según tu sensor

    if dist_left < umbral_cercano and dist_right < umbral_cercano:
        return 3  # CENTRO / STOP
    elif dist_left < umbral_cercano:
        return 2  # REV_IZQUIERDA
    elif dist_right < umbral_cercano:
        return 1  # REV_DERECHA
    else:
        return 0  # AVANZAR

# ==============================
# Bucle principal con fallback y UDP
# ==============================
MAX_FALLOS_CONSECUTIVOS = 10
fallos_consecutivos = 0

print("Controlador iniciado. Esperando datos de sensores...")

try:
    while True:
        sensores = leer_sensores()
        if sensores is not None:
            entrada_escalada = scaler.transform(sensores)
            with torch.no_grad():
                salida = model(torch.tensor(entrada_escalada, dtype=torch.float32))
                clase_predicha_red = torch.argmax(salida, dim=1).item()
            
            clase_fallback = fallback(sensores)
            
            # Verificar predicción segura
            if clase_predicha_red != clase_fallback:
                fallos_consecutivos += 1
                print(f"Predicción insegura detectada. Fallos consecutivos: {fallos_consecutivos}")
            else:
                fallos_consecutivos = 0

            # Activar fallback si excede máximo de fallos
            if fallos_consecutivos >= MAX_FALLOS_CONSECUTIVOS:
                clase_predicha = clase_fallback
                print("**FALLBACK ACTIVADO**")
            else:
                clase_predicha = clase_predicha_red

            # Enviar comandos
            enviar_comando_arduino(clase_predicha)
            enviar_comando_webots(clase_predicha)

            print("Sensores:", sensores.flatten(),
                  "-> Clase red:", clase_predicha_red,
                  "-> Clase final:", clase_predicha)

        time.sleep(0.1)

except KeyboardInterrupt:
    print("Finalizando controlador...")
    ser.close()
except Exception as e:
    print("Error inesperado:", e)
    ser.close()








