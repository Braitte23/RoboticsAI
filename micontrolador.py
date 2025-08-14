from controller import Robot, Motor
import socket
import struct

# ==============================
# Configuración robot Webots
# ==============================
TIME_STEP = 64  # ms

robot = Robot()

# Nombre de los motores según tu robot en Webots
motor_names = ["wheel1", "wheel2", "wheel3", "wheel4"]
motors = []

for name in motor_names:
    motor = robot.getDevice(name)
    if motor is None:
        print(f"ERROR: motor '{name}' no encontrado en el robot de Webots.")
    else:
        motor.setPosition(float('inf'))  # Modo velocidad
        motor.setVelocity(0.0)
    motors.append(motor)

# ==============================
# Configuración socket para recibir comandos
# ==============================
HOST = "127.0.0.1"  # localhost
PORT = 5005         # puerto donde tu controlador envía comandos

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.bind((HOST, PORT))
s.setblocking(False)  # No bloquea si no hay datos

# ==============================
# Función para mapear acción a velocidad
# ==============================
MAX_SPEED = 6.28  # ajusta según tu motor en Webots

def accion_a_velocidad(accion):
    """
    0: AVANZAR
    1: REV_DERECHA
    2: REV_IZQUIERDA
    3: CENTRO/STOP
    """
    if accion == 0:  # AVANZAR
        return MAX_SPEED, MAX_SPEED
    elif accion == 1:  # REV_DERECHA
        return -MAX_SPEED*0.5, -MAX_SPEED
    elif accion == 2:  # REV_IZQUIERDA
        return -MAX_SPEED, -MAX_SPEED*0.5
    elif accion == 3:  # CENTRO / STOP
        return 0.0, 0.0
    else:
        return 0.0, 0.0

# ==============================
# Bucle principal
# ==============================
print("Controlador Webots iniciado. Esperando comandos...")

while robot.step(TIME_STEP) != -1:
    # Recibir comando desde Python
    try:
        data, addr = s.recvfrom(1024)
        accion = int(data.decode("utf-8").strip())
    except BlockingIOError:
        continue
    except Exception as e:
        print("Error recibiendo datos:", e)
        continue

    vel_left, vel_right = accion_a_velocidad(accion)

    # Mapear 2 motores reales → 4 motores simulados
    if all(motors):
        motors[0].setVelocity(vel_left)   # left_front
        motors[2].setVelocity(vel_left)   # left_rear
        motors[1].setVelocity(vel_right)  # right_front
        motors[3].setVelocity(vel_right)  # right_rear

    print(f"Accion recibida: {accion} -> vel_left: {vel_left:.2f}, vel_right: {vel_right:.2f}")



