import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib
import os

# ==============================
# Rutas de archivos
# ==============================
csv_file = r"C:\Users\brait\OneDrive\Documentos\panda\rutas\datasetrutas_limpio.csv"
output_dir = r"C:\Users\brait\OneDrive\Escritorio\Escirotio 2\IAccd"

# Crear carpeta de salida si no existe
os.makedirs(output_dir, exist_ok=True)

# ==============================
# Cargar dataset
# ==============================
try:
    df = pd.read_csv(csv_file)
    print("Dataset cargado exitosamente.")
    print(f"Ruta del archivo: {csv_file}")
    print(f"Número total de registros: {len(df)}")
except FileNotFoundError:
    print(f"Error: No se encontró el archivo en la ruta especificada: {csv_file}")
    exit()

# Mapear códigos de dirección a etiquetas numéricas
action_codes = sorted(df['direction_code'].unique())
action_map = {code: i for i, code in enumerate(action_codes)}
df['action_label'] = df['direction_code'].map(action_map)

print("\nAcciones identificadas (direction_code):", action_codes)
print("Mapeo de acciones a etiquetas:", action_map)

# ==============================
# Preparación de datos
# ==============================
features = ['distance_left', 'distance_right']
X = df[features].values
y = df['action_label'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_tensor = torch.FloatTensor(X_train_scaled)
X_test_tensor = torch.FloatTensor(X_test_scaled)
y_train_tensor = torch.LongTensor(y_train)
y_test_tensor = torch.LongTensor(y_test)

print(f"\nDatos listos para el entrenamiento:")
print(f" - {len(X_train)} muestras de entrenamiento")
print(f" - {len(X_test)} muestras de prueba")

# ==============================
# Definición del modelo
# ==============================
class NavigationNet(nn.Module):
    def __init__(self, input_features, num_classes):
        super(NavigationNet, self).__init__()
        self.fc1 = nn.Linear(input_features, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

input_size = X_train.shape[1]
output_size = len(action_codes)
model = NavigationNet(input_features=input_size, num_classes=output_size)

print("\nArquitectura del Modelo:")
print(model)

# ==============================
# Entrenamiento
# ==============================
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 50
print("\nIniciando el entrenamiento del modelo...")

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

print("Entrenamiento finalizado.")

# ==============================
# Evaluación
# ==============================
model.eval()
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    _, predicted = torch.max(test_outputs.data, 1)
    total = y_test_tensor.size(0)
    correct = (predicted == y_test_tensor).sum().item()
    accuracy = 100 * correct / total
    print(f'\nPrecisión en el conjunto de prueba: {accuracy:.2f}%')

# ==============================
# Guardar modelo y escalador
# ==============================
model_path = os.path.join(output_dir, 'robot_navigation_model.pth')
scaler_path = os.path.join(output_dir, 'scaler.pkl')

torch.save(model.state_dict(), model_path)
joblib.dump(scaler, scaler_path)

print(f"\nModelo guardado en: {model_path}")
print(f"Escalador guardado en: {scaler_path}")
