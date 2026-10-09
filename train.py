from ultralytics import YOLO

# Cargar el modelo base para aprovechar lo que ya sabe de bordes y formas
model = YOLO('yolov8n.pt')

# Iniciar el entrenamiento (device=0 fuerza el uso de tu tarjeta NVIDIA)
# epochs=50 indica cuántas veces repasará el dataset completo para aprender
if __name__ == '__main__':
    model.train(data='data.yaml', epochs=50, imgsz=640, device=0)