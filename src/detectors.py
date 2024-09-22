import cv2
import dlib
from facenet_pytorch import MTCNN
from ultralytics import YOLO


class HaarcascadeDetector:
    """
    Implementa detecção facial usando o algoritmo Haar Cascade do OpenCV.
    """

    def __init__(self):
        """
        Inicializa o detector Haar Cascade com o modelo pré-treinado
        'haarcascade_frontalface_default.xml'.
        """
        self.detector = cv2.CascadeClassifier(
            f"{cv2.data.haarcascades}haarcascade_frontalface_default.xml"
        )

    def detect(self, image):
        """
        Detecta faces em uma imagem usando o algoritmo Haar Cascade.

        Args:
            image (numpy.ndarray): Imagem na qual as faces serão detectadas.

        Returns:
            list: Lista de bounding boxes (x, y, largura, altura) para as faces detectadas.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        boxes = self.detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        return boxes


class MTCNNDetector:
    """
    Implementa detecção facial usando a Multi-task Cascaded Convolutional Neural Networks (MTCNN).
    """

    def __init__(self):
        """
        Inicializa o detector MTCNN.
        """
        self.detector = MTCNN(keep_all=True)

    def detect(self, image):
        """
        Detecta faces em uma imagem usando MTCNN.

        Args:
            image (numpy.ndarray): Imagem na qual as faces serão detectadas.

        Returns:
            list: Lista de bounding boxes (x_min, y_min, x_max, y_max) para as faces detectadas.
        """
        boxes, _ = self.detector.detect(image)
        return boxes


class HOGDetector:
    """
    Implementa detecção facial usando o algoritmo HOG + SVM (Histogram of Oriented Gradients) do dlib.
    """

    def __init__(self):
        """
        Inicializa o detector HOG + SVM do dlib.
        """
        self.detector = dlib.get_frontal_face_detector()

    def detect(self, image):
        """
        Detecta faces em uma imagem usando HOG + SVM.

        Args:
            image (numpy.ndarray): Imagem na qual as faces serão detectadas.

        Returns:
            list: Lista de bounding boxes (x, y, largura, altura) para as faces detectadas.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray, 1)
        boxes = [
            [face.left(), face.top(), face.width(), face.height()] for face in faces
        ]
        return boxes


class YoloDetector:
    """
    Implementa detecção facial usando o modelo YOLOv8, um detector de objetos
    baseado em redes neurais convolucionais.
    """

    def __init__(self, weight="../weights/yolov8n-face.pt"):
        """
        Inicializa o detector YOLOv8 com um modelo pré-treinado específico para detecção de faces.

        Args:
            weight (str): Caminho para o arquivo de pesos do modelo YOLOv8.
        """
        self.detector = YOLO(weight)

    def detect(self, image, conf_thres=0.25, iou_thres=0.46):
        """
        Detecta faces em uma imagem usando YOLOv8.

        Args:
            image (numpy.ndarray): Imagem na qual as faces serão detectadas.
            conf_thres (float): Limiar de confiança mínima para a detecção.
            iou_thres (float): Limiar de IoU para a Non-Maximum Suppression (NMS).

        Returns:
            tuple: Uma tupla contendo:
                - boxes (numpy.ndarray): Coordenadas das bounding boxes (x_min, y_min, x_max, y_max).
                - confidences (numpy.ndarray): Confiança associada a cada detecção.
                - class_ids (numpy.ndarray): IDs das classes detectadas.
        """
        results = self.detector(image, conf=conf_thres, iou=iou_thres)

        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()  # Coordenadas das caixas
            confidences = result.boxes.conf.cpu().numpy()  # Confiança das detecções
            class_ids = result.boxes.cls.cpu().numpy()  # Classes detectadas

        return boxes, confidences, class_ids
