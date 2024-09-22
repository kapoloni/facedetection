import cv2
import matplotlib.pyplot as plt


COCO_CLASSES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "TV",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


def plot_image(image):
    """
    Exibe uma imagem usando matplotlib após converter de BGR para RGB.

    Args:
        image (numpy.ndarray): A imagem que será exibida.
    """
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def insert_boundingboxes(boxes, image, method=None):
    """
    Insere bounding boxes em uma imagem e retorna a imagem com as caixas desenhadas.

    Args:
        boxes (list): Lista de coordenadas das caixas delimitadoras. Pode ser no formato (x, y, largura, altura)
                      ou (x1, y1, x2, y2) dependendo do método.
        image (numpy.ndarray): A imagem original onde as caixas serão desenhadas.
        method (str, optional): Define o método de desenho. Se for 'CNN', usa o formato (x1, y1, x2, y2).
                                Caso contrário, usa (x, y, largura, altura). O padrão é None.

    Returns:
        numpy.ndarray: A imagem com as bounding boxes desenhadas.
    """
    frame = image.copy()

    if boxes is not None:
        for box in boxes:
            if method == "CNN":
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            else:
                x, y, w, h = map(int, box)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return frame


def yolo_bounding_boxes(image, boxes, confidences, class_ids):
    """
    Desenha bounding boxes, confidências e classes para objetos detectados com YOLO.

    Args:
        image (numpy.ndarray): A imagem original onde as caixas serão desenhadas.
        boxes (list): Lista de coordenadas das caixas delimitadoras no formato (x_min, y_min, x_max, y_max).
        confidences (list): Lista de confidências para cada detecção.
        class_ids (list): IDs das classes detectadas, que serão mapeadas para seus nomes usando COCO_CLASSES.

    Returns:
        numpy.ndarray: A imagem com as bounding boxes, confidências e classes desenhadas.
    """
    frame = image.copy()

    for box, confidence, class_id in zip(boxes, confidences, class_ids):
        x_min, y_min, x_max, y_max = map(int, box)
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"Conf: {confidence:.2f} Class: {COCO_CLASSES[int(class_id)]}",
            (x_min, y_min - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    return frame
