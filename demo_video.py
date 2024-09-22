import cv2
import sys
import argparse
from src.detectors import HaarcascadeDetector, HOGDetector, MTCNNDetector, YoloDetector
from src.utils import yolo_bounding_boxes, insert_boundingboxes


def get_detector(algorithm):
    """
    Retorna o detector de faces apropriado com base no algoritmo selecionado.

    Args:
        algorithm (str): O nome do algoritmo a ser utilizado.
        Pode ser 'haarcascade', 'hog', 'mtcnn', 'yolo', ou 'yolo_object'.

    Returns:
        object: O detector de faces correspondente ao algoritmo escolhido.
    """
    if algorithm == "haarcascade":
        return HaarcascadeDetector()
    elif algorithm == "hog":
        return HOGDetector()
    elif algorithm == "mtcnn":
        return MTCNNDetector()
    elif algorithm == "yolo":
        return YoloDetector(weight="weights/yolov8n-face.pt")
    elif algorithm == "yolo_object":
        return YoloDetector(weight="weights/yolov8n.pt")
    else:
        print("Algoritmo inválido. Use: haarcascade, hog, mtcnn, yolo ou yolo_object")
        sys.exit(1)


def main():
    """
    Função principal que configura o argparse para capturar argumentos
    de linha de comando e
    executa o detector de faces escolhido, exibindo os resultados
    em uma janela de vídeo.

    Os argumentos esperados incluem:
    - `--algorithm`: O algoritmo de detecção a ser usado
    (haarcascade, hog, mtcnn, yolo, yolo_object).
    - `--webcam_index`: O índice da webcam a ser utilizado (padrão é 1).

    A função também inclui a lógica para acessar a webcam,
    aplicar o detector de faces e
    exibir as detecções até que o usuário pressione a tecla 'q' para sair.
    """
    # Configura o argparse para capturar os argumentos da linha de comando
    parser = argparse.ArgumentParser(
        description="Seleção de algoritmo de detecção e webcam."
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        choices=["haarcascade", "hog", "mtcnn", "yolo", "yolo_object"],
        help="O algoritmo de detecção a ser usado.",
    )
    parser.add_argument(
        "--webcam_index",
        type=int,
        help="O índice da webcam a ser utilizada.",
        default=1,
    )

    args = parser.parse_args()

    # Acessa a webcam com o índice escolhido
    cap = cv2.VideoCapture(args.webcam_index)

    if not cap.isOpened():
        print(f"Erro ao acessar a webcam com índice {args.webcam_index}.")
        sys.exit(1)

    # Obter o detector com base no algoritmo escolhido
    detector = get_detector(args.algorithm)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Falha ao capturar imagem")
            break

        # Se for YOLO, executa a lógica específica para YOLO
        if "yolo" in args.algorithm:
            boxes, confidences, class_ids = detector.detect(
                frame, conf_thres=0.25, iou_thres=0.46
            )
            face_detected = yolo_bounding_boxes(frame, boxes, confidences, class_ids)
        else:
            # Detecta faces usando o algoritmo escolhido
            boundingboxes = detector.detect(frame)
            if args.algorithm == "mtcnn":
                face_detected = insert_boundingboxes(boundingboxes, frame, method="CNN")
            else:
                face_detected = insert_boundingboxes(boundingboxes, frame)

        # Exibe o frame com as detecções
        cv2.imshow("Face Detection", face_detected)

        # Interrompe o loop se a tecla 'q' for pressionada
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Libera a câmera e destrói todas as janelas criadas pelo OpenCV
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


# python -m demo_video --algorithm haarcascade --webcam_index 1
# python -m demo_video --algorithm hog --webcam_index 1
# python -m demo_video --algorithm mtcnn --webcam_index 1
# python -m demo_video --algorithm yolo --webcam_index 1
# python -m demo_video --algorithm yolo_object --webcam_index 1
