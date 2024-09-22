
# Mini Curso: Detecção de Objetos e Faces - Ada Lovelace Day

Este repositório contém os códigos utilizados para um mini curso apresentado no **Ada Lovelace Day**, com foco na **detecção de objetos** e **detecção facial** utilizando diferentes algoritmos e técnicas, como **Haar Cascade**, **HOG + SVM**, **MTCNN**, e **YOLOv8**.

## Estrutura do Repositório

O repositório está organizado da seguinte forma:

```
├── src
│   └── detectors.py   # Implementação de detectores de face (Haarcascade, HOG, MTCNN, YOLOv8)
│   └── utils.py       # Funções auxiliares para manipulação de imagens e desenho de bounding boxes
├── notebooks/         # Laboratório Jupyter mostrando o funcionamento dos detectores
│   └── detectors_lab.ipynb
├── images/            # Imagens utilizadas nos exemplos e testes
├── weights/           # Diretório para armazenar os arquivos de pesos YOLOv8 (yolov8n.pt e yolov8n-face.pt)
├── demo_video.py      # Script principal que permite selecionar o algoritmo de detecção via linha de comando
├── requirements.txt   # Arquivo com as bibliotecas necessárias 
├── README.md          # Arquivo de documentação (este arquivo)
```

## Requisitos

- Python 3.7+
- Bibliotecas necessárias:
  - `opencv-python`
  - `dlib`
  - `torch`
  - `facenet-pytorch`
  - `ultralytics`
  - `matplotlib`
  - `jupyter`

Você pode instalar as dependências usando o seguinte comando:

```bash
pip install -r requirements.txt
```

## Detecção de Faces

O arquivo `detectors.py` contém implementações para diferentes abordagens de detecção facial, incluindo:

1. **Haar Cascade** (via OpenCV)
   - Baseado no artigo "Rapid Object Detection using a Boosted Cascade of Simple Features" de Viola e Jones (2001).
   - Referência: [OpenCV Haar Cascade](https://docs.opencv.org/4.x/db/d28/tutorial_cascade_classifier.html)

2. **HOG + SVM** (via dlib e face_recognition)
   - Baseado no uso de Histogram of Oriented Gradients (HOG) combinado com um classificador de Suporte Vetorial (SVM).
   - Utiliza a biblioteca `face_recognition` para simplificar a aplicação.
   - Referência: [Dlib HOG + SVM](http://dlib.net/) e [Face Recognition](https://github.com/ageitgey/face_recognition)

3. **MTCNN** (via `facenet-pytorch`)
   - Baseado na Multi-task Cascaded Convolutional Neural Networks (MTCNN) para detecção de faces.
   - Referência: [MTCNN](https://github.com/timesler/facenet-pytorch)

4. **YOLOv8** (via `ultralytics`)
   - You Only Look Once (YOLO) é uma abordagem baseada em redes neurais convolucionais para detecção de objetos em tempo real.
   - Referência: [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)

### Uso dos Detectores

Você pode escolher o algoritmo de detecção passando-o como argumento para o script principal `demo_video.py`. O script permite também selecionar qual webcam utilizar.

Exemplo de uso:

```bash
python demo_video.py --algorithm haarcascade --webcam_index 0
```

Onde:

- `--algorithm`: Define o algoritmo de detecção a ser usado. Pode ser:
  - `haarcascade`
  - `hog`
  - `mtcnn`
  - `yolo`
  - `yolo_object`
- `--webcam_index`: Define qual webcam será usada (padrão é `0`).

## Laboratório de Detectores

A pasta **`notebooks/`** contém um notebook Jupyter chamado `detectors_lab.ipynb`, que demonstra a aplicação prática dos detectores implementados.

Para rodar o notebook, siga estes passos:

1. Certifique-se de ter o **Jupyter Notebook** instalado. Se necessário, instale-o com:

   ```bash
   pip install jupyter
   ```

2. No diretório raiz do projeto, rode o seguinte comando para abrir o notebook:

   ```bash
   jupyter notebook notebooks/detectors_lab.ipynb
   ```

O notebook contém exemplos interativos de como utilizar os detectores e aplicar bounding boxes em imagens fornecidas na pasta **`images/`**.

## Manipulação de Imagens e Bounding Boxes

O arquivo `utils.py` contém funções utilitárias para manipulação de imagens, como:

- **`plot_image(image)`**: Exibe uma imagem usando `matplotlib` após convertê-la de BGR para RGB.
- **`insert_boundingboxes(boxes, image, method=None)`**: Insere bounding boxes em uma imagem, podendo utilizar dois formatos diferentes de coordenadas.
- **`yolo_bounding_boxes(image, boxes, confidences, class_ids)`**: Desenha as bounding boxes, classes e confidências para detecções YOLO.

### Exemplo de Uso

```python
from utils import plot_image, insert_boundingboxes, yolo_bounding_boxes

# Exibir uma imagem
plot_image(image)

# Inserir bounding boxes em uma imagem
frame_with_boxes = insert_boundingboxes(boundingboxes, image)

# Inserir bounding boxes, classes e confidências YOLO
frame_with_yolo_boxes = yolo_bounding_boxes(image, boxes, confidences, class_ids)
```

## Pesos para YOLOv8

Os arquivos de pesos para o YOLOv8 devem ser colocados na pasta **`weights/`**. Por padrão, o código está configurado para usar o arquivo `yolov8n-face.pt` para detecção de faces e `yolov8n.pt` para detecção de objetos. Esses arquivos podem ser baixados diretamente do [repositório oficial do YOLO](https://github.com/ultralytics/ultralytics).

## Contribuindo

Este repositório foi desenvolvido para um mini curso no **Ada Lovelace Day**. 
