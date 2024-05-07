"""
Jose Miguel Gonzalez Zaragoza
631145 IRSI 6to Semestre
May 2024
COMPUTER-VISION
UDEM

USING BRIEF AS A WAY TO DETECT MOVING OBJECTS

HOW TO USE IT:
python object_tracking.py --img_obj .\(YOUR IMAGE AND FORMAT ON THE FOLDER)


"""




import numpy as np
import cv2 as cv
import argparse as arg
import time

def read_parser():
    """
    Lee los argumentos de la línea de comandos y los analiza.

    Returns:
        Namespace: El objeto que contiene los argumentos analizados.
    """
    parser = arg.ArgumentParser(description="Program for feature detection and matching with BRIEF descriptor using STAR detector")
    parser.add_argument("--img_obj", 
                        dest="train_path", 
                        type=str, 
                        help="Path to train image")
    args = parser.parse_args()
    return args

def brief_descriptor(img):
    """
    Calcula los keypoints y descriptores BRIEF de una imagen.

    Args:
        img (numpy.ndarray): La imagen de entrada.

    Returns:
        tuple: Una tupla que contiene los keypoints y descriptores.
    """
    star = cv.xfeatures2d.StarDetector_create()
    brief = cv.xfeatures2d.BriefDescriptorExtractor_create()
    kp = star.detect(img, None)
    kp, des = brief.compute(img, kp)
    return kp, des 

def match_images(query_img, train_img):
    """
    Encuentra los matches entre los descriptores de dos imágenes.

    Args:
        query_img (numpy.ndarray): La imagen de consulta.
        train_img (numpy.ndarray): La imagen de entrenamiento.

    Returns:
        tuple: Una tupla que contiene la imagen con los matches y los puntos correspondientes.
    """
    pts = np.array([])  # Inicializar pts con un array vacío

    kp1, des1 = brief_descriptor(query_img)
    kp2, des2 = brief_descriptor(train_img)

    # Convertir descriptores a tipo np.float32
    des1 = np.float32(des1)
    des2 = np.float32(des2)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)
    good_matches = []

    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    img_matches = cv.drawMatches(query_img, kp1,
                                 train_img, kp2, 
                                 good_matches, 
                                 None, 
                                 flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    # Dibujar polígono basado en los tres matches más cercanos
    if len(good_matches) >= 3:
        pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        cv.polylines(img_matches, [np.int32(pts)], True, (0, 0, 255), 2)

    return img_matches, pts

def find_centroid_and_location(frame, pts):
    """
    Encuentra el centroide de un polígono y determina su ubicación.

    Args:
        frame (numpy.ndarray): La imagen donde se encuentra el polígono.
        pts (numpy.ndarray): Los puntos que forman el polígono.

    Returns:
        tuple: Una tupla que contiene las coordenadas del centroide y la ubicación del polígono.
    """
    # Encontrar el centroide del polígono
    M = cv.moments(pts)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        centroid = (cx, cy)
    else:
        centroid = None

    # Determinar si el centroide está a la izquierda o a la derecha de la línea vertical
    if centroid is not None:
        if centroid[0] < frame.shape[1] // 2:
            location = "Izquierda"
        else:
            location = "Derecha"
    else:
        location = None

    return centroid, location

def initialize_camera():
    """
    Inicializa la cámara y comprueba si está disponible.

    Returns:
        cv2.VideoCapture or None: El objeto de captura de video de la cámara o None si no se puede inicializar.
    """
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to initialize camera.")
        return None
    return cap

def display_matches_window(img):
    """
    Muestra una imagen en la ventana "Matches".

    Args:
        img (numpy.ndarray): La imagen a mostrar.
    """
    cv.imshow("Matches", img)
    cv.waitKey(1)

def display_video_window(img):
    """
    Muestra una imagen en la ventana "Video".

    Args:
        img (numpy.ndarray): La imagen a mostrar.
    """
    cv.imshow("Video", img)
    cv.waitKey(1)

def process_frame(cap, train_img):
    """
    Captura un fotograma de la cámara y procesa la imagen.

    Args:
        cap (cv2.VideoCapture): El objeto de captura de video de la cámara.
        train_img (numpy.ndarray): La imagen de entrenamiento.

    Returns:
        tuple: Una tupla que contiene el fotograma, la imagen con los matches y los puntos correspondientes.
    """
    ret, frame = cap.read()
    if not ret:
        print("Error capturing frame from camera")
        return None, None

    query_img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    matches_img, pts = match_images(query_img, train_img)

    return frame, matches_img, pts

def draw_matches(frame, pts, train_img):
    """
    Dibuja los matches encontrados en la imagen.

    Args:
        frame (numpy.ndarray): La imagen en la que se dibujarán los matches.
        pts (numpy.ndarray): Los puntos correspondientes a los matches.
        train_img (numpy.ndarray): La imagen de entrenamiento.

    Returns:
        numpy.ndarray: La imagen con los matches dibujados.
    """
    if len(pts) > 0:
        kp1, des1 = brief_descriptor(frame)
        kp2, des2 = brief_descriptor(train_img)
        des1 = np.float32(des1)
        des2 = np.float32(des2)

        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(des1, des2, k=2)
        good_matches = []

        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)

        if len(good_matches) >= 3:
            pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            cv.polylines(frame, [np.int32(pts)], True, (0, 0, 255), 2)
    return frame

def draw_centroid(frame, pts):
    """
    Dibuja el centroide del polígono en la imagen.

    Args:
        frame (numpy.ndarray): La imagen en la que se dibujará el centroide.
        pts (numpy.ndarray): Los puntos que forman el polígono.

    Returns:
        tuple: Una tupla que contiene las coordenadas del centroide, la ubicación del polígono y la imagen con el centroide dibujado.
    """
    centroid, location = find_centroid_and_location(frame, pts) if len(pts) >= 3 else (None, None)
    if centroid is not None:
        cv.rectangle(frame, (centroid[0] - 37, centroid[1] - 37), (centroid[0] + 37, centroid[1] + 37), (255, 0, 0), 2)
    return centroid, location, frame

def update_location(prev_location, current_location, location):
    """
    Actualiza el estado de la ubicación del polígono.

    Args:
        prev_location (str or None): La ubicación anterior del polígono.
        current_location (str or None): La ubicación actual del polígono.
        location (str or None): La ubicación a actualizar.

    Returns:
        tuple: Una tupla que contiene la ubicación anterior y la ubicación actualizada.
    """
    if location != current_location:
        prev_location = current_location
        current_location = location
    return prev_location, current_location

def show_location_message(frame, current_location):
    """
    Muestra un mensaje con la ubicación del polígono en la imagen.

    Args:
        frame (numpy.ndarray): La imagen en la que se mostrará el mensaje.
        current_location (str or None): La ubicación actual del polígono.

    Returns:
        numpy.ndarray: La imagen con el mensaje de ubicación.
    """
    if current_location is not None:
        cv.putText(frame, current_location, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)
    return frame

def main():
    """
    Función principal del programa.
    """
    args = read_parser()
    train_img = cv.imread(args.train_path, cv.IMREAD_GRAYSCALE)

    if train_img is None:
        print("Train image not available")
        return
    
    cap = initialize_camera()
    if cap is None:
        return
    
    cv.namedWindow("Matches", cv.WINDOW_NORMAL)
    cv.resizeWindow("Matches", 800, 600)
    cv.namedWindow("Video", cv.WINDOW_NORMAL)
    cv.resizeWindow("Video", 800, 600)

    prev_location = None
    current_location = None

    while True:
        frame, matches_img, pts = process_frame(cap, train_img)
        if frame is None:
            break

        display_matches_window(matches_img)
        frame = draw_matches(frame, pts, train_img)
        cv.line(frame, (frame.shape[1] // 2, 0), (frame.shape[1] // 2, frame.shape[0]), (0, 255, 0), 2)
        centroid, location, frame = draw_centroid(frame, pts)
        prev_location, current_location = update_location(prev_location, current_location, location)
        frame = show_location_message(frame, current_location)
        display_video_window(frame)

        key = cv.waitKey(1)
        if key == 27:
            break

        time.sleep(0.05)

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
