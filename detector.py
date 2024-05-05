import cv2 as cv
import numpy as np
import argparse as arg

def read_parser():
    parser = arg.ArgumentParser(description="Program for feature detection and matching with BRIEF descriptor using STAR detector")
    parser.add_argument("--train_image", dest="train_path", type=str, help="Path to train image")
    args = parser.parse_args()
    return args

def load_train_image(train_path):
    train_img = cv.imread(train_path, cv.IMREAD_GRAYSCALE)
    if train_img is None:
        print("Train image not available")
    return train_img

def initialize_camera(train_path):
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to initialize camera.")
        return None, None
    train_img = load_train_image(train_path)
    if train_img is None:
        return None, None
    return cap, train_img

def brief_descriptor(img):
    star = cv.xfeatures2d.StarDetector_create()
    brief = cv.xfeatures2d.BriefDescriptorExtractor_create()
    kp = star.detect(img, None)
    kp, des = brief.compute(img, kp)
    print(brief.descriptorSize())
    print(des.shape)
    return kp, des 

def match_images(query_img, train_img):
    kp1, des1 = brief_descriptor(query_img)
    des1 = np.float32(des1)
    kp2, des2 = brief_descriptor(train_img)
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
    img_matches = cv.drawMatches(query_img, kp1, train_img, kp2, good_matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv.namedWindow("Matches", cv.WINDOW_NORMAL)  
    cv.resizeWindow("Matches", 1000, 800)  
    cv.imshow("Matches", img_matches) 
    cv.waitKey(0)
    cv.destroyAllWindows()

def main():
    args = read_parser()
    cap, train_img = initialize_camera(args.train_path)
    if cap is None or train_img is None:
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error capturing frame from camera")
            break
        
        query_img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        match_images(query_img, train_img)

        key = cv.waitKey(1)
        if key == 27:  # ESC key code
            break

    cap.release()

if __name__ == "__main__":
    main()
