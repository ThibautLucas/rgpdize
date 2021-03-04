from PIL import Image
import os
import numpy as np
import cv2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow-gpu as tf
tf.get_logger().setLevel('ERROR')   


def blur(img, bbox, shape="rectangle"):
    
    imagette = img[bbox[0]:bbox[2], bbox[1]:bbox[3]]
    print(imagette.shape)
    w, h = imagette.shape[:2]
    
    to_blur = imagette[0:int(0.7*w), :, :]
    print(to_blur.shape)
    blur = cv2.medianBlur(to_blur,55)
    
    imagette[0:int(0.7*w), :, :] = blur
    
    img[bbox[0]:bbox[2], bbox[1]:bbox[3]] = imagette
    return img

def anonymize(face_detector, imgpath, output_dir, threshold=0.5):
    
    im = cv2.imread(imgpath)
    input_tensor = tf.convert_to_tensor(im)
    input_tensor = input_tensor[tf.newaxis, ...]
    detections = face_detector(input_tensor)

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                   for key, value in detections.items()}
    
    detections['num_detections'] = num_detections
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
    (w, h) = im.shape[:2]
    im1 = im.copy()
    for bbox, conf in list(zip(detections["detection_boxes"], detections["detection_scores"])):
        if conf > threshold:
            bbox = bbox * np.array([w, h, w, h])
            bbox = bbox.astype("int")
            im = blur(im, bbox)
            
    output_path = os.path.join(output_dir,imgpath.split('/')[-1])
    cv2.imwrite(os.path.join(output_dir, imgpath.split('/')[-1]), im)
    return 


if __name__ == "__main__":
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        print(gpu)
        tf.config.experimental.set_memory_growth(gpu, True)

    face_detector = tf.saved_model.load('saved_model')

    anonymize(face_detector, 'inputs/face1.jpg', 'outputs')