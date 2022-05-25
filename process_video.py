import cv2
import os
# import tensorflow as tf
# print(tf.test.is_gpu_available())
# print(tf.test.is_built_with_cuda())
from copy import deepcopy, copy

import tensorflow as tf


config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
import numpy as np
from scipy.spatial import distance
from kalmanfilter import KalmanFilter
from object_detection import object_detection_on_an_image

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
from vgg_19 import VGG_19

model = VGG_19(weights='imagenet', include_top=False)

sim_limit = 0.25
frame_step = 10


def compare(img_1, img_2):
    height, width, channels = img_1.shape
    img_1 = np.expand_dims(img_1, axis=0)
    img_1 = preprocess_input(img_1)
    features_1 = model.predict(img_1).flatten()
    img_2 = cv2.resize(img_2, (width, height))
    img_2 = np.expand_dims(img_2, axis=0)
    img_2 = preprocess_input(img_2)
    features_2 = model.predict(img_2).flatten()
    return 1.0 - distance.cosine(features_1.flatten(), features_2.flatten())


def image_to_features(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    features = np.array(model.predict(img_data))
    return np.array(features.flatten())


def video_to_frames(path):
    os.makedirs('frames', exist_ok=True)

    cap = cv2.VideoCapture(path)
    i = 0
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        if i % frame_step == 0:
            cv2.imwrite('frames/' + str(i) + '.jpg', frame)
        i += 1

    cap.release()
    cv2.destroyAllWindows()


def process_video(video_path):
    all_objects = []
    video_to_frames(video_path)
    for frame in os.listdir("frames"):
        all_objects=object_detection_on_an_image(os.path.join("frames", frame), all_objects)

    cur_cl = 0
    colors = {}
    for obj in all_objects:
        print(obj['x'])
        if obj['cls'] != -1:
            continue
        obj['cls'] = cur_cl
        for obj2 in all_objects:
            if obj2['cls'] != -1:
                continue
            im1 = cv2.imread(obj['image_path'])
            im2 = cv2.imread(obj2['image_path'])
            sim = compare(im1, im2)
            print(sim)
            if sim >= sim_limit:
                obj2['cls'] = cur_cl
        colors[cur_cl] = list(np.random.random(size=3) * 256)
        print(colors[0])
        cur_cl += 1

    for obj in all_objects:
        img = cv2.imread(obj['frame'])
        img = cv2.circle(img, (obj['x'], obj['y']), 7, colors[obj['cls']], 7)
        cv2.imwrite(obj['frame'], img)

    os.makedirs('classes', exist_ok=True)

    for obj in all_objects:
        print(obj['image_path'])
        os.makedirs(os.path.join('classes', str(obj['cls'])), exist_ok=True)
        img = cv2.imread(obj['image_path'])
        cv2.imwrite(os.path.join('classes', str(obj['cls']), os.path.basename(obj['image_path'])), img)

    pnts = []

    for i in range(len(all_objects)):
        for j in range(i + 1, len(all_objects)):
            if all_objects[i]['cls'] == all_objects[j]['cls'] and not all_objects[i]['frame'] == all_objects[j][
                'frame']:

                frame_number = int(os.path.splitext(os.path.basename(all_objects[i]['frame']))[0])
                frame_number2 = int(os.path.splitext(os.path.basename(all_objects[j]['frame']))[0])
                if frame_number < frame_number2 and frame_number2 - frame_number < frame_step + 2:
                    pnts.append(
                        [all_objects[i]['x'], all_objects[i]['y'], all_objects[j]['x'], all_objects[j]['y']])
                    img = cv2.imread(all_objects[i]['frame'])
                    for pn in pnts:
                        img = cv2.arrowedLine(img, (pn[0], pn[1]),
                                              (pn[2], pn[3]),
                                              colors[all_objects[i]['cls']], 5)
                    cv2.imwrite(all_objects[i]['frame'], img)

    positions = {}

    for i in range(len(all_objects)):
        if all_objects[i]['cls'] not in positions:
            positions[all_objects[i]['cls']] = [(all_objects[i]['x'], all_objects[i]['y'])]
        else:
            positions[all_objects[i]['cls']].append((all_objects[i]['x'], all_objects[i]['y']))

        kf = KalmanFilter()
        for pt in positions[all_objects[i]['cls']]:
            predicted = kf.predict(pt[0], pt[1])

        predicted = kf.predict(predicted[0], predicted[1])
        img = cv2.imread(all_objects[i]['frame'])
        img = cv2.arrowedLine(img, (all_objects[i]['x'], all_objects[i]['y']),
                              (predicted[0], predicted[1]),
                              (0, 0, 255), 5)
        cv2.imwrite(all_objects[i]['frame'], img)




def process_stream():
    all_objects = []
    cur_class = 0
    colors = {}
    filters = {}
    last_point = {}
    cur_point = {}

    os.makedirs('frames', exist_ok=True)
    cap = cv2.VideoCapture(0)#path если видео
    i = 0
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        if i % frame_step == 0:
            cv2.imwrite('frames/' + str(i) + '.jpg', frame)
            prev_objects = copy(all_objects)
            all_objects.clear()
            all_objects=object_detection_on_an_image('frames/' + str(i) + '.jpg', all_objects)
            if i == 0:
                for obj in all_objects:
                    obj['cls'] = cur_class
                    filters[cur_class] = KalmanFilter()
                    predicted = filters[cur_class].predict(obj['x'], obj['y'])
                    cur_point[cur_class] = (obj['x'], obj['y'])
                    cur_class += 1
            else:
                for obj in prev_objects:
                    mx = 0
                    ind = 0
                    for j, obj2 in enumerate(all_objects):
                        if obj2['cls'] != -1:
                            continue
                        im1 = cv2.imread(obj['image_path'])
                        im2 = cv2.imread(obj2['image_path'])
                        sim = compare(im1, im2)
                        print(sim)
                        if sim > mx:
                            mx = sim
                            ind = j
                    if mx >= sim_limit:
                        all_objects[ind]['cls'] = obj['cls']
                        last_point[obj['cls']] = cur_point[obj['cls']]
                        cur_point[obj['cls']] = (all_objects[ind]['x'], all_objects[ind]['y'])
                        if not obj['cls'] in colors:
                            colors[obj['cls']] = list(np.random.random(size=3) * 256)
                        frame = cv2.arrowedLine(frame, last_point[obj['cls']],
                                                cur_point[obj['cls']],
                                                colors[obj['cls']], 5)
                        predicted = filters[obj['cls']].predict(all_objects[ind]['x'], all_objects[ind]['y'])
                        cur_filter = deepcopy(filters[obj['cls']])
                        predicted = cur_filter.predict(predicted[0], predicted[1])
                        frame = cv2.arrowedLine(frame, cur_point[obj['cls']],
                                                predicted,
                                                (0, 0, 255), 5)
                for obj in all_objects:
                    if obj['cls'] != -1:
                        continue
                    obj['cls'] = cur_class
                    filters[cur_class] = KalmanFilter()
                    predicted = filters[cur_class].predict(obj['x'], obj['y'])
                    cur_point[cur_class] = (obj['x'], obj['y'])
                    cur_class += 1
            cv2.imwrite('frames/' + str(i) + '.jpg', frame)
            frame = cv2.resize(frame, (960, 540))
            cv2.namedWindow("frame", cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty("frame", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow('frame', frame)
            cv2.waitKey(10)
        i += 1

    cap.release()
    cv2.destroyAllWindows()
