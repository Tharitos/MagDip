import os
import cv2
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from pixellib.instance import instance_segmentation

# config = tf.compat.v1.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.9  # 0.6 sometimes works better for folks
# keras.backend.tensorflow_backend.set_session(tf.Session(config=config))

score_limit = 0.98

segment_image = instance_segmentation()
segment_image.load_model("ves.h5")

target_class = segment_image.select_target_classes(person=True)


def object_detection_on_an_image(image_path, all_objects):

    os.makedirs('out', exist_ok=True)
    os.makedirs('objects', exist_ok=True)

    bboxes, out = segment_image.segmentImage(
        image_path=image_path,
        show_bboxes=True,
        segment_target_classes=target_class,
        # extract_segmented_objects=True,
        # save_extracted_objects=True,
        output_image_name=os.path.join('out', f"{os.path.basename(image_path)}")
    )
    for i, bbox in enumerate(bboxes['rois']):
        if bboxes['scores'][i] < score_limit:
            continue
        bbox_data = {}
        bbox_data['x'] = (int(bbox[1]) + int(bbox[3])) // 2
        bbox_data['y'] = (int(bbox[0]) + int(bbox[2])) // 2
        bbox_data['frame'] = image_path

        img = cv2.imread(image_path)
        cropped_image = img[bbox[0]:bbox[2], bbox[1]:bbox[3]]
        cv2.imwrite(f"objects/{i}_{os.path.basename(image_path)}", cropped_image)
        bbox_data['image_path'] = f"objects/{i}_{os.path.basename(image_path)}"
        bbox_data['cls'] = -1
        all_objects.append(bbox_data)
    return all_objects