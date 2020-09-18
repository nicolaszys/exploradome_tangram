import os
# comment out below line to enable tensorflow outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from core.output import output_box
from core.output import sort_by
from core.output import filter_game_areas
from core.output import format_game_areas
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
import numpy as np
import operator
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

flags.DEFINE_string('weights', './checkpoints/yolov4-416','path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('video', './data/video/video.mp4', 'path to input video or set to 0 for webcam')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.00, 'score threshold') # we put it to zero to dectect all classes 
flags.DEFINE_boolean('dont_show', False, 'dont show video output')

#####  TangrIAm custom flags
flags.DEFINE_string('classes', './data/classes/custom.names', 'path to classes file')
flags.DEFINE_integer('num_classes', 12, 'number of classes in the model')
flags.DEFINE_string('output_file', './detections/', 'path to output file containing object coordinates and classe predictions') # {'object_position':{'x1': coord_1, 'y1':coord_2, 'x2':coord_3,'y2':coord_4}, 'object_prediction':{'class_1':proba_1, 'class_2':proba_2, etc...}}
flags.DEFINE_integer('sort_by', 5, 'if sort_by Flag is true then probabilities will be sorted by descending order. The number of returned classes is specified by the  max_classes flag')
flags.DEFINE_integer('num_objects', 2, 'number of  detected objects')
flags.DEFINE_integer('margin', 4, ' number of bounding box object corner points (soft-->  k<4 or hard margin --> k=4) that have to belong to the game area in order to be selected ')
flags.DEFINE_string('areas', 'game_area_coords.txt', 'path to game area coordinates .txt file')

def main(_argv):
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size
    video_path = FLAGS.video
    k = FLAGS.margin  
    max_classes = FLAGS.sort_by
    num_objects= FLAGS.num_objects
    game_area_coords_path = FLAGS.areas

    # game areas in normalized format (they will be formatted later in the code according to image shape)
    game_areas=[]
    with open(game_area_coords_path) as file:
        lines=file.readlines()[1:]
        for line in lines:  
            number_string=line.strip()
            number_string = number_string.split(',')
            number_list = [float(i) for i in number_string]
            game_areas.append(number_list)
    print('game_areas')
    print(game_areas)

    
    #### saved file path and name
    save_file = open(FLAGS.output_file + 'boxes_and_predictions.txt', 'w')
    save_file_2 = open(FLAGS.output_file + 'area_boxes_and_predictions.txt', 'w')

    ### classe list
    class_names = [c.strip() for c in open(FLAGS.classes).readlines()] # ie: ['bateau', 'bol', 'chat', 'coeur',cygne', 'lapin', 'marteau','maison', 'montagne','pon','renard' ,'tortue']
    

    saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
    infer = saved_model_loaded.signatures['serving_default']

    # begin video capture
    try:
        vid = cv2.VideoCapture(int(video_path))
    except:
        vid = cv2.VideoCapture(video_path)

    out = None

    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))

    while True:
        return_value, frame = vid.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
        else:
            print('Video has ended or failed, try a different video format!')
            break
    
        frame_size = frame.shape[:2]
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        start_time = time.time()

        # inference
        batch_data = tf.constant(image_data)
        pred_bbox = infer(batch_data)

        for key, value in pred_bbox.items():
            boxes = value[:, :, 0:4]
            pred_conf = value[:, :, 4:]

        # tensor to numpy array 
        all_boxes=boxes[0].numpy()
        all_preds=pred_conf[0].numpy()

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=num_objects,
            max_total_size=2,
            iou_threshold=FLAGS.iou,
            score_threshold=FLAGS.score)
        

        print(all_boxes)
        print("----")
        print(all_preds)
        # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, xmax, ymax
        original_h, original_w, _ = frame.shape
        bboxes = utils.format_boxes(boxes.numpy()[0], original_h, original_w)
        print('******')
        print(bboxes)

        # generate list of dictionnary containing bbox coordinates and classe predictions for each bboxes detected in the frame
 

        if not np.any(bboxes) : 
            pass 
        else :
            boxes_and_preds=output_box(all_boxes, all_preds, boxes, bboxes, class_names)

        # sorted the predictions by descending order and keep 'max_classes' number of them
        if FLAGS.sort_by :
            boxes_and_preds = sort_by(boxes_and_preds, max_classes)

        # write boxes coordinates and classe predictions into a file and save it 
        save_file.write(f"{boxes_and_preds} \n")

        # format bounding game areas from normalized to image-formatted
        formatted_game_areas=format_game_areas(game_areas, original_h, original_w)

        # filtering box coordinates according to the game areas  
        if FLAGS.margin : 
            area_boxes_and_preds=filter_game_areas(boxes_and_preds,formatted_game_areas,k)

        # write box coordinates and classe predictions into a file according to the game areas and save it 
        save_file_2.write(f"{area_boxes_and_preds} \n")
    
        pred_bbox = [bboxes, scores.numpy()[0], classes.numpy()[0], valid_detections.numpy()[0]]


        image= utils.draw_bbox(frame, pred_bbox)
        
        ####################################
        
        fps = 1.0 / (time.time() - start_time)
        print("FPS: %.2f" % fps)
        print("#############################################################")
        result = np.asarray(image)
        cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
        result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        if not FLAGS.dont_show:
            cv2.imshow("result", result)
        
        if FLAGS.output:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass