# Hashing imports
import hashlib

# Detection imports
import os 
import cv2
import tensorflow as tf
import numpy as np 
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder


# Kivy imports
from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.screenmanager import Screen, ScreenManager
from kivy.lang import Builder
from kivy.core.text import LabelBase
from kivy.properties import ObjectProperty
from kivy.uix.popup import Popup


# Setting up login Screen
class Login(Screen):

    def credentials(self):
        self.ids.Login_btn.color = (0.5, 0, 0.5, 0.65)
        self.ids.Login_btn.font_size = 14

        username = ObjectProperty(None)
        password = ObjectProperty(None)

        # Hashing username and password
        # username
        chk_username = hashlib.sha256(self.username.text.encode())
        chk_username = chk_username.hexdigest()

        # password
        chk_password = hashlib.sha256(self.password.text.encode())
        chk_password = chk_password.hexdigest()

        digest1 = 'c1c224b03cd9bc7b6a86d77f5dace40191766c485cd55dc48caf9ac873335d6f'
        digest2 = 'f473597cd4efbeca4e305eddf67769d7cf7bba007bf36239f0dd857a30dc63ff'

        if chk_username == digest1 and chk_password == digest2:
            print('Access Granted')
            self.parent.current = 'mainwindow'

        else:
            print('Access denied')
            self.ids.invalid.opacity = 1

        self.username.text = ''
        self.password.text = ''

    def release(self):
        self.ids.Login_btn.color = (1, 1, 1, 0.65)
        self.ids.Login_btn.font_size = 17


class MainWindow(Screen):
    def pressed(self):
        self.ids.start_engine.size_hint = (0.25, 0.25)
        self.ids.systems_id.color = (0,0,0,1)

    def released(self):
        self.ids.start_engine.size_hint = (0.35, 0.35)
        self.ids.systems_id.color = (0,0,0,0)

    def startengine(self):
        print('[+]   Systems starting  [+]')
        
        config_path = 'my_ssd_mobilenet_v2fpnlit_320x320//pipeline.config'
        checkpoint_path = 'my_ssd_mobilenet_v2fpnlit_320x320'

        configs = config_util.get_configs_from_pipeline_file(config_path)
        detection_model = model_builder.build(model_config=configs['model'],is_training=False)

        ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
        ckpt.restore(os.path.join(checkpoint_path, 'ckpt-11')).expect_partial()
        category_index = label_map_util.create_category_index_from_labelmap('annotations//Label_map.pbtxt', use_display_name=True)

        @tf.function
        def detect_fn(image):
            image,shapes = detection_model.preprocess(image)
            prediction_dict = detection_model.predict(image, shapes)
            detections = detection_model.postprocess(prediction_dict,shapes)
            return detections,prediction_dict, tf.reshape(shapes, [-1])

        cap = cv2.VideoCapture(0)
        while True:
            ret,frame = cap.read()
            image_np = np.array(frame)
            
            image_npexpanded = np.expand_dims(image_np, axis=0)
            input_tensor = tf.convert_to_tensor(np.expand_dims(image_np,0),dtype=tf.float32)
            detections, predictions_dict, shapes = detect_fn(input_tensor)
            
            label_id_offset = 1
            image_np_with_detections = image_np.copy()
            
            viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections,
                detections['detection_boxes'][0].numpy(),
                (detections['detection_classes'][0].numpy() + label_id_offset).astype(int),
                detections['detection_scores'][0].numpy(),
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=2,
                min_score_thresh=0.3,
                agnostic_mode=False)
            
            cv2.imshow('FaceMask Detection', cv2.resize(image_np_with_detections, (800,600)))
            
            if cv2.waitKey(1) & 0xff == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()





class WindowManager(ScreenManager):
    pass

# loading kv file
kv = Builder.load_file('myfirst.kv')

# fonts
LabelBase.register(name='DancingScript', fn_regular='fonts/DancingScript-Regular.otf')
LabelBase.register(name='3Fs', fn_regular='fonts/FFF_Tusj.ttf')
LabelBase.register(name='seasrn', fn_regular='fonts/SEASRN__.ttf')
LabelBase.register(name='oswald-stencil', fn_regular='fonts/Oswald-Stencil.ttf')

class Myfirst(App):
    def build(self):
        return kv

if __name__ == "__main__":
    Myfirst().run()