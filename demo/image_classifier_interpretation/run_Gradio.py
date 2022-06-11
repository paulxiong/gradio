# %%
import requests
import tensorflow as tf
import os
import sys
import numpy as np
from PIL import Image
import json

sys.path.append(os.getcwd())
root_dir = os.getcwd()
sys.path.insert(1,root_dir) 

import ml_collections
from models import ar_model as model_lib
from data import data_utils
from tasks.object_detection import TaskObjectDetection
from tasks.visualization import vis_utils
import gradio as gr

#@title Load model.
model_dir = '/mnt/gradio/demo/image_classifier_interpretation/model_dw/resnet_640x640/' #@param
with tf.io.gfile.GFile(os.path.join(model_dir, 'config.json'), 'r') as f:
  config = ml_collections.ConfigDict(json.loads(f.read()))


# %%
# Set batch size to 1.
config.eval.batch_size = 1

# Remove the annotation filepaths.
config.dataset.coco_annotations_dir = None

# Update config fields.
config.task.vocab_id = 10  # object_detection task vocab id.
config.training = False
config.dataset.val_filename='instances_val2017.json'

assert config.task.name == "object_detection"
task = TaskObjectDetection(config)

# Restore checkpoint.
# model = model_lib.Model(config)
# checkpoint = tf.train.Checkpoint(
#     model=model, global_step=tf.Variable(0, dtype=tf.int64))
# ckpt = tf.train.latest_checkpoint(model_dir)
# checkpoint.restore(ckpt).expect_partial()
# global_step = checkpoint.global_step
model = tf.saved_model.load(model_dir)

#@title Category names for COCO.
categories_str = '{"categories": [{"supercategory": "person","id": 1,"name": "person"},{"supercategory": "vehicle","id": 2,"name": "bicycle"},{"supercategory": "vehicle","id": 3,"name": "car"},{"supercategory": "vehicle","id": 4,"name": "motorcycle"},{"supercategory": "vehicle","id": 5,"name": "airplane"},{"supercategory": "vehicle","id": 6,"name": "bus"},{"supercategory": "vehicle","id": 7,"name": "train"},{"supercategory": "vehicle","id": 8,"name": "truck"},{"supercategory": "vehicle","id": 9,"name": "boat"},{"supercategory": "outdoor","id": 10,"name": "traffic light"},{"supercategory": "outdoor","id": 11,"name": "fire hydrant"},{"supercategory": "outdoor","id": 13,"name": "stop sign"},{"supercategory": "outdoor","id": 14,"name": "parking meter"},{"supercategory": "outdoor","id": 15,"name": "bench"},{"supercategory": "animal","id": 16,"name": "bird"},{"supercategory": "animal","id": 17,"name": "cat"},{"supercategory": "animal","id": 18,"name": "dog"},{"supercategory": "animal","id": 19,"name": "horse"},{"supercategory": "animal","id": 20,"name": "sheep"},{"supercategory": "animal","id": 21,"name": "cow"},{"supercategory": "animal","id": 22,"name": "elephant"},{"supercategory": "animal","id": 23,"name": "bear"},{"supercategory": "animal","id": 24,"name": "zebra"},{"supercategory": "animal","id": 25,"name": "giraffe"},{"supercategory": "accessory","id": 27,"name": "backpack"},{"supercategory": "accessory","id": 28,"name": "umbrella"},{"supercategory": "accessory","id": 31,"name": "handbag"},{"supercategory": "accessory","id": 32,"name": "tie"},{"supercategory": "accessory","id": 33,"name": "suitcase"},{"supercategory": "sports","id": 34,"name": "frisbee"},{"supercategory": "sports","id": 35,"name": "skis"},{"supercategory": "sports","id": 36,"name": "snowboard"},{"supercategory": "sports","id": 37,"name": "sports ball"},{"supercategory": "sports","id": 38,"name": "kite"},{"supercategory": "sports","id": 39,"name": "baseball bat"},{"supercategory": "sports","id": 40,"name": "baseball glove"},{"supercategory": "sports","id": 41,"name": "skateboard"},{"supercategory": "sports","id": 42,"name": "surfboard"},{"supercategory": "sports","id": 43,"name": "tennis racket"},{"supercategory": "kitchen","id": 44,"name": "bottle"},{"supercategory": "kitchen","id": 46,"name": "wine glass"},{"supercategory": "kitchen","id": 47,"name": "cup"},{"supercategory": "kitchen","id": 48,"name": "fork"},{"supercategory": "kitchen","id": 49,"name": "knife"},{"supercategory": "kitchen","id": 50,"name": "spoon"},{"supercategory": "kitchen","id": 51,"name": "bowl"},{"supercategory": "food","id": 52,"name": "banana"},{"supercategory": "food","id": 53,"name": "apple"},{"supercategory": "food","id": 54,"name": "sandwich"},{"supercategory": "food","id": 55,"name": "orange"},{"supercategory": "food","id": 56,"name": "broccoli"},{"supercategory": "food","id": 57,"name": "carrot"},{"supercategory": "food","id": 58,"name": "hot dog"},{"supercategory": "food","id": 59,"name": "pizza"},{"supercategory": "food","id": 60,"name": "donut"},{"supercategory": "food","id": 61,"name": "cake"},{"supercategory": "furniture","id": 62,"name": "chair"},{"supercategory": "furniture","id": 63,"name": "couch"},{"supercategory": "furniture","id": 64,"name": "potted plant"},{"supercategory": "furniture","id": 65,"name": "bed"},{"supercategory": "furniture","id": 67,"name": "dining table"},{"supercategory": "furniture","id": 70,"name": "toilet"},{"supercategory": "electronic","id": 72,"name": "tv"},{"supercategory": "electronic","id": 73,"name": "laptop"},{"supercategory": "electronic","id": 74,"name": "mouse"},{"supercategory": "electronic","id": 75,"name": "remote"},{"supercategory": "electronic","id": 76,"name": "keyboard"},{"supercategory": "electronic","id": 77,"name": "cell phone"},{"supercategory": "appliance","id": 78,"name": "microwave"},{"supercategory": "appliance","id": 79,"name": "oven"},{"supercategory": "appliance","id": 80,"name": "toaster"},{"supercategory": "appliance","id": 81,"name": "sink"},{"supercategory": "appliance","id": 82,"name": "refrigerator"},{"supercategory": "indoor","id": 84,"name": "book"},{"supercategory": "indoor","id": 85,"name": "clock"},{"supercategory": "indoor","id": 86,"name": "vase"},{"supercategory": "indoor","id": 87,"name": "scissors"},{"supercategory": "indoor","id": 88,"name": "teddy bear"},{"supercategory": "indoor","id": 89,"name": "hair drier"},{"supercategory": "indoor","id": 90,"name": "toothbrush"}]}'
categories_dict = json.loads(categories_str)
categories_dict = {c['id']: c for c in categories_dict['categories']}

# %%
url = 'http://images.cocodataset.org/val2017/000000039769.jpg' #@param

im = Image.open(requests.get(url, stream=True).raw)
im


# %%
num_instances_to_generate = 10 #@param
min_score_thresh = 0.5 #@param

# Build inference graph.
task.config.task.max_instances_per_image_test = num_instances_to_generate
@tf.function
def infer(model, preprocessed_outputs):
  return task.infer(model, preprocessed_outputs)

# Construct features and dummy labels.
im = np.array(im)
features = {
    'image': tf.image.convert_image_dtype(im, tf.float32),
    'image/id': 0, # dummy image id.
    'orig_image_size': tf.shape(im)[0:2],
}
labels = {
    'label': tf.zeros([1], tf.int32),
    'bbox': tf.zeros([1, 4]),
    'area': tf.zeros([1]),
    'is_crowd': tf.zeros([1]),
}
features, labels = data_utils.preprocess_eval(
    features,
    labels,
    max_image_size=config.model.image_size,
    max_instances_per_image=1)

# Batch features and labels.
features = {
    k: tf.expand_dims(v, 0) for k, v in features.items()
}
labels = {
    k: tf.expand_dims(v, 0) for k, v in labels.items()
}
# Inference.
preprocessed_outputs = (features['image'], None, (features, labels))
infer_outputs = infer(model, preprocessed_outputs)
_, pred_seq, _ = infer_outputs
results = task.postprocess_tpu(*infer_outputs)

# %%
(images, _, pred_bboxes, _, pred_classes, scores, _, _, _, _, _) = results
#print(images)

# %%
#response = requests.get("https://git.io/JJkYN")
#labels = response.text.split("\n")
#print(f"original:{labels}")
#f = open("JJkYN_labes.txt", 'r')
#labels = f.read().split("\n") 
#print(f"local readed:{labels}")


# %%


#inception_net = tf.keras.applications.MobileNetV2()  # load the model

# Download human-readable labels for ImageNet.
#response = requests.get("https://git.io/JJkYN")
#labels = response.text.split("\n")


def classify_image(inp):
    #inp = inp.reshape((-1, 224, 224, 3))
    #inp = tf.keras.applications.mobilenet_v2.preprocess_input(inp)
    #prediction = inception_net.predict(inp).flatten()
    # return {labels[i]: float(prediction[i]) for i in range(1000)}
    #return 'punks.png'
    #return tf.image.convert_image_dtype(images[0], tf.uint8).numpy()

    #im = np.array(im)
    im = inp
    features = {
        'image': tf.image.convert_image_dtype(im, tf.float32),
        'image/id': 0, # dummy image id.
        'orig_image_size': tf.shape(im)[0:2],
    }
    labels = {
        'label': tf.zeros([1], tf.int32),
        'bbox': tf.zeros([1, 4]),
        'area': tf.zeros([1]),
        'is_crowd': tf.zeros([1]),
    }
    features, labels = data_utils.preprocess_eval(
        features,
        labels,
        max_image_size=config.model.image_size,
        max_instances_per_image=1)

    # Batch features and labels.
    features = {
        k: tf.expand_dims(v, 0) for k, v in features.items()
    }
    labels = {
        k: tf.expand_dims(v, 0) for k, v in labels.items()
    }
    # Inference.
    preprocessed_outputs = (features['image'], None, (features, labels))
    infer_outputs = infer(model, preprocessed_outputs)
    _, pred_seq, _ = infer_outputs
    results = task.postprocess_tpu(*infer_outputs)




    
    (images, _, pred_bboxes, _, pred_classes, scores, _, _, _, _, _) = results
    vis = vis_utils.visualize_boxes_and_labels_on_image_array(
        image=tf.image.convert_image_dtype(images[0], tf.uint8).numpy(),
        boxes=pred_bboxes[0].numpy(),
        classes=pred_classes[0].numpy(),
        scores=scores[0].numpy(),
        category_index=categories_dict,
        use_normalized_coordinates=True,
        min_score_thresh=min_score_thresh,
        max_boxes_to_draw=100)

    return Image.fromarray(vis)



image = gr.inputs.Image(shape=(224, 224))
label = gr.outputs.Label(num_top_classes=3)

gr.Interface(
    # fn=classify_image, inputs=image, outputs=label, interpretation="default"
    fn=classify_image, 
    inputs=image, 
    outputs="image", 
    interpretation="default"
).launch()


# %%
