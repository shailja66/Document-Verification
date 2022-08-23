**Hardware Requirements**
Laptop with any Operating System(Windows,MacOS etc)
8GB Ram
NVIDIA GTX 650+ Graphic Card

**Software Requirements**
1)TensorFlow GPU
2)NVIDIA CUDA
3)NVIDIA CUDA Deep Neural Network Library(cuDNN)
4)Visual Studio Build Tools(atleast 2014 version)



Note:-The versions of CUDA and cuDNN have to be compatible with tensorflow GPU for the 
object detection api to work properly

**Preparing the Workspace**

We shall begin by creating the following directory structure. The image given below 
shows the general template. In our case the name of the root directory is project 

project/
 ├─ addons/ (Optional)
│ └─ labelImg/
 └─ models/
 ├─ community/
 ├─ official/
 ├─ orbit/
 ├─ research/
 └─ ...

Now create a new folder under (project) and call it workspace . It is within
the that we will store all our training set-ups. Now let’s go under
workspace and create another folder named training

The folder training shall be our training folder, which will contain all files
related to our model training. 

Below given is the function of each folder:


annotations : This folder will be used to store all files and the respective
TensorFlow files, which contain the list of annotations for our dataset

exported-models : This folder will be used to store exported versions of our trained
model(s).

images : In an ideal scenario, This folder contains a copy of all the images in our
dataset, as well as the respective
objects.
*.xml files produced for each one, once labelImg is used to annotate


respective images
images/train:This folder contains a copy of all images, and the
files, which will be used to train our model.
o images/test : This folder contains a copy of all images, and the
respective files, which will be used to test our model.


For our project, since we did not have a lot of images to train,we created the test 
and training images separately

models : This folder will contain a sub-folder for each of training job. Each
subfolder will contain the training pipeline configuration file *.config , as well as

all files generated during the training and evaluation of our model.

pre-trained-models : This folder will contain the downloaded pre-trained models,
which shall be used as a starting checkpoint for our training jobs.

**Preparing the Dataset**

**Annotate the Dataset**

Open a new Terminal window and activate the tensorflow_gpu environment (if
you  not done so already)

Run the following command to install labelImg :

pip install labelImg

labelimg can then be run as follows:

labelImg

**Annotate Images**

Once you have collected all the images to be used to test your model (ideally more
than 100 per class) but in our case we gave around 40 images

A File Explorer Dialog windows should open, which points to the spot where 
training images are stored.
In our case it is the C:\Users\Aditya\Desktop\Id_Images\train folder

Press the “Select Folder” button, to start annotating your images.


**Partition the dataset**
Once you have finished annotating your image dataset, it is a general convention to use only
part of it for training, and the rest is used for evaluation purposes (e.g. as discussed
in Evaluating the Model (Optional)).
Typically, the ratio is 9:1, i.e. 90% of the images are used for training and the rest 10% is
maintained for testing, but you can chose whatever ratio suits your needs.

Once you have decided how you will be splitting your dataset, copy all training images,
together with their corresponding files, and place them inside
the directory where training images are present. Similarly, copy all testing images, with their *.xml files, and

paste them inside directory where testing images are present 


**Create Label Map**

TensorFlow requires a label map, which namely maps each of the used labels to an integer
values. This label map is used both by the training and detection processes.

item {
 id: 1
 name: 'vit_logo'
}
item {
 id: 2
 name: 'stay'
}


Label map files have the extention .pbtxt and should be placed inside
the training_/annotations folder


**Create TensorFlow Records**

Now that we have generated our annotations and split our dataset into the desired training and
testing subsets, it is time to convert our annotations into the so called TFRecord format.


**Convert *.xml to *.record**

To do this we can use a simple script that iterates through all *.xml files in
the training_demo/images/train and training_demo/images/test folders, and generates a *.record file for
each of the two. This script comes with the tensorflow models we earlier cloned from 
github

Install the pandas package:

cd into training/scripts/preprocessing and run:

python generate_tfrecord.py -x [PATH_TO_IMAGES_FOLDER]/train -l
[PATH_TO_ANNOTATIONS_FOLDER]/label_map.pbtxt -o
[PATH_TO_ANNOTATIONS_FOLDER]/train.record

Repeat the above procedure for creating test.record


**Configuring a Training Job**

The model we will be using is  ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8 

To use this model we will have to first install the requisite tar file and extract it to the pre-trained models folder in the training
directory


Note:The above process can be repeated for all other pre-trained models

**Configure the Training Pipeline**

Under the training/models create a new directory
named my_model and copy the training/pre-trained- model/pipeline.config


Our training/models directory should now look like this:

training/
├─ ...
├─ models/
│ └─ my_model/
│ └─ pipeline.config
└─ ...

Changes to be applied

 1 model {
 2 ssd {
 3 num_classes: 1 # Set this to the number of different label classes
 4 image_resizer {
 5 fixed_shape_resizer {
 6 height: 640
 7 width: 640
 8 }
 9 }
10 feature_extractor {
11 type: "ssd_resnet50_v1_fpn_keras"
12 depth_multiplier: 1.0
13 min_depth: 16
14 conv_hyperparams {
15 regularizer {
l2_regularizer
{
17 weight: 0.00039999998989515007
18
}
19
}
20 initializer
{
21 truncated_normal_initializer
{
22 mean: 0.0
23 stddev: 0.029999999329447746
24
}
25
}
26 activation: RELU_6
27 batch_norm {
28 decay: 0.996999979019165
29 scale: true
30 epsilon: 0.0010000000474974513
31
}
32
}
33 override_base_feature_extractor_hyperparams: true
34 fpn {
35 min_level:
3
36 max_level:
7
37
}
38
}
39 box_coder {
40 faster_rcnn_box_coder
{
41 y_scale: 10.0
42 x_scale: 10.0
43 height_scale: 5.0
44 width_scale: 5.0
45
}
46
}
47 matcher
{
48 argmax_matcher
{
49 matched_threshold: 0.5
50 unmatched_threshold: 0.5
51 ignore_thresholds: false
52 negatives_lower_than_unmatched: true
53 force_match_for_each_row: true
54 use_matmul_gather: true
55
}
56
}
57 similarity_calculator
{
58 iou_similarity {
59
}
60
}
61 box_predictor
{
62 weight_shared_convolutional_box_predictor
{
63 conv_hyperparams
{
64 regularizer
{
65 l2_regularizer
{
66 weight: 0.00039999998989515007
67
}
68
}
69 initializer
{
70 random_normal_initializer
{
71 mean: 0.0
72 stddev: 0.009999999776482582
73
}
74
}
75 activation: RELU_6
76 batch_norm
{
77 decay: 0.996999979019165
78 scale: true
79 epsilon: 0.0010000000474974513
80
}
81
}
82 depth: 256

num_layers_before_predictor: 4
84 kernel_size: 3
85 class_prediction_bias_init: -4.599999904632568
86 }
87 }
88 anchor_generator {
89 multiscale_anchor_generator {
90 min_level: 3
91 max_level: 7
92 anchor_scale: 4.0
93 aspect_ratios: 1.0
94 aspect_ratios: 2.0
95 aspect_ratios: 0.5
96 scales_per_octave: 2
97 }
}
98 }
99 post_processing {
100 batch_non_max_suppression {
101 score_threshold: 9.99999993922529e-09
102 iou_threshold: 0.6000000238418579
103 max_detections_per_class: 100
104 max_total_detections: 100
105 use_static_shapes: false
106 }
107 score_converter: SIGMOID
108 }
109 normalize_loss_by_num_matches: true
110 loss {
111 localization_loss {
112 weighted_smooth_l1 {
113 }
114 }
115 classification_loss {
116 weighted_sigmoid_focal {
117 gamma: 2.0
118 alpha: 0.25
119 }
120 }
121 classification_weight: 1.0
122 localization_weight: 1.0
123 }
124 encode_background_as_zeros: true
125 normalize_loc_loss_by_codesize: true
126 inplace_batchnorm_update: true
127 freeze_batchnorm: false
128
129}
130train_config {
131 batch_size: 6 # Increase/Decrease this value depending on the available memory (Higher values require more memory and
vice-versa)
132 data_augmentation_options {
133 random_horizontal_flip {
134 }
135 }
136 data_augmentation_options {
137 random_crop_image {
138 min_object_covered: 0.0
139 min_aspect_ratio: 0.75
140 max_aspect_ratio: 3.0
141 min_area: 0.75
142 max_area: 1.0
143 overlap_thresh: 0.0
144 }
145 }
146 sync_replicas: true
147 optimizer {
148 momentum_optimizer {
149 learning_rate {
150 cosine_decay_learning_rate {
151 learning_rate_base: 0.03999999910593033
152 total_steps: 3000
153 warmup_learning_rate: 0.013333000242710114
154 warmup_steps: 1000
155 }
156 }
157 momentum_optimizer_value: 0.8999999761581421
158 }
159 use_moving_average: false
160 }
161 fine_tune_checkpoint: "pre-trained-models/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/checkpoint/ckpt-0" # Path to
checkpoint of pre-trained model
162 num_steps: 25000
163 startup_delay_steps: 0.0
164 replicas_to_aggregate: 8
165 max_number_of_boxes: 100
166 unpad_groundtruth_tensors: false
167 fine_tune_checkpoint_type: "detection" # Set this to "detection" since we want to be training the full detection model
168 use_bfloat16: false # Set this to false if you are not training on a TPU
169 fine_tune_checkpoint_version: V2
170}
171train_input_reader {
172 label_map_path: "annotations/label_map.pbtxt" # Path to label map file
173 tf_record_input_reader {
174 input_path: "annotations/train.record" # Path to training TFRecord file
175 }
176}
177eval_config {
178 metrics_set: "coco_detection_metrics"
179 use_moving_averages: false
180}
181eval_input_reader {
182 label_map_path: "annotations/label_map.pbtxt" # Path to label map file
183 shuffle: false
184 num_epochs: 1
185 tf_record_input_reader {
186 input_path: "annotations/test.record" # Path to testing TFRecord
187 }
188}


**Training the Model**

Before we begin training our model, let’s go and copy
the script and paste it straight into
our training folder. We will need this script in order to train our model.

python model_main_tf2.py --model_dir=models/my_model/pipeline_config_path=models/my_model/pipeline.config


**Exporting a Trained Model**

Once our training job is complete, we need to extract the newly trained inference graph,
which will be later used to perform the object detection. This can be done as follows:


Copy the TensorFlow/models/research/object_detection/exporter_main_v2.py script and paste it
straight into your training folder.


Now, open a Terminal, cd inside training folder,run the following command:


After the above process has completed, you should find a new folder my_model under
the training/exported-models , that has the following structure:

training_demo/
├─ ...
├─ exported-models/
│ └─ my_model/
│ ├─ checkpoint/
│ ├─ saved_model/
│ └─ pipeline.config
└─ ..


**OCR Prerequisites**

1)pytesseract
2)tesseract

After the above softwares are installed,the ocr program will run correctly