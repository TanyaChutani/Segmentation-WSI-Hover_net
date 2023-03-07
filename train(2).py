import sys
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from glob import glob
from tensorflow import keras as K


import sys
#!{sys.executable} -m pip install --upgrade pip
#!{sys.executable} -m pip install tensorflow
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from glob import glob

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
'''
path = "/home/ubuntu/aira/consep/consep/train/masks1/"

for j in os.listdir(path):
    try:
        mask = cv2.imread(path+j)
        black = str([0, 0, 0])
        yellow = str([255,  255, 0])
        red = str([255,  0,   0])

        labels = {black:0, yellow:2, red:1}

        width = 540
        height = 540
        values = [list(mask[i,j]) for i in range(height) for j in range(width)]

        mask = np.array(([0]*width*height))

        for i, value in enumerate(values):
            mask[i]= labels[value]
        ann = np.asarray(mask, dtype=np.int32).reshape(height,width)
        cv2.imwrite("/home/ubuntu/aira/consep/consep/train/masks/"+j, ann)
    except:
        print(j)
        if j.split(".")[-1] == ".png":
            mask = np.zeros((540, 540, 3))
            cv2.imwrite("/home/ubuntu/aira/consep/consep/train/masks/"+j, mask)  
print("done1")
path = "/home/ubuntu/aira/consep/consep/valid/masks1/"
for j in os.listdir(path):
    try:
        mask = cv2.imread(path+j)
        black = str([0, 0, 0])
        yellow = str([255,  255, 0])
        red = str([255,  0,   0])

        labels = {black:0, yellow:2, red:1}

        width = 540
        height = 540
        values = [str(list(mask[i,j])) for i in range(height) for j in range(width)]


        mask = np.array(([0]*width*height))

        for i, value in enumerate(values):
            mask[i]= labels[value]
        ann = np.asarray(mask, dtype=np.int32).reshape(height,width)
        cv2.imwrite("/home/ubuntu/aira/consep/consep/valid/masks/"+j, ann)
    except:
        print(j)
        if j.split(".")[-1] == ".png":
            mask = np.zeros((540, 540, 3))
            cv2.imwrite("/home/ubuntu/aira/consep/consep/valid/masks/"+j, mask)             
print("done2")
'''
img_w, img_h = 512, 512
batch_size = 4
num_classes = 3

def read_img(image_path, mask=False):
    img = tf.io.read_file(image_path)
    if mask:
        img = tf.image.decode_png(img, channels=1)
        img.set_shape([None, None, 1])
        img = tf.image.resize(images=img, size=[img_h, img_w])
        img = tf.cast(img, tf.int32) 
    else:
        img = tf.image.decode_png(img, channels=3)
        img.set_shape([None, None, 3])
        img = tf.image.resize(images=img, size=[img_h, img_w])
        img = tf.cast(img, tf.float32) / 127.5 - 1
    return img


def load_data(img_list, mask_list):
    img = read_img(img_list)
    mask = read_img(mask_list, mask=True)
    return img, mask

def data_generator(img_list, mask_list):
    
    dataset = tf.data.Dataset.from_tensor_slices((img_list, mask_list))
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.map(load_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.repeat()
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset


train_images_folder = (
    os.listdir("/home/ubuntu/aira/consep/consep/train/images/")
)
valid_images_folder = (
    os.listdir("/home/ubuntu/aira/consep/consep/valid/images/")
)

train_imgs_folder =  sorted(train_images_folder)
valid_imgs_folder =  sorted(valid_images_folder)

train_img_folder = list(map(lambda orig_string:  "/home/ubuntu/aira/consep/consep/train/images/" + orig_string, train_imgs_folder))
train_mask_folder = list(map(lambda orig_string: "/home/ubuntu/aira/consep/consep/train/masks/" + orig_string, train_imgs_folder))
valid_img_folder = list(map(lambda orig_string: "/home/ubuntu/aira/consep/consep/valid/images/"+ orig_string, valid_imgs_folder))
valid_mask_folder = list(map(lambda orig_string: "/home/ubuntu/aira/consep/consep/valid/masks/"+ orig_string, valid_imgs_folder))


train_dataset = data_generator(train_img_folder, train_mask_folder)
valid_dataset = data_generator(valid_img_folder, valid_mask_folder)
print("done3")

def AtrousSpatialPyramidPooling(model_input):
    dims = tf.keras.backend.int_shape(model_input)

    layer = tf.keras.layers.AveragePooling2D(pool_size=(dims[-3], dims[-2]))(
        model_input
    )
    layer = tf.keras.layers.Conv2D(
        256, kernel_size=1, padding="same", kernel_initializer="he_normal"
    )(layer)
    layer = tf.keras.layers.BatchNormalization()(layer)
    layer = tf.keras.layers.ReLU()(layer)
    out_pool = tf.keras.layers.UpSampling2D(
        size=(dims[-3] // layer.shape[1], dims[-2] // layer.shape[2]),
        interpolation="bilinear",
    )(layer)

    layer = tf.keras.layers.Conv2D(
        256,
        kernel_size=1,
        dilation_rate=1,
        padding="same",
        kernel_initializer="he_normal",
        use_bias=False,
    )(model_input)
    layer = tf.keras.layers.BatchNormalization()(layer)
    out_1 = tf.keras.layers.ReLU()(layer)

    layer = tf.keras.layers.Conv2D(
        256,
        kernel_size=3,
        dilation_rate=6,
        padding="same",
        kernel_initializer="he_normal",
        use_bias=False,
    )(model_input)
    layer = tf.keras.layers.BatchNormalization()(layer)
    out_6 = tf.keras.layers.ReLU()(layer)

    layer = tf.keras.layers.Conv2D(
        256,
        kernel_size=3,
        dilation_rate=12,
        padding="same",
        kernel_initializer="he_normal",
        use_bias=False,
    )(model_input)
    layer = tf.keras.layers.BatchNormalization()(layer)
    out_12 = tf.keras.layers.ReLU()(layer)

    layer = tf.keras.layers.Conv2D(
        256,
        kernel_size=3,
        dilation_rate=18,
        padding="same",
        kernel_initializer="he_normal",
        use_bias=False,
    )(model_input)
    layer = tf.keras.layers.BatchNormalization()(layer)
    out_18 = tf.keras.layers.ReLU()(layer)

    layer = tf.keras.layers.Concatenate(axis=-1)(
        [out_pool, out_1, out_6, out_12, out_18]
    )

    layer = tf.keras.layers.Conv2D(
        256,
        kernel_size=1,
        dilation_rate=1,
        padding="same",
        kernel_initializer="he_normal",
        use_bias=False,
    )(layer)
    layer = tf.keras.layers.BatchNormalization()(layer)
    model_output = tf.keras.layers.ReLU()(layer)
    return model_output

def DeeplabV3Plus(nclasses=3):
    model_input = tf.keras.Input(shape=(img_h, img_w, 3))
    resnet50 = tf.keras.applications.ResNet50(
        weights="imagenet", include_top=False, input_tensor=model_input
    )
    layer = resnet50.get_layer("conv4_block6_2_relu").output
    layer = AtrousSpatialPyramidPooling(layer)
    input_a = tf.keras.layers.UpSampling2D(
        size=(img_h // 4 // layer.shape[1], img_w // 4 // layer.shape[2]),
        interpolation="bilinear",
    )(layer)

    input_b = resnet50.get_layer("conv2_block3_2_relu").output
    input_b = tf.keras.layers.Conv2D(
        48,
        kernel_size=(1, 1),
        padding="same",
        kernel_initializer=tf.keras.initializers.he_normal(),
        use_bias=False,
    )(input_b)
    input_b = tf.keras.layers.BatchNormalization()(input_b)
    input_b = tf.keras.layers.ReLU()(input_b)

    layer = tf.keras.layers.Concatenate(axis=-1)([input_a, input_b])

    layer = tf.keras.layers.Conv2D(
        256,
        kernel_size=3,
        padding="same",
        activation="relu",
        kernel_initializer=tf.keras.initializers.he_normal(),
        use_bias=False,
    )(layer)
    layer = tf.keras.layers.BatchNormalization()(layer)
    layer = tf.keras.layers.ReLU()(layer)
    layer = tf.keras.layers.Conv2D(
        256,
        kernel_size=3,
        padding="same",
        activation="relu",
        kernel_initializer=tf.keras.initializers.he_normal(),
        use_bias=False,
    )(layer)
    layer = tf.keras.layers.BatchNormalization()(layer)
    layer = tf.keras.layers.ReLU()(layer)
    layer = tf.keras.layers.UpSampling2D(
        size=(img_h // layer.shape[1], img_w // layer.shape[2]),
        interpolation="bilinear",
    )(layer)
    model_output = tf.keras.layers.Conv2D(
        num_classes, kernel_size=(1, 1), padding="same",
    )(layer)
    return tf.keras.Model(inputs=model_input, outputs=model_output)



class Dice_Loss(tf.keras.losses.Loss):
    def __init__(self, **kwargs):
        super(Dice_Loss, self).__init__()


    def call(self, y_true, y_pred):
        y_true = tf.squeeze(y_true)
        y_true_f =tf.one_hot(tf.cast(y_true, tf.int32), depth=3)
        
        y_pred_f = tf.nn.softmax(y_pred)
        intersect = tf.math.reduce_sum(y_true_f * y_pred_f, axis=-1)
        denom = tf.math.reduce_sum(y_true_f + y_pred_f, axis=-1)
        return tf.nn.softmax_cross_entropy_with_logits(y_true_f, y_pred)+( 1 - tf.math.reduce_mean((2. * intersect / (denom))))
    
class Dice_Metric(tf.keras.metrics.Metric):
    def __init__(self,class_id, **kwargs):
        super(Dice_Metric, self).__init__()
        self.dice_metric = self.add_weight(name='dm', initializer='zeros')
        self.dice_metric_c = self.add_weight(name='dmc', initializer='zeros')
        self.class_id= class_id
       
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.squeeze(y_true)
        y_true_f =tf.one_hot(tf.cast(y_true, tf.int32), depth=3)
        y_true_c = y_true_f[..., self.class_id]
        
        y_pred_f = tf.nn.softmax(y_pred)
        
        intersect = tf.math.reduce_sum(y_true_f * y_pred_f, axis=-1)
        denom = tf.math.reduce_sum(y_true_f + y_pred_f, axis=-1)
        
        y_pred_c = tf.cast(tf.equal(tf.math.argmax(y_pred_f, axis=-1), self.class_id), dtype=tf.float32)

        intersect_c = tf.math.reduce_sum(y_true_c * y_pred_c)
        denom_c = tf.math.reduce_sum(y_true_c + y_pred_c)
        
        self.dice_metric.assign(tf.math.reduce_mean((2. * intersect / (denom))))
        self.dice_metric_c.assign(tf.math.reduce_mean((2. * intersect_c / (denom_c))))
        
    def result(self):
        return self.dice_metric_c

    def reset_states(self):
        self.dice_metric_c.assign(0.) 



epochs = 50
step_per_epoch = len(train_images_folder) // batch_size
model = DeeplabV3Plus()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss=Dice_Loss(),
    metrics=[Dice_Metric(class_id=i) for i in range(3)]
)

filepath = f"/home/ubuntu/aira/deeplab_model.h5"
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath, monitor="val_loss", save_best_only=True, mode="min", save_freq='epoch'
)


model.fit(
    train_dataset,
    steps_per_epoch=step_per_epoch,
    epochs=epochs,
    validation_data=valid_dataset,
    validation_steps=len(valid_images_folder) // batch_size,
    callbacks=[checkpoint],
)
model.save_weights("/home/ubuntu/aira/deeplab_model1.h5")


def inference(model, dataset):
    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(20, 20))
    for i in range(batch_size):
        for val in dataset.take(1):
            img = tf.cast((val[0][i] + 1) * 127.5, tf.uint8)
            axes[0, i].imshow(img)
            axes[0, i].title.set_text("Original Image")
            y_true = tf.squeeze(val[1][i])
            y_true =tf.one_hot(tf.cast(y_true, 'int32'), depth=3)

            print(f"Original Mask {i}",y_true)
            axes[1, i].imshow((y_true*255))
            axes[1, i].title.set_text("Original Mask")

            predsTrain = model.predict(np.expand_dims(val[0][i], axis=0))
            
            out = tf.squeeze(predsTrain)
            y = tf.math.argmax(out, axis=-1)
            y_pred =tf.one_hot(tf.cast(y, 'int32'), depth=3)
            print(f"Pred Mask {i}",y_pred)
            axes[2, i].imshow(y_pred*255)
            axes[2, i].title.set_text("Predicted Mask")  

    fig.savefig('/home/ubuntu/aira/pred.jpg')
inference(model, train_dataset)
