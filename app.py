from keras.src.models import Sequential
from keras.src.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.src.utils import to_categorical
from keras.src.datasets import mnist

import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image, ImageOps

# from keras.src.models import load_model   # tried but not working
from keras.models import load_model     # working


def train_save_model():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # if we want to see a sample image
    # from PIL import Image
    # Image.fromarray(x_train[0])

    # reshaping the date
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    # normalizing it
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    # correcting labels
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    # building the model
    model = Sequential()
    model.add(Conv2D(15, (3,3), padding="valid", activation="relu", kernel_initializer="he_normal", input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2,2), strides=(2,2), padding="valid"))

    model.add(Conv2D(30, (3,3), padding=("valid"), activation="relu", kernel_initializer="he_normal"))
    model.add(MaxPooling2D((2,2), (2,2), padding="valid"))

    model.add(Conv2D(20, (3,3), padding="valid", activation="relu", kernel_initializer="he_normal"))
    model.add(MaxPooling2D((2,2), (2,2), padding="valid"))

    model.add(Flatten())

    model.add(Dense(50, activation="relu", kernel_initializer="he_normal"))
    model.add(Dense(10, activation="softmax", kernel_initializer="glorot_uniform"))

    model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])
    model.fit(x=x_train, y=y_train, batch_size=200, epochs=20, verbose=1, validation_split=.1)

    # testing the model
    pred_results=model.predict(x_test)

    # evaluating the model on accuracy
    eval_results=model.evaluate(x_test, y_test)

    # displaying the results
    for i in range(10):
        plt.subplot(2, 5, i+1)
        # displaying original val
        plt.title(np.argmax(y_test[i]))
        plt.imshow(x_test[i], cmap='gray')
        plt.axis('off')
        # displaying predicted val
        plt.text(12, 35, np.argmax(np.round(pred_results[i], 0)), {'color': 'blue'})

    # saving results
    plt.savefig("src\results.png")

    # saving the model
    model.save("src\cnn-mnist-model.keras")
    return

def enhance_img(image_path: str, factor= 100, enhanced_dir= "src\enhanced_imgs"):
    """First inverts (representing 255 as black and 0 as white) the given image and then converts all the pixels whose value is less then factor to 0, making pixels < factor to black. Returns enhanced_img_path"""
    if not os.path.exists(image_path):
        raise Exception(f"Image doesn't exist.")
    
    if not os.path.exists(enhanced_dir):
        os.makedirs(enhanced_dir, exist_ok=True)

    # inverting and enhancing the image
    img=Image.open(image_path).convert('L')
    img=ImageOps.invert(img)
    img=img.point(lambda x: 0 if x<factor else 255)
    
    # building enhanced_img_path
    n_path=os.path.join(enhanced_dir, image_path.rsplit(".", 1)[0].rsplit("\\", 1)[1]+"-enhanced.jpg")

    print(f"Saving at: {n_path}")
    img.save(n_path)
    return n_path

def get_prediction(model_path, image_path):
    """Loads the model and the image. Transform image, get prediction and return it."""
    if not os.path.exists(model_path):
        raise Exception(f"Model path doesn't exist.")
    
    if not os.path.exists(image_path):
        raise Exception(f"Image doesn't exist.")

    try:
        model=load_model(model_path)
        print("Model loaded.")
    except Exception as e:
        raise Exception(f"Model loading failed. {e}")
    
    # opening and transforming image
    image=np.array(Image.open(image_path)).astype("float32") / 255
    
    if image.shape != (28,28):
        raise Exception(f"Image dimensions are incompatible. Required are: {(28, 28)} These are: {image.shape}")
    
    image=image.reshape(1, 28, 28, 1)

    # making prediction
    prediction=model.predict(image)
    # print(f"Shape is: {image.shape}")
    return str(np.argmax(np.round(prediction, 0)))

def predict_cdir(folder_name, model_path= r"src\cnn-mnist-model.keras", result_imgpath=r"src\folder-digit-results.jpg"):
    """Requires a folder_name and does the prediction (doesn't enhance) for each image and plots on a figure. At the end saves the image representing each image and relevant predicted digit."""
    for i, img_filename in enumerate(os.listdir(folder_name)):
        img_path=os.path.join(folder_name, img_filename)

        prediction = get_prediction(model_path, img_path)

        plt.subplot(1, len(os.listdir(folder_name)), i+1)
        plt.title(prediction)
        plt.imshow(Image.open(img_path), cmap='gray')
        plt.axis('off')
        print(f"Processed: {img_filename}")

    plt.savefig(result_imgpath)


# code to process each image and get prediction
# img_path = r"src\hand-written-28x28-numbers\b-no-5-28x28.jpg"
# prediction = get_prediction("src\cnn-mnist-model.keras", enhance_img(img_path))
# print("The predicted number is:", prediction)


# process all the enhanced images and save the resulting fig
imgdir_name="src\enhanced_imgs"
predict_cdir(imgdir_name)
