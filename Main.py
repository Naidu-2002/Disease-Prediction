from matplotlib import pyplot as plt
from tkinter import messagebox
from tkinter import *
from tkinter.filedialog import askopenfilename
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
from keras import applications
from keras.layers import Input
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.models import model_from_json
import os
from keras.preprocessing import image
import numpy as np
from keras.layers import Convolution2D
import cv2
import imutils
import pickle

root = tkinter.Tk()

root.title("PREDICTION OF DISEASES BASED ON FACIAL DIAGNOSIS USING DEEP LEARNING ")
root.geometry("1200x850")

global filename
global vgg_classifier
global training_set
global test_set

classes = ['betathalassemia','downsyndrome','hyperthyroidism','leprosy']


def upload():
    global filename
    filename = filedialog.askdirectory(initialdir=".")
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n");

def preprocess():
    text.delete('1.0', END)
    global training_set
    global test_set
    train_datagen = ImageDataGenerator()
    test_datagen = ImageDataGenerator()
    training_set = train_datagen.flow_from_directory('Dataset/train',target_size = (224, 224), batch_size = 2, class_mode = 'categorical', shuffle=True)
    test_set = test_datagen.flow_from_directory('Dataset/test',target_size = (224, 224), batch_size = 2, class_mode = 'categorical', shuffle=False)
    text.insert(END,"Dataset preprocessing completed\n")
    text.insert(END,"Total classes found in dataset : "+str(training_set.class_indices)+"\n")
    
    
def buildCNNModel():
    text.delete('1.0', END)
    global training_set
    global test_set
    global vgg_classifier
    if os.path.exists('model/model.json'):
        with open('model/model.json', "r") as json_file:
            type_model_json = json_file.read()
            vgg_classifier = model_from_json(type_model_json)

        vgg_classifier.load_weights("model/model_weights.h5")
        vgg_classifier._make_predict_function()   
        vgg_classifier.summary()
        f = open('model/history.pckl', 'rb')
        data = pickle.load(f)
        f.close()
        acc = data['accuracy']
        accuracy = acc[9] * 100
        text.insert(END,"Fine Tuning VGG16 Transfer Learning Prediction Accuracy : "+str(accuracy)+"\n\n")
    else:
        input_tensor = Input(shape=(224, 224, 3))
        vgg_model = applications.VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3)) #VGG16 transfer learning code here
        vgg_model.summary()
        layer_dict = dict([(layer.name, layer) for layer in vgg_model.layers])
        x = layer_dict['block2_pool'].output
        x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Flatten()(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(4, activation='softmax')(x)
        vgg_classifier = Model(input=vgg_model.input, output=x)
        for layer in vgg_classifier.layers[:7]:
            layer.trainable = False
        vgg_classifier.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        hist = vgg_classifier.fit_generator(training_set,samples_per_epoch = 8000,nb_epoch = 10,validation_data = test_set,nb_val_samples = 2000)
        vgg_classifier.save_weights('model/model_weights.h5')
        model_json = vgg_classifier.to_json()
        with open("model/model.json", "w") as json_file:
            json_file.write(model_json)
        print(training_set.class_indices)
        print(custom_model.summary)
        f = open('model/history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()
        f = open('model/history.pckl', 'rb')
        data = pickle.load(f)
        f.close()
        acc = data['accuracy']
        accuracy = acc[9] * 100
        text.insert(END,"Fine Tuning VGG16 Transfer Learning Prediction Accuracy : "+str(accuracy)+"\n\n")
        


def predict():   
    filename = filedialog.askopenfilename(initialdir="testImages")
    image = cv2.imread(filename)
    img = cv2.resize(image, (64,64))
    im2arr = np.array(img)
    im2arr = im2arr.reshape(1,64,64,3)
    img = np.asarray(im2arr)
    img = img.astype('float32')
    img = img/255
    preds = vgg_classifier.predict(img)
    predict = np.argmax(preds)
    print(predict)
    img = cv2.imread(filename)
    img = cv2.resize(img, (600,400))
    cv2.putText(img, "Disease Predicted as : "+classes[predict], (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 255, 0), 2)
    cv2.imshow("Disease Predicted as : "+classes[predict], img)
    cv2.waitKey(0)
    
    
    

    

def graph():
    f = open('model/history.pckl', 'rb')
    data = pickle.load(f)
    f.close()

    accuracy = data['accuracy']
    loss = data['loss']
    plt.figure(figsize=(10,6))
    plt.grid(True)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy/Loss')
    plt.plot(loss, 'ro-', color = 'red')
    plt.plot(accuracy, 'ro-', color = 'green')
    plt.legend(['Loss', 'Accuracy'], loc='upper left')
    #plt.xticks(wordloss.index)
    plt.title('VGG16 Transfer Learning Accuracy & Loss Graph')
    plt.show()

def exit():
    root.destroy()
    

font = ('times', 18, 'bold')
title = Label(root, text='PREDICTION OF DISEASES BASED ON FACIAL DIAGNOSIS USING DEEP LEARNING')
title.config(font=font)           
title.config(height=3, width=80)       
title.place(x=5,y=5)

font1 = ('times', 13, 'bold')

upload = Button(root, text="Upload Facial Diagnosis Dataset", command=upload)
upload.place(x=20,y=100)
upload.config(font=font1)  


processButton = Button(root, text="Preprocess Dataset", command=preprocess)
processButton.place(x=330,y=100)
processButton.config(font=font1)  

vggbutton = Button(root, text="Fine Tune VGG16 Transfer Learning", command=buildCNNModel)
vggbutton.place(x=650,y=100)
vggbutton.config(font=font1)

graphbutton = Button(root, text="Accuracy & Loss Graph", command=graph)
graphbutton.place(x=20,y=150)
graphbutton.config(font=font1)

predictbutton = Button(root, text="Upload Test Image & Predict Disease", command=predict)
predictbutton.place(x=330,y=150)
predictbutton.config(font=font1)

exitbutton = Button(root, text="Exit", command=exit)
exitbutton.place(x=650,y=150)
exitbutton.config(font=font1)

text=Text(root,height=20,width=100)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=200)
text.config(font=font1)  

root.mainloop()
