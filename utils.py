import tensorflow as tf

from tqdm import tqdm 
import cv2
import os 
from pathlib import Path
import matplotlib.pyplot as plt 
import numpy as np 
import pickle
from PIL import Image
import subprocess
from mtcnn import MTCNN
from sklearn.metrics import mean_absolute_error, mean_squared_error

import librosa
import tensorflow_hub as hub
import pandas as pd 
from nltk.corpus import stopwords

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re


def loss_val_graph(history):
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    train_acc = history.history['acc']
    val_acc = history.history['val_acc']

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # Plot Accuracy
    ax[0].plot(train_acc, label="Training Accuracy", color='blue')
    ax[0].plot(val_acc, label="Validation Accuracy", color='orange')
    ax[0].set_title("Training and Validation Accuracy")
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("Accuracy (%)")
    ax[0].legend()
    ax[0].grid()

    # Plot Loss
    ax[1].plot(train_loss, label="Training Loss", color='blue')
    ax[1].plot(val_loss, label="Validation Loss", color='orange')
    ax[1].set_title("Training and Validation Loss")
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("Loss")
    ax[1].legend()
    ax[1].grid()

    plt.show()

    # Cetak hasil akhir
    print(f"Final Training Loss: {train_loss[-1]:.4f} | Final Validation Loss: {val_loss[-1]:.4f}")
    print(f"Final Training Accuracy: {train_acc[-1]:.2f}% | Final Validation Accuracy: {val_acc[-1]:.2f}%")


def preprocess_text(text):
    # Remove unnecessary characters
    pattern = r'[^A-Za-z0-9\s]' #0-9
    cleaned_text = re.sub(pattern, '', text)
    
    # Convert to lowercase
    cleaned_text = cleaned_text.lower()
    
    # Remove stopwords
    stop = stopwords.words('english')
    final_text = []
    for word in cleaned_text.split():
        if word.strip().lower() not in stop:
            final_text.append(word.strip())
    cleaned_text = ' '.join(final_text)
    
    stemmer = PorterStemmer()
    # Apply stemming
    tokens = word_tokenize(cleaned_text)
    stemmed_words = [stemmer.stem(token) for token in tokens]
    cleaned_text = ' '.join(stemmed_words)
    
    return cleaned_text


def mean_abs_error(y_true, y_pred, multioutput='raw_values'):
    
    return (1-mean_absolute_error(y_true, y_pred,multioutput)) * 100


# #def extract_fullscene(video_dir, save_dir, num_images=10, image_size = (224, 224)):
    
#     for video_name in tqdm(os.listdir(video_dir)):
    
#         file_name = Path(video_name).stem
        
#         try: 
#             save_path = Path(save_dir).joinpath(file_name)
#             if not save_path.exists():
#                 save_path.mkdir()

#         except OSError:
#             print('Error: creating directory data')

#         cap = cv2.VideoCapture(os.path.join(video_dir,video_name))

#         n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
#         interval = n_frames // num_images
#         frame_index = 0

#         i = 0
#         while cap.isOpened():
#             ret, frame = cap.read()

#             if ret == False:
#                 break
#             if (frame_index % interval == 0) & (i < num_images):
#                 # Resample the frame to 224x224 resolution using OpenCV's resize function
#                 # frame_resized = cv2.resize(frame, image_size )
#                 # frame_resised = cv2.cvtColor(frame_resized, cv2.COLOR_RGB2BGR)

#                 # Save the global scene image to the output folder
#                 output_path = os.path.join(save_path, f"image_{frame_index}.jpg")
#                 cv2.imwrite(output_path, frame)
#                 i += 1
#             frame_index += 1

# #        cap.release()


# def extract_face(image_dir, save_dir, image_size = (224,224)):
#     detector = MTCNN()

#     for folder_name in tqdm(os.listdir(image_dir)):

#         try: 
#             save_path = Path(save_dir).joinpath(folder_name)
#             if not save_path.exists():
#                 save_path.mkdir()
                

#         except OSError:
#             print('Error: creating directory data')
            
        
#         folder = os.path.join(image_dir,folder_name)
#         faces_tmp = []
#         for filename in os.listdir(folder):
#             image_path = os.path.join(folder, filename)
#             # image = Image.open(image_path)
#             image = plt.imread(image_path)
#             image = np.asarray(image)

#             boxes = detector.detect_faces(image)

#             fake_image = np.ones((224,224,3)) # white image 
#             # Crop and save the first detected face
#             if len(boxes) > 0:
#                 x1, y1, width, height = boxes[0]['box']
#                 x2, y2 = x1 + width, y1 + height
#                 face = image[y1:y2, x1:x2]

#                 image = Image.fromarray(face)
#                 image = image.resize(image_size)
#                 image = np.asarray(image)
#                 faces_tmp.append(image) ## add image to tmp list 
#                 plt.imsave(f'{save_path}/face_{filename}', image) 
#             # else:
#             #     if not len(faces_tmp):
#             #         # save the last image ! 
#             #         plt.imsave(f'{save_path}/face_{filename}', faces_tmp[-1])
#             #     else:
#             #         # if not save a  whight image ! 
#             #         plt.imsave(f'{save_path}/face_{filename}', fake_image)


def extract_face_from_video(video_dir, save_dir, num_images=10, image_size=(55, 47)):
    detector = MTCNN()

    for video_name in tqdm(os.listdir(video_dir)):
        file_name = Path(video_name).stem
        save_path = Path(save_dir).joinpath(file_name)
        save_path.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(os.path.join(video_dir, video_name))

        if not cap.isOpened():
            print(f'Error: Unable to open video {video_name}')
            continue

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Pastikan jumlah frame minimal 10
        if total_frames < num_images:
            print(f'Warning: {video_name} memiliki {total_frames} frame, mungkin hasil tidak optimal.')
            frame_indices = np.linspace(0, total_frames - 1, num_images, dtype=int)  # Ambil secara merata
        else:
            frame_indices = np.linspace(0, total_frames - 1, num_images, dtype=int)  # Ambil 10 titik tetap

        faces = []
        last_valid_face = np.ones(image_size + (3,)) * 255  # Default ke gambar putih jika tidak ada wajah sama sekali

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                faces.append(last_valid_face)  # Jika frame gagal diambil, pakai wajah terakhir
                continue

            boxes = detector.detect_faces(frame)
            if len(boxes) > 0:
                x1, y1, width, height = boxes[0]['box']
                x2, y2 = x1 + width, y1 + height
                face = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]

                if face.size > 0:
                    image = Image.fromarray(face)
                    image = image.resize(image_size)
                    image = np.asarray(image)
                    faces.append(image)
                    last_valid_face = image  # Update wajah terakhir yang valid
                else:
                    faces.append(last_valid_face)  # Jika deteksi gagal, pakai wajah terakhir
            else:
                faces.append(last_valid_face)  # Jika tidak ada wajah terdeteksi, pakai wajah terakhir

        cap.release()

        # **Pastikan selalu ada 10 gambar**
        for i, face in enumerate(faces):
            cv2.imwrite(str(save_path / f'face_{i}.jpg'), face)

    print('Face extraction completed.')


# def extract_audio(video_dir, save_dir):
#     for video_name in tqdm(os.listdir(video_dir)):
    
#         file_name = Path(video_name).stem
        
#         if not os.path.exists(save_dir):
#             try:
#                 os.makedirs(save_dir)
#             except OSError:
#                 print(f'Error, creating {save_dir} directory')
                
#         cmd = "ffmpeg -i {}/{}.mp4 -ab 320k -ac 2 -ar 44100 -vn {}/{}.wav"\
#             .format(video_dir, file_name, save_dir, file_name)
#         subprocess.call(cmd, shell=True)


def load_annotations():
    print('Update annotation path !')
    annotation_train = pickle.load(open('./annotations/annotation_training.pkl', 'rb'), encoding='latin1')
    annotation_valid = pickle.load(open('./annotations/annotation_validation.pkl', 'rb'), encoding='latin1')
    annotation_test = pickle.load(open('./annotations/annotation_test.pkl', 'rb'), encoding='latin1')

    return annotation_train, annotation_valid, annotation_test 


def read_ocean_data(video_name, annotation):
    video_name = f"{os.path.basename(video_name)}.mp4"
    scores = [
        annotation["openness"][video_name],
        annotation["conscientiousness"][video_name],
        annotation["extraversion"][video_name],
        annotation["agreeableness"][video_name],
        annotation["neuroticism"][video_name],
    ]

    return np.asarray(scores, np.float32)


def load_images(data_dir, annotation, image_size=(55, 47), batch_size = 8):
    X = []
    y = []

    
    folders = os.listdir(data_dir)

    for img_dir in tqdm(folders):
        p = os.path.join(data_dir, img_dir)
        
        scores = read_ocean_data(p, annotation)
        
        files = os.listdir(p)
        
        # 
        x = []
        for f in files: 
            image = Image.open(os.path.join(p, f))
            image = np.asarray(image, np.float32)
                
            image = cv2.resize(image, (image_size[1], image_size[0]))  # (width, height)

            #image = image / 255.0
            x.append(image) # changed from X to x 
        X.append(x)
        y.append(scores)
    
    X = np.asarray(X, np.float32)
    y = np.asarray(y, np.float32)
    #return X, y

    
    def generator():
        for a, b in zip(X, y):
            yield a, b

    train_ds = tf.data.Dataset.from_generator(generator, output_signature=(
        tf.TensorSpec(shape=(X.shape[1], X.shape[2], X.shape[3], X.shape[4]), dtype=tf.float32),
        tf.TensorSpec(shape=(y.shape[1],), dtype=tf.float32)
    ))
    
    return train_ds.batch(batch_size=batch_size)



# def process_audio(audio_dir, annotation_file):
   
#     vggish_model = hub.load('https://tfhub.dev/google/vggish/1')
    
#     data = []
#     labels = []
    
#     for i, file in tqdm(enumerate(os.listdir(audio_dir)[:])): 
        
#         file_name = Path(file).stem
        
#         audio, sr = librosa.load(Path(audio_dir).joinpath(file), sr=16000, mono=True)

        
#         data_wave = vggish_model(audio)

#         #if data_wave.shape[0] < 15:
#         #    data_wave = tf.concat([data_wave, data_wave[-1:]], axis=0)

#         if data_wave.shape[0] < 15:
#             repetitions = 15 - data_wave.shape[0]
#             last_row = data_wave[-1:]
#             b_repeated = np.repeat(last_row, repetitions, axis=0)
#             b_adjusted = np.vstack([data_wave, b_repeated])

#             data_wave = b_adjusted


#         label = read_ocean_data(file_name, annotation_file)

#         data.append(data_wave)    
#         labels.append(label)
        

#     X = np.stack(data, axis=0)
#     y = np.stack(labels, axis=0)

#     def generator():
#         for a, b in zip(X,y):
#             yield a, b
#     dataset = tf.data.Dataset.from_generator(generator, output_signature=(
#         tf.TensorSpec(shape=(15, 160), dtype=tf.float32),
#         tf.TensorSpec(shape=(5,), dtype=tf.float32)
#     ))  

#     return dataset.batch(batch_size=32)
#     #return np.stack(data), np.stack(labels)

def check_number_of_images(_dir, n=10):
    files = os.listdir(_dir)
    for f in files: 
        l = os.path.join(_dir, f)
        k = os.listdir(l)
        if len(k) < n:
            print(f, len(k))


def load_transcriptions():
    print('Update transcriptions path !')
    transcr_train = pickle.load(open('./transcriptions/transcription_training.pkl', 'rb'), encoding='latin1')
    transcr_valid = pickle.load(open('./transcriptions/transcription_validation.pkl', 'rb'), encoding='latin1')
    transcr_test  = pickle.load(open('./transcriptions/transcription_test.pkl', 'rb'), encoding='latin1')

    return transcr_train, transcr_valid, transcr_test


# def process_text(video_dir, annotation, transc):
#     texts   = []
#     labels = []
    
#     files = os.listdir(video_dir)
    
#     for file in files:
#         texts.append(transc[file]) 
#         labels.append(read_ocean_data(file, annotation))
        
   
#     texts = np.asarray(texts)
#     labels = np.asarray(labels)


#     a = pd.DataFrame(texts, columns=['text'])
#     b = pd.DataFrame(labels, columns=['o','c','e','a','n'])
#     df = pd.concat([a, b], axis=1)

#     return df


# def remove_stop(text):
#     stop = stopwords.words('english')
#     final_text = []
#     for i in text.split():
#         if i.strip().lower() not in stop:
#             final_text.append(i.strip())
#     return " ".join(final_text)
    