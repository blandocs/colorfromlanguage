import pickle
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms

def color_transform(color_img):
    input_size = 224
    t = transforms.Compose([transforms.Resize((input_size, input_size))])
    return t(color_img)
def load_images(load_type, file_id_list):

    if load_type == 'train':
        data_dir = 'rgb_train'

    elif load_type == 'val':
        data_dir = 'rgb_test'
    else:
        return


    image_files = []
    loaded_file_ids = []
    image_count = 0

    for file_id in file_id_list:
        color_path = Path(f"../dataset/{data_dir}/{file_id}.png")
        # print(color_path)
        if not (color_path).exists():
            continue


        color_img = Image.open(color_path).convert('RGB')
        # pix = np.array(color_img)
        # print(pix.shape)
        color_img = color_transform(color_img)
        # print(color_img)
        rgb = np.array(color_img, dtype= 'uint8')
        bgr = rgb[...,::-1]

        # print(pix.shape)
        image_files.append(bgr)
        loaded_file_ids.append(file_id)

        image_count += 1

        if image_count % 100 == 0:
            print(image_count)

    return np.array(image_files), loaded_file_ids

def get_file_id_list(load_type):
    
    f = open('tag2sentence.pkl', 'rb')
    pkl = pickle.load(f)
    sentence_data = pkl[load_type]
    return sentence_data.keys()


def load_words(load_type, file_id_list, train_vocab):
    f = open('tag2sentence.pkl', 'rb')
    pkl = pickle.load(f)

    sentence_data = pkl[load_type]    
    max_length = 20

    train_words = []
    train_lengths = []
    contain_indexes = []

    for i, file_id in enumerate(file_id_list):
        nl = sentence_data[file_id]
        word_to_index = [0] * max_length

        length = 0

        if len(nl) >= 20:
            # del_indexes.append(i)
            continue
        else:
            contain_indexes.append(i)
            

        for word in nl:
            embedding_index = train_vocab[word]
            word_to_index[length] = embedding_index

            length += 1



        train_words.append(word_to_index)
        train_lengths.append(length)

    train_words = np.array(train_words, dtype=np.uint32)
    train_lengths = np.array(train_lengths, dtype=np.uint8)

    # print(train_words, train_lengths)
    return train_words, train_lengths, contain_indexes



