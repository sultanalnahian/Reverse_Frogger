from xlrd import open_workbook
from collections import Counter
import os
import json
import pickle
import nltk
from build_vocab import Vocabulary
from math import floor
import cv2

def load_images(current_image_dir, next_image_dir, good_ids, tr_indices):
    current_images = os.listdir(current_image_dir)
    current_images = sorted(current_images ,key = numericalSort)
    
    next_images = os.listdir(next_image_dir)
    next_images = sorted(next_images ,key = numericalSort)
    num_images = len(current_images)
    
    cur_training_images = []
    next_training_images = []
    cur_test_images = []
    next_test_images = []
    for i, file in enumerate(current_images):
        if i in good_ids:
            if good_ids[tr_indices[0]]<=i and i<=good_ids[tr_indices[1]]:
                cur_training_images.append(file)
                next_training_images.append(next_images[i])
            else:
                cur_test_images.append(file)
                next_test_images.append(next_images[i])
    
    return cur_training_images, next_training_images, cur_test_images, next_test_images

def load_data(data_file, vocab_path):
    wb = open_workbook(data_file)
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    #vocab = vocab
    #read the rationalizations from the excel file and create a list of training/testing rationalizations. 
    for sheet in wb.sheets():
        number_of_rows = sheet.nrows
        number_of_columns = sheet.ncols
        rationalizations = []
        items = []
        rows = []
        lengths = []
        max_length = 0
        
        bad_worker_ids = ['A2CNSIECB9UP05','A23782O23HSPLA','A2F9ZBSR6AXXND','A3GI86L18Z71XY','AIXTI8PKSX1D2','A2QWHXMFQI18GQ','A3SB7QYI84HYJT',
'A2Q2A7AB6MMFLI','A2P1KI42CJVNIA','A1IJXPKZTJV809','A2WZ0RZMKQ2WGJ','A3EKETMVGU2PM9','A1OCEC1TBE3CWA','AE1RYK54MH11G','A2ADEPVGNNXNPA',
'A15QGLWS8CNJFU','A18O3DEA5Z4MJD','AAAL4RENVAPML','A3TZBZ92CQKQLG','ABO9F0JD9NN54','A8F6JFG0WSELT','ARN9ET3E608LJ','A2TCYNRAZWK8CC',
'A32BK0E1IPDUAF','ANNV3E6CIVCW4','AXMQBHHU22TSP','AKATSYE8XLYNL','A355PGLV2ID2SX','A55CXM7QR7R0N','A111ZFNLXK1TCO']
        
        good_ids = []
        good_rationalizations = []
        actions = []
        counter = Counter()
        for row in range(1, number_of_rows):
            values = []
            worker_id = sheet.cell(row,0).value
            if worker_id not in bad_worker_ids:
                good_ids.append(row-1)
                line = sheet.cell(row,4).value
                tokens = nltk.tokenize.word_tokenize(line.lower())
                # if tokens!=[]:
                _action = sheet.cell(row,2).value
                actions.append(actions_map[_action])
                line = line.lower()
                good_rationalizations.append(line)
                line = re.sub('[^a-z\ ]+', " ", line)
                words = line.split()
                length = len(tokens)
                lengths.append(length)
                if length>max_length:
                    max_length = length
                for index,word in enumerate(tokens): 
                    tokens[index] = vocab.word2idx[word]
                rationalizations.append(words)
        rationalizations=[np.array(xi) for xi in rationalizations]

    split = int(floor((90.0/100)*len(rationalizations)))
    
    # zzzz = nltk.tokenize.word_tokenize(' lksdfjoisd posidjf')
    # print(zzzz)
    # exit(0)
    tr = slice(0,split)
    tr_indices = [0,split-1]
    te_indices = [split,len(rationalizations)-1]
    te = slice(split,len(rationalizations))
    training_rationalizations = good_rationalizations[tr]
    testing_rationalizations = good_rationalizations[te]
    training_actions = actions[tr]
    testing_actions = actions[te]
    # print(good_rationalizations)
    # print(self.training_rationalizations)
    # for r in self.training_rationalizations:
    # 	if r==None:
    # 		print("first")
    # 		exit(0)
    # exit(0)
    training_rationalizations_text = good_rationalizations[tr]
    testing_rationalizations_text = good_rationalizations[te]
    
    
    #current_image_dir = self.current_image_dir
    #next_image_dir = self.next_image_dir
    #output_dir = self.output_dir
    #concatenated_images_dir = self.concatenated_images_dir
    #subtracted_images_dir = self.subtracted_training_images_dir
    #image_size = [self.image_size, self.image_size]
    #image preprocessing
    #crop and resize images. 
    
    return good_ids, tr_indices, te_indices, training_rationalizations, testing_rationalizations, training_actions, testing_actions, vocab
    
