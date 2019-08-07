import time
import os
import tensorflow as tf
import json
import pickle
import copy
import numpy as np
import re
from build_vocab import Vocabulary
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

def lstm_cell(size):
    return tf.contrib.rnn.BasicLSTMCell(size, reuse=tf.AUTO_REUSE)

def create_rationalization_matrix(rationalizations, max_rationalization_len, vocab):
    rationalization_matrix = []
    sequence_lenght_arr = []
    for sent in rationalizations:
        sent = sent.replace("'","")
        sent = sent.strip()
        words = sent.lower().split(' ')
        
        ration_sent = np.zeros([ max_rationalization_len ], dtype=np.int32)
        ration_sent[0] = 1
        idx = 1
    
        for k, word in enumerate(words):
            if idx == max_rationalization_len:
                break
            if word in vocab.word2idx:
                ration_sent[idx] = vocab.word2idx[word]
            else:
                ration_sent[idx] = 3
            idx +=1
        
        if idx < max_rationalization_len:
            ration_sent[idx] = 2
            idx += 1
        rationalization_matrix.append(ration_sent)
        sequence_lenght_arr.append(idx)

    return rationalization_matrix, sequence_lenght_arr

class Reverse_Frogger:

    def __init__(self):
        self.explanation_lstm_dim = 512
        self.batch_size = 1
        self.width = 320
        self.height = 320
        self.depth = 3
        self.max_words = 40
        self.n_words = 1000
        self.word_embed_dim = 512
        self.fc1_size = 4096
        self.image_embedding_size = 2048
        self.dim_hidden = 512
        self.comb_embedding_size = self.explanation_lstm_dim * 4 + self.dim_hidden
        self.num_output = 5
        self.drop_out_rate = 0.7
        self.max_rationalization_len = self.max_words
        self.learning_rate = 0.001
        self.max_epoch = 101

        self.explanation_lstm = tf.contrib.rnn.MultiRNNCell([lstm_cell(self.explanation_lstm_dim) for _ in range(2)])
        self.W_embeddings = tf.Variable(tf.constant(0.0, shape=[self.n_words, self.word_embed_dim]), trainable=True, name="wemb")

        self.embed_image_W = tf.Variable(tf.random_uniform([self.image_embedding_size, self.dim_hidden], -0.08, 0.08), name='embed_image_W')
        self.embed_image_b = tf.Variable(tf.random_uniform([self.dim_hidden], -0.08, 0.08), name='embed_image_b')

        self.embed_scor_W = tf.Variable(tf.random_uniform([self.comb_embedding_size, self.num_output], -0.08, 0.08), name='embed_scor_W')
        self.embed_scor_b = tf.Variable(tf.random_uniform([self.num_output], -0.08, 0.08), name='embed_scor_b')
        #with open('data/vocab_frogger.pkl', 'rb') as f:
        #    vocab = pickle.load(f)

    def image_encoder(self, img_input):
        with tf.name_scope("conv1"):
            conv1 = tf.layers.conv2d(img_input, 32, 3, activation=tf.nn.relu)
            conv1 = tf.layers.conv2d(conv1, 32, 3, activation=tf.nn.relu)
            conv1 = tf.layers.max_pooling2d(conv1, 2, 2)
            print("conv1: ",conv1)
        with tf.name_scope("conv2"):
            conv2 = tf.layers.conv2d(conv1, 64, 3, activation=tf.nn.relu)
            conv2 = tf.layers.conv2d(conv2, 64, 3, activation=tf.nn.relu)
            conv2 = tf.layers.max_pooling2d(conv2, 2, 2)
            print("conv2: ",conv2)
        with tf.name_scope("conv3"):
            conv3 = tf.layers.conv2d(conv2, 128, 3, activation=tf.nn.relu)
            conv3 = tf.layers.conv2d(conv3, 128, 3, activation=tf.nn.relu)
            conv3 = tf.layers.max_pooling2d(conv3, 2, 2)
            print("conv3: ",conv3)
        with tf.name_scope("conv4"):
            conv4 = tf.layers.conv2d(conv3, 128, 3, activation=tf.nn.relu)
            conv4 = tf.layers.max_pooling2d(conv4, 2, 2)
            print("conv4: ",conv4)
        with tf.name_scope("conv5"):
            conv5 = tf.layers.conv2d(conv4, 256, 3, activation=tf.nn.relu)
            conv5 = tf.layers.max_pooling2d(conv5, 2, 2)
            print("conv5: ",conv5)
        with tf.name_scope("flatten"):
            flattened = tf.contrib.layers.flatten(conv5)
            print("flattened: ", flattened)
            
        with tf.name_scope("fc1"):
            fc1 = tf.layers.dense(flattened, self.fc1_size) 
        with tf.name_scope("fc"):
            fc = tf.layers.dense(fc1, self.image_embedding_size)
        
        return fc
    
    def build_model(self):
        image = tf.placeholder(tf.float32, [self.batch_size, self.width, self.height, self.depth])
        explanation = tf.placeholder(tf.int32, [self.batch_size, self.max_words])
        explanation_sequence_length = tf.placeholder(tf.int32, [self.batch_size])
        action = tf.placeholder(tf.int32, [self.batch_size]) 
            
        state = explanation_lstm.zero_state(self.batch_size, tf.float32)
        loss = 0.0
        with tf.variable_scope("encoder"):
            text_embedding = tf.nn.embedding_lookup(self.W_embeddings, explanation)
            
            with tf.variable_scope('explanation_context'):
                exp_outputs, state = tf.nn.dynamic_rnn(self.explanation_lstm, 
                                                    inputs = text_embedding, 
                                                    sequence_length= explanation_sequence_length,
                                                    initial_state = state,
                                                    dtype=tf.float32)
            
            exp_embedding = tf.reshape(tf.transpose(state, [2, 1, 0, 3]), [self.batch_size, -1])
            
            image_emb = self.image_encoder(image)
            
            image_emb = tf.nn.xw_plus_b(image_emb, self.embed_image_W, self.embed_image_b)
            image_emb = tf.nn.relu(image_emb)
            
            comb_emb = tf.concat([image_emb, exp_embedding], 1)
            comb_emb = tf.nn.dropout(comb_emb, 1 - self.drop_out_rate)
            
            scores_emb = tf.nn.xw_plus_b(comb_emb, self.embed_scor_W, self.embed_scor_b) 
            
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=action, logits=scores_emb)
            
            _action = tf.reshape(action, [batch_size, -1])
            onehot_action = tf.one_hot(_action, depth = self.num_output)
            onehot_action = tf.reshape(onehot_action, [batch_size, -1])
            
            prediction = tf.nn.softmax(scores_emb)
            correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(onehot_action, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

            # Calculate loss
            loss = tf.reduce_mean(cross_entropy)
            
        return loss, accuracy, image, explanation, explanation_sequence_length, action, prediction
    
    def build_generator(self, batch_size):
        image = tf.placeholder(tf.float32, [batch_size, self.width, self.height, self.depth])
        explanation = tf.placeholder(tf.int32, [batch_size, self.max_words])
        explanation_sequence_length = tf.placeholder(tf.int32, [batch_size])
        action = tf.placeholder(tf.int32, [batch_size]) 
            
        state = explanation_lstm.zero_state(batch_size, tf.float32)
        loss = 0.0
        with tf.variable_scope("encoder"):
            text_embedding = tf.nn.embedding_lookup(self.W_embeddings, explanation)
            
            with tf.variable_scope('explanation_context'):
                exp_outputs, state = tf.nn.dynamic_rnn(self.explanation_lstm, 
                                                    inputs = text_embedding, 
                                                    sequence_length= explanation_sequence_length,
                                                    initial_state = state,
                                                    dtype=tf.float32)
            
            exp_embedding = tf.reshape(tf.transpose(state, [2, 1, 0, 3]), [batch_size, -1])
            
            image_emb = self.image_encoder(image)
            
            image_emb = tf.nn.xw_plus_b(image_emb, self.embed_image_W, self.embed_image_b)
            image_emb = tf.nn.relu(image_emb)
            

            comb_emb = tf.concat([image_emb, exp_embedding], 1)
            #comb_emb = tf.nn.dropout(comb_emb, 1 - drop_out_rate)
            
            scores_emb = tf.nn.xw_plus_b(comb_emb, self.embed_scor_W, self.embed_scor_b) 
            
            _action = tf.reshape(action, [batch_size, -1])
            onehot_action = tf.one_hot(_action, depth = self.num_output)
            onehot_action = tf.reshape(onehot_action, [batch_size, -1])
            
            prediction = tf.nn.softmax(scores_emb)
            result_action = tf.argmax(prediction, 1)
            correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(onehot_action, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        return accuracy, image, explanation, explanation_sequence_length, action, prediction, result_action
    
    def build_generator_v2(self, batch_size):
        image = tf.placeholder(tf.float32, [batch_size, self.width, self.height, self.depth])
        explanation = tf.placeholder(tf.int32, [batch_size, self.max_words])
        explanation_sequence_length = tf.placeholder(tf.int32, [batch_size])
            
        state = self.explanation_lstm.zero_state(batch_size, tf.float32)
        loss = 0.0
        with tf.variable_scope("encoder"):
            text_embedding = tf.nn.embedding_lookup(self.W_embeddings, explanation)
            
            with tf.variable_scope('explanation_context'):
                exp_outputs, state = tf.nn.dynamic_rnn(self.explanation_lstm, 
                                                    inputs = text_embedding, 
                                                    sequence_length= explanation_sequence_length,
                                                    initial_state = state,
                                                    dtype=tf.float32)
            
            exp_embedding = tf.reshape(tf.transpose(state, [2, 1, 0, 3]), [batch_size, -1])
            
            image_emb = self.image_encoder(image)
            
            image_emb = tf.nn.xw_plus_b(image_emb, self.embed_image_W, self.embed_image_b)
            image_emb = tf.nn.relu(image_emb)
            

            comb_emb = tf.concat([image_emb, exp_embedding], 1)
            #comb_emb = tf.nn.dropout(comb_emb, 1 - drop_out_rate)
            
            scores_emb = tf.nn.xw_plus_b(comb_emb, self.embed_scor_W, self.embed_scor_b) 
            prediction = tf.nn.softmax(scores_emb)
            result_action = tf.argmax(prediction, 1)
            

        return image, explanation, explanation_sequence_length, result_action
    
    def train(self, cur_image_dir, next_image_dir):
    
        good_ids, training_indices, testing_indices, training_rationalizations, testing_rationalizations, trn_act, tst_act, vocab = load_data("Turk_Master_File.xlsx", 'data/vocab_frogger.pkl')
        cur_training_images, next_training_images, cur_test_images, next_test_images = load_images(current_image_dir, next_image_dir, good_ids, training_indices)
        #print("training ration: ", len(training_rationalizations))
        #print("current_training_images : ",len(cur_training_images))
        #print("next_training_images : ", next_training_images)
        rationalization_matrix, ration_sqn_len = create_rationalization_matrix(training_rationalizations, self.max_rationalization_len, vocab)
        num_train = len(training_rationalizations)
        rationalization_matrix = np.array(rationalization_matrix)
        ration_sqn_len = np.array(ration_sqn_len)
        cur_training_images = np.array(cur_training_images)
        trn_act = np.array(trn_act)
        tf_loss, tf_acc, tf_image, tf_explanation, tf_explanation_sequence_length, tf_action, tf_pred = build_model()
        
        sess = tf.InteractiveSession()
        
        with tf.device('/cpu:0'):
            saver = tf.train.Saver(max_to_keep=100)

        train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(tf_loss)
        tf.global_variables_initializer().run()
        loss_arr = []
        acc_arr = []
        for epoch in range(0, max_epoch):
            rationalization_matrix, ration_sqn_len, cur_training_images, trn_act = shuffle(
                rationalization_matrix, ration_sqn_len, cur_training_images, trn_act)

            #print(np.any(np.isnan(trn_act)))
            tStart = time.time()
            niter = 0
            sum_loss = 0
            sum_accuracy = 0
            for current_batch_start_idx in range(0,num_train,batch_size):
                if current_batch_start_idx + batch_size < num_train:
                    current_batch_file_idx = range(current_batch_start_idx,current_batch_start_idx+batch_size)
                else:
                    current_batch_file_idx = range(current_batch_start_idx,num_train)

                #print(current_batch_file_idx)

                current_ration_text = rationalization_matrix[current_batch_file_idx,:]
                current_sqn_len = ration_sqn_len[current_batch_file_idx]
                current_img_list = cur_training_images[current_batch_file_idx]
                current_actions = trn_act[current_batch_file_idx]
                current_imgs_data = []
                for img in current_img_list:
                    img_path = os.path.join(cur_image_dir, img)
                    img_arr = cv2.imread(img_path)
                    resized_img = cv2.resize(img_arr, (320,320))
                    current_imgs_data.append(resized_img)

                current_imgs_data = np.array(current_imgs_data)
                
                
                if len(current_ration_text) == batch_size:
                    _, loss, accuracy, prediction = sess.run([train_op, tf_loss, tf_acc, tf_pred],
                        feed_dict={
                            tf_image: current_imgs_data,
                            tf_explanation: current_ration_text,
                            tf_explanation_sequence_length: current_sqn_len,
                            tf_action: current_actions})
                    niter +=1
                    sum_loss += loss
                    sum_accuracy += accuracy
                    #print("iter: ", niter, " loss: ", loss, " accuracy: ", accuracy)
                    #print("current_actions", current_actions)
                    #print("prediction", prediction)
                    
            
            avg_loss = sum_loss / niter
            avg_acc = sum_accuracy/ niter
            print("epoch: ", epoch, " loss: ", avg_loss, " accuracy: ", avg_acc, " time: ", time.time() - tStart)
            f_acc= open("train_acc.txt","a+")
            f_acc.write("epoch: "+ str(epoch) + " "+"Accuracy is: " + str(avg_acc)+ "%\n")
            f_acc.close()
            
            loss_arr.append(avg_loss)
            acc_arr.append(avg_acc)
            
            if np.mod(epoch, 10) == 0:
                print ("Epoch ", epoch, " is done. Saving the model ...")
                saver.save(sess, os.path.join(model_path, 'frogger_model'), global_step=epoch)
            
        return loss_arr, acc_arr
    
    def test(self, cur_image_dir, next_image_dir):
        good_ids, training_indices, testing_indices, training_rationalizations, testing_rationalizations, trn_act, tst_act, vocab = load_data("Turk_Master_File.xlsx", 'data/vocab_frogger.pkl')
        cur_training_images, next_training_images, cur_test_images, next_test_images = load_images(current_image_dir, next_image_dir, good_ids, training_indices)
        
        rationalization_matrix, ration_sqn_len = create_rationalization_matrix(testing_rationalizations, self.max_rationalization_len, vocab)
        num_test = len(testing_rationalizations)
        rationalization_matrix = np.array(rationalization_matrix)
        ration_sqn_len = np.array(ration_sqn_len)
        cur_training_images = np.array(cur_training_images)
        trn_act = np.array(trn_act)
        batch_size = 1
        tf_accuracy, tf_image, tf_explanation, tf_explanation_sequence_length, tf_action, tf_prediction, tf_out_action = build_generator(batch_size)
        
        sess = tf.Session()
        with tf.device('/cpu:0'):
            saver = tf.train.Saver()
            saver.restore(sess, os.path.join(model_path, 'frogger_model-100'))


            
        tStart = time.time()
        
        sum_accuracy = 0
        niter = 0
        
        for current_batch_start_idx in range(0,num_test,batch_size):
            if current_batch_start_idx + batch_size < num_test:
                current_batch_file_idx = range(current_batch_start_idx,current_batch_start_idx+batch_size)
            else:
                current_batch_file_idx = range(current_batch_start_idx,num_test)

                #print(current_batch_file_idx)

            current_ration_text = rationalization_matrix[current_batch_file_idx,:]
            current_sqn_len = ration_sqn_len[current_batch_file_idx]
            current_img_list = cur_training_images[current_batch_file_idx]
            current_actions = trn_act[current_batch_file_idx]
            current_imgs_data = []
            for img in current_img_list:
                img_path = os.path.join(cur_image_dir, img)
                img_arr = cv2.imread(img_path)
                resized_img = cv2.resize(img_arr, (320,320))
                current_imgs_data.append(resized_img)

            current_imgs_data = np.array(current_imgs_data)
                
                
            if len(current_ration_text) == batch_size:
                accuracy, prediction, out_action = sess.run([tf_accuracy, tf_prediction, tf_out_action],
                                                feed_dict={
                                                    tf_image: current_imgs_data,
                                                    tf_explanation: current_ration_text,
                                                    tf_explanation_sequence_length: current_sqn_len,
                                                    tf_action: current_actions})
                print("original actions: ", current_actions)
                print("prediction: ", prediction)
                print("output action: ", out_action)
                niter +=1
                sum_accuracy += accuracy
                    
            
        avg_acc = sum_accuracy/ niter
        print("accuracy: ", avg_acc, " time: ", time.time() - tStart)
            
        return

    def inference(self, testing_rationalizations, test_images):
        model_path = './models_batch/'
        vocab = Vocabulary()
        with open('data/vocab_frogger.pkl', 'rb') as f:
            vocab = pickle.load(f)
        rationalization_matrix, ration_sqn_len = create_rationalization_matrix(testing_rationalizations, self.max_rationalization_len, vocab)
        num_test = len(testing_rationalizations)
        rationalization_matrix = np.array(rationalization_matrix)
        ration_sqn_len = np.array(ration_sqn_len)
        test_images = np.array(test_images)
        batch_size = 1
        tf_image, tf_explanation, tf_explanation_sequence_length, tf_action = self.build_generator_v2(batch_size)
        
        sess = tf.Session()
        with tf.device('/cpu:0'):
            saver = tf.train.Saver()
            saver.restore(sess, os.path.join(model_path, 'frogger_model-100'))


            
        tStart = time.time()
        actions = []
        print("num_test: ",num_test)
        for current_batch_start_idx in range(0,num_test):
            print("in")
            current_batch_file_idx = current_batch_start_idx
            

            current_ration_text = rationalization_matrix[current_batch_file_idx]
            current_ration_text = np.array(current_ration_text)
            current_ration_text = np.reshape(current_ration_text, (batch_size, -1))
            current_sqn_len = ration_sqn_len[current_batch_file_idx]
            current_sqn_len = np.reshape(current_sqn_len, (batch_size))
            #current_sqn_len = np.array(current_sqn_len)
            current_imgs_data = test_images[current_batch_file_idx]
            current_imgs_data = np.array(current_imgs_data)
            current_imgs_data = np.reshape(current_imgs_data, (batch_size, self.width, self.height, self.depth))
            print(current_imgs_data.shape)    
            print(current_sqn_len.shape)    
            print("current_sqn: ",current_sqn_len)
            print("batch_size: ",batch_size)
            print("len: ", len(current_ration_text))
            if len(current_ration_text) == batch_size:
                out_action = sess.run([tf_action],
                feed_dict={
                    tf_image: current_imgs_data,
                    tf_explanation: current_ration_text,
                    tf_explanation_sequence_length: current_sqn_len})
                
                print("output action: ", out_action)
                actions.append(out_action)
                
                
        print(" time: ", time.time() - tStart)
            
        return actions
    def test_rf(self):
        print("rf")

def test():
    print("test")