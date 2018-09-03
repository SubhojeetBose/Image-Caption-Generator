from text_extractor import load_txt
from pickle import load
from pickle import dump
from numpy import array
from numpy import argmax
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Dropout
from keras.layers.merge import add
from keras.callbacks import ModelCheckpoint
from nltk.translate.bleu_score import corpus_bleu
#the dataset contain Flickr_8k.trainImages.txt and Flickr_8k.devImages.txt files
#they contain the nameof the seperated out train and val set

#Load set of photo name in a given set
def load_set(filename):
	txt = load_txt(filename)
	img_set = list()
	for line in txt.split('\n'):
		if len(line) < 1:
			continue
		# get the image name
		name = line.split('.')[0]
		img_set.append(name)
	return set(img_set)
 
#load clean descriptions of iamges in memory, we need to put stseq and endseq to inform the sequence generator
def load_clean_descrip(filename, dataset):
	txt = load_txt(filename)
	clean = dict()
	for line in txt.split('\n'):
		tokens = line.split()
		img_id, img_desc = tokens[0], tokens[1:]
		#Skip images not in the set
		if img_id in dataset:
			#Create list if not present
			if img_id not in clean:
				clean[img_id] = list()
			#Wrap description in tokens
			desc = 'startseq ' + ' '.join(img_desc) + ' endseq'
			clean[img_id].append(desc)
	return clean

#load photo features
def load_features(filename, dataset):
    #load all features and filter them
    all_feat = load(open(filename, 'rb'))
    feat = {k: all_feat[k] for k in dataset}
    return feat

#We have to one hot encode each word in dexcription in order that rnn can use it to print sequence
#Here image features will be encoded with respective words and the rnn will try to predict the nxt sequence based on the feature encoding

#tokenize and save the instance
def tokenize(descriptions):
    desc_text = list(descriptions.values())
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(desc_text)
    dump(tokenizer,open('tokenizer.pkl', 'wb'))
    return tokenizer

#convert a given int value to corresponding word
def inv_tokenize(val, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == val:
            return word
    return None

#tokenizing and then onehot encoding the descriptions
def create_seq(vocab_size,max_len,tokenizer,descriptions,photos):
    X1, X2, y = list(), list(), list()  #list to store input image and input seq for which it predicts a output word
    for key, desc_list in descriptions.item():
        for desc in desc_list:                      #encode each description
            seq = tokenizer.texts_to_sequences([desc])[0]
            #split seq in multiple x and y pairs
            for i in range(1, len(seq)):
                in_seq, out_seq = seq[:i], seq[i]     #divide string in prefix and nxt element pair
                #pad the in sequences so that all have max_len
                in_seq = pad_sequences([in_seq], maxlen=max_len)[0]   #we use prepadding by-default
                #one hot encode the output only as only nxt word have probability of get chosen
                out_seq = to_categorical([out_seq], num_classes=vocab_size)
                #store
                X1.append(photos[key][0])
                X2.append(in_seq)
                y.append(out_seq)
    return array(X1), array(X2), array(y)



#We will use merge model as a image caption generation. in merge model, rnn is used to encode the word seq
#seperate multimodal neural network is used to combine the image feature and rnn encoded word
    
#We are using 4096*256 image layer and max_len*256 rnn encoding layer and multimaodal layer of 256*vocab_size
def define_model(vocab_size, max_len):
    #-----------model---------
    #Feature extractor model
	inputs1 = Input(shape=(4096,))
	fe1 = Dropout(0.5)(inputs1)
	fe2 = Dense(256, activation='relu')(fe1)
	#Sequence model
	inputs2 = Input(shape=(max_len,))
	se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
	se2 = Dropout(0.5)(se1)
	se3 = LSTM(256)(se2)
	#Decoder model(multimodal neural network)
	decoder1 = add([fe2, se3])
	decoder2 = Dense(256, activation='relu')(decoder1)
	outputs = Dense(vocab_size, activation='softmax')(decoder2)
	#Tie it together [image, seq] [word]
	model = Model(inputs=[inputs1, inputs2], outputs=outputs)
	model.compile(loss='categorical_crossentropy', optimizer='adam')
	#Summarize model
	print(model.summary())
	plot_model(model, to_file='model.png', show_shapes=True)
	return model

#To generate description for new image we recursively feed the till formed seq of word and predict the next word in the line

#Generate a description for an image
def generate_desc(model, tokenizer, photo, max_length):
	#Give the st art of the description to predict the next word based on photo feature and
	in_text = 'startseq'
	#Iterate till max_length or end_seq whichever comes first and this recursively call the .predict to predict the nest word in the line
	for i in range(max_length):
		#Integer encode input sequence
		seq = tokenizer.texts_to_sequences([in_text])[0]
		#Pad input as in create_seq
		seq = pad_sequences([seq], maxlen=max_length)
		#Predict nxext word based on photo features and prefix of result seq till now
		yhat = model.predict([photo,seq], verbose=0)
		#Convert probability to integer as the seq is one hot encoded on all the vocab, we choose the one with highest prob 
		yhat = argmax(yhat)
		#Map the index with max probab to word
		word = inv_tokenize(yhat, tokenizer)
		#Stop if we cannot map the word
		if word is None:
			break
		#Append as input for generating the next word
		in_text += ' ' + word
		#Stop if we predict the end of the sequence
		if word == 'endseq':
			break
	return in_text


#We use BLEU score for evaluating the result
def evaluate_model(model, descriptions, photos, tokenizer, max_length):
	actual, predicted = list(), list()
	# step over the whole set
	for key, descr_list in descriptions.items():
		#predicted description
		yhat = generate_desc(model, tokenizer, photos[key], max_length)
		#actual description
		references = [d.split() for d in descr_list]
		actual.append(references)
		predicted.append(yhat.split())
	#calculate BLEU score
	print('BLEU-1 Score: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
	print('BLEU-2 Score: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
	print('BLEU-3 Score: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
	print('BLEU-4 Score: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))



 
#TRAIN DATASET

#load train data, its feature and descriptions
filename = 'Flickr8k_text/Flickr_8k.trainImages.txt'
train = load_set(filename)
print('Dataset: %d' % len(train))
# descriptions
train_descr = load_clean_descrip('descriptions.txt', train)
print('Descriptions: train=%d' % len(train_descr))
#tokenize train data
tokenizer = tokenize(train_descr)
#save the tokenizer for further use
dump(tokenizer, open('tokenizer.pkl', 'wb'))
#forming a list of descrip
desc_text = list(train_descr.values())
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)
max_len = max(len(d.split()) for d in desc_text)
print('Description Length: %d' % max_len)
# photo features
train_feat = load_features('features.pkl', train)
print('Photos: train=%d' % len(train_feat))
#prepare sequence
X1train, X2train, ytrain = create_seq(vocab_size, max_len, tokenizer, train_descr, train_feat)




#VAL DATASET

#load validation data, its feature and descriptions
filename = 'Flickr8k_text/Flickr_8k.devImages.txt'
val = load_set(filename)
print('Dataset: %d' % len(val))
# descriptions
val_descr = load_clean_descrip('descriptions.txt', val)
print('Descriptions: val=%d' % len(val_descr))
# photo features
val_feat = load_features('features.pkl', val)
print('Photos: val=%d' % len(val_feat))
#prepare sequence
X1val, X2val, yval = create_seq(vocab_size, max_len, tokenizer, val_descr, val_feat)



#TEST DATASET

#load test data, its feature and descriptions
filename = 'Flickr8k_text/Flickr_8k.testImages.txt'
test = load_set(filename)
print('Dataset: %d' % len(test))
# descriptions
test_descr = load_clean_descrip('descriptions.txt', test)
print('Descriptions: test=%d' % len(test_descr))
# photo features
test_feat = load_features('features.pkl', test)
print('Photos: test=%d' % len(test_feat))



#FIT MODEL

#define model
model = define_model(vocab_size, max_len)
# define checkpoint callback
filepath = 'model-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
# fit model
model.fit([X1train, X2train], ytrain, epochs=20, verbose=2, callbacks=[checkpoint], validation_data=([X1val, X2val], yval))



#EVALUATE MODEL

#load the best model

#evaluate model
evaluate_model(model, test_descr, test_feat, tokenizer, max_len)