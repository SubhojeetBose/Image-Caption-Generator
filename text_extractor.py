import string

#load txt file 
def load_txt(filename):
    file = open(filename, 'r')
    txt = file.read()
    file.close()
    return txt

#extract text descriptions
def load_descrip(text):
    mapping = dict()
    #process each lines
    for line in text.split('\n'):
        #split around space and store image_id and image_des
        tokens = line.split()
        if len(line)<2:
            continue
        img_id, img_desc = tokens[0], tokens[1:]
        img_desc = ' '.join(img_desc)
        img_id = img_id.split('.')[0]
        #create a new mapping if needed
        if img_id not in mapping:
            mapping[img_id] = list()
        mapping[img_id].append(img_desc)
    return mapping

#cleaning the text data
def clean_descrip(descr):
	# prepare translation table for removing punctuation
	table = str.maketrans('', '', string.punctuation)    #it first maps first->second and then removes the elements in 3rd string
	for key, desc_list in descr.items():
		for i in range(len(desc_list)):
        #cleaning, converting to lowercase and removing one character words
			clean = desc_list[i]
			clean = clean.split()
			clean = [word.lower() for word in clean]
			# remove punctuation from each token
			clean = [w.translate(table) for w in clean]
			clean = [word for word in clean if len(word)>1]
			clean = [word for word in clean if word.isalpha()]
			# store as string
			desc_list[i] =  ' '.join(clean)

#convert to a set to check the number of descriptions
def to_vocab(descr):
    all_vocab = set()
    for key in descr.keys():
        [all_vocab.update(d.split()) for d in descr[key]]
    return all_vocab

#save the descrips to file, one line per new line
def save_descrip(descr, filename):
	lines = list()
	for key, desc_list in descr.items():
		for desc in desc_list:
			lines.append(key + ' ' + desc)
	data = '\n'.join(lines)
	file = open(filename, 'w')
	file.write(data)
	file.close()
    
#calling function to extract features and store them to a file
filename = 'Flickr8k_text\Flickr8k.token.txt'
text = load_txt(filename)
descriptions = load_descrip(text)
print('Loaded: %d' %len(descriptions))
clean_descrip(descriptions)
vocab = to_vocab(descriptions)#summary of the total number of description
print('Vocab size: %d' %len(vocab))
save_descrip(descriptions,'descriptions.txt')