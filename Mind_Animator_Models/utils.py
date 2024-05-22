import numpy as np
import random
from nltk.corpus import wordnet
from nltk.corpus import stopwords
import torch
from torch.autograd import Variable

device = torch.device('cuda:2')
stop_words = stopwords.words('english')


def synonym_replacement(words, n):
    new_words = words.copy()
    random_word_list = list(set([word for word in words if word not in stopwords.words('english')]))
    random.shuffle(random_word_list)
    num_replaced = 0
    for random_word in random_word_list:
        synonyms = get_synonyms(random_word)
        if len(synonyms) >= 1:
            synonym = random.choice(synonyms)
            new_words = [synonym if word == random_word else word for word in new_words]
            num_replaced += 1
        if num_replaced >= n:
            break
    sentence = ' '.join(new_words)
    new_words = sentence.split(' ')
    return new_words

def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonym = lemma.name().replace("_", " ").replace("-", " ").lower()
            synonym = "".join([char for char in synonym if char in 'abcdefghijklmnopqrstuvwxyz '])
            synonyms.add(synonym)
    if word in synonyms:
        synonyms.remove(word)
    return list(synonyms)

def random_deletion(words, p):
    if len(words) == 1:
        return words
    new_words = []
    for word in words:
        r = random.uniform(0, 1)
        if r > p:
            new_words.append(word)
    if len(new_words) == 0:
        rand_int = random.randint(0, len(words)-1)
        return [words[rand_int]]
    return new_words

def random_swap(words, n):
    new_words = words.copy()
    for _ in range(n):
        new_words = swap_word(new_words)
    return new_words

def swap_word(new_words):
    random_idx_1 = random.randint(0, len(new_words)-1)
    random_idx_2 = random_idx_1
    counter = 0
    while random_idx_2 == random_idx_1:
        random_idx_2 = random.randint(0, len(new_words)-1)
        counter += 1
        if counter > 3:
            return new_words
    new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1]
    return new_words

def random_insertion(words, n):
    new_words = words.copy()
    for _ in range(n):
        add_word(new_words)
    return new_words

def add_word(new_words):
    synonyms = []
    counter = 0
    while len(synonyms) < 1:
        random_word = new_words[random.randint(0, len(new_words)-1)]
        synonyms = get_synonyms(random_word)
        counter += 1
        if counter >= 10:
            return
    random_synonym = synonyms[0]
    random_idx = random.randint(0, len(new_words)-1)
    new_words.insert(random_idx, random_synonym)

def eda(sentence, alpha_sr=0.5, alpha_ri=0.2, alpha_rs=0.2, p_rd=0.2, num_aug=9):
    sentence = sentence.lower()
    words = sentence.split(' ')
    num_words = len(words)
    augmented_sentences = []
    num_new_per_technique = int(num_aug/4)+1
    n_sr = max(1, int(alpha_sr*num_words))
    n_ri = max(1, int(alpha_ri*num_words))
    n_rs = max(1, int(alpha_rs*num_words))

    # Synonym Replacement
    for _ in range(num_new_per_technique):
        a_words = synonym_replacement(words, n_sr)
        augmented_sentences.append(' '.join(a_words))

    # Random Insertion
    for _ in range(num_new_per_technique):
        a_words = random_insertion(words, n_ri)
        augmented_sentences.append(' '.join(a_words))

    # Random Swap
    for _ in range(num_new_per_technique):
        a_words = random_swap(words, n_rs)
        augmented_sentences.append(' '.join(a_words))

    # Random Deletion
    for _ in range(num_new_per_technique):
        a_words = random_deletion(words, p_rd)
        augmented_sentences.append(' '.join(a_words))

    augmented_sentences = [sentence] + augmented_sentences
    augmented_sentence = random.choice(augmented_sentences)
    return augmented_sentence

#----------------------------------------------------------Random sparsification------------------------------------------------------------------------
def fMRI_sparsification(data, rate1=0.2 , rate2=0.5):
    num_voxels = int(rate1 * data.size)
    voxel_indices = np.random.choice(data.size, num_voxels, replace=False)
    num_zeros = int(rate2 * num_voxels)
    zero_indices = np.random.choice(voxel_indices, num_zeros, replace=False)
    data.flat[zero_indices] = 0
    return data
# ----------------------------------------------------------Tool kits------------------------------------------------------------------
def _reshape(l):  # (8859,15,768)>>(8859,15*768)
    b = []
    for i in range(l.shape[0]):
        a = np.concatenate([l[i, j, :] for j in range(l.shape[1])], axis=0)
        b.append(a)
    b = np.array(b)
    return b

def decay(initial_weight, decay_rate, step):
    weight = initial_weight * np.exp(-decay_rate * step)
    return weight

def subsequent_mask(size, mask_ratio):
    #Random Causal mask
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    for i in range(size):
        if i >1:
            for j in range(size):
                if random.random() < 1 -mask_ratio and j!=0:
                    subsequent_mask[:,i,j] = 1

    return torch.from_numpy(subsequent_mask) == 0

def greedy_decode(first_frame,  model, src, src_mask,  max_len ,mask_ratio):
    decoder_input = first_frame
    src = torch.tensor(src)
    memory = model.encode(src, src_mask)                                        #src的数据类型为tensor
    ys = torch.ones(1, 1).fill_(1).type_as(src.data)

    for i in range(max_len-1):
        out = model.decode(memory, src_mask,
                           Variable(decoder_input),
                           Variable(subsequent_mask(ys.size(1), mask_ratio).type_as(src.data).to(device))   )
        next_frame = (out[:,-1,:]).unsqueeze(dim=1)
        ys = torch.cat([ys,torch.ones(1, 1).fill_(1).type_as(src.data)], dim=1)
        decoder_input = torch.cat([decoder_input,next_frame] , dim=1)
    return decoder_input

