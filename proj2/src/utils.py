import re
import sys
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import math


class Corpus:
    def __init__(self):
        self.stemmed_data       = None
        self.vocabulary_mapping = None

    def create_vocabulary(self):
        self.vocabulary_mapping     = {}
        for d in self.stemmed_data:
            for key, value in d.items():
                for word in value:
                    if word in self.vocabulary_mapping:
                        postings_list               = self.vocabulary_mapping[word]
                    else:
                        postings_list               = PostingsList()
                        self.vocabulary_mapping[word]    = postings_list
                    
                    postings_list.insert_at_last(key)

class PostingNode:
    def __init__(self, data):
        self.data               = data
        self.next               = None
        self.prev               = None
        self.freq               = 1
        self.tf                 = None
        self.idf                = None
        self.tfidf              = None
        self.skipped_from       = None
        self.skipped_to         = None


class PostingsList:
    def __init__(self):
        self.head       = None
        self.tail       = None
        self.skip_ptr   = None
        self.length     = None

    def insert_at_beginning(self, data):
        new_node            = PostingNode(data)
        if self.head        == None:
            self.head       = new_node
            self.tail       = new_node
        else:
            new_node.next       = self.head
            new_node.next.prev  = new_node
            self.head           = new_node

            if self.tail        == None:
                running_node    = self.head
                while running_node.next:
                    running_node    = running_node.next
                self.tail       = running_node

    def insert_at_last(self, data):
        if self.tail:
            if self.tail.data == data:
                self.tail.freq += 1
                return

        new_node            = PostingNode(data)
        if self.head        == None:
            self.head       = new_node
            self.tail       = new_node
            return

        running_node        = self.head
        while running_node.next:
            running_node    = running_node.next

        running_node.next   = new_node
        new_node.prev       = running_node
        self.tail           = new_node

    def print_postings_list(self):
        running_node        = self.head
        print("[ ",end = "")
        while running_node:
            print({running_node.data:running_node.freq}, end = ", ")
            running_node    = running_node.next
        print("]")

    def obtain_postings_list(self):
        postings_list           = []
        running_node            = self.head
        while running_node:
            postings_list.append(running_node.data)
            running_node        = running_node.next
        
        return postings_list
    
    def obtain_postings_list_with_skip(self):
        postings_list           = []
        running_node            = self.head
        while running_node:
            if running_node.skipped_to:
                postings_list.append(running_node.data)
            elif running_node.skipped_from and running_node == self.tail:
                postings_list.append(running_node.data)

            running_node        = running_node.next
        
        return postings_list
    
    def get_postings_list_length(self):
        length = 0
        running_node            = self.head
        while running_node:
            length              += 1
            running_node        = running_node.next
        return length
    
    def get_node_at_skip_distance_from_current(self, current_node):
        counter = 0
        while counter < self.skip_ptr and current_node.next:
            current_node = current_node.next
            counter += 1
        return current_node
    
    def set_skip_pointers(self):
        self.length = self.get_postings_list_length()

        if self.length                  < 3:
            self.skip_ptr               = 0
            return

        elif self.length                <= 4:
            self.skip_ptr               = 2

        elif math.sqrt(self.length)     == int(math.sqrt(self.length)):
            self.skip_ptr               = math.floor(math.sqrt(self.length)) - 1
        
        else:
            self.skip_ptr               = math.floor(math.sqrt(self.length))

        current_node = self.head

        while self.get_node_at_skip_distance_from_current(current_node) != self.tail:
            running_node = self.get_node_at_skip_distance_from_current(current_node)
            current_node.skipped_to = running_node
            running_node.skipped_from = current_node
            current_node = running_node
        
        current_node.skipped_to = self.tail
        self.tail.skipped_from = current_node
        self.tail.skipped_to = None

    def print_skip_pointers(self):
        running_node                    = self.head
        while running_node.next:
            _from                       = running_node.skipped_from
            _to                         = running_node.skipped_to
            print({running_node.data: [_from.data if _from else None, _to.data if _to else None]}, end = ",")
            running_node = running_node.next
        _from                           = running_node.skipped_from
        _to                             = running_node.skipped_to
        print({running_node.data: [_from.data if _from else None, _to.data if _to else None]})

    def set_tf_idf(self, doc_token_count):
        running_node            = self.head
        num_docs                = len(doc_token_count)
        while running_node:
            running_node.tf     = running_node.freq / doc_token_count[running_node.data]
            running_node.idf    = num_docs / self.length if self.length else 0
            running_node.tfidf  = running_node.tf * running_node.idf
            running_node        = running_node.next


def clean_text(text):
    cleaned_text            = re.sub(r'-', '', text)
    cleaned_text            = re.sub(r'[^a-zA-Z0-9 ]', ' ', cleaned_text)
    cleaned_text            = re.sub(r'\s+', ' ', cleaned_text)
    cleaned_text            = cleaned_text.strip()
    return cleaned_text


def corpus_preprocessing():
    f                       = open("data/input_corpus.txt")
    # f                       = open("data/test_corpus.txt")
    data                    = []

    for line in f:
        data.append({int(line[:line.find("\t")]) : line[line.find("\t")+1:]})
    f.close()

    sorted_data             = sorted(data, key=lambda x: list(x.keys())[0])
    lowercased_data         = [{k: v.lower()} for d in sorted_data for k, v in d.items()]
    cleaned_data            = [{k: clean_text(v)} for d in lowercased_data for k, v in d.items()]
    tokenized_data          = [{k: v.split()} for d in cleaned_data for k, v in d.items()]

    stop_words              = set(stopwords.words('english'))
    stop_words_removed_data = [{k: [word for word in v if word.lower() not in stop_words]} for d in tokenized_data for k, v in d.items()]

    porter_stemmer          = PorterStemmer()
    stemmed_data            = [{k: [porter_stemmer.stem(word) for word in v]} for d in stop_words_removed_data for k, v in d.items()]

    return stemmed_data


def query_preprocessing(query : str):
    lowercased_query            = query.lower()
    cleaned_query               = clean_text(lowercased_query)
    tokenized_query             = cleaned_query.split()

    stop_words                  = set(stopwords.words('english'))
    stop_words_removed_query    = [word for word in tokenized_query if word not in stop_words]

    porter_stemmer              = PorterStemmer()
    stemmed_data                = [porter_stemmer.stem(word) for word in stop_words_removed_query]

    return set(stemmed_data)

def get_postings_list_without_skip(queryset : set, vocabulary_mapping):
    postingsList = {'postingsList': {}}
    for item in queryset:
        try:
            postingsList['postingsList'][item] = vocabulary_mapping[item].obtain_postings_list()
        except:
            postingsList['postingsList'][item] = []

    return postingsList['postingsList']

def get_postings_list_with_skip(queryset : set, vocabulary_mapping):
    postingsList = {'postingsListSkip': {}}
    for item in queryset:
        try:
            postingsList['postingsListSkip'][item] = vocabulary_mapping[item].obtain_postings_list_with_skip()
        except:
            postingsList['postingsListSkip'][item] = []

    return postingsList['postingsListSkip']


def set_tfidf(vocabulary_mapping, doc_token_count):
    for key, value in vocabulary_mapping.items():
        value.set_tf_idf(doc_token_count)


def daat_without_skip(vocabulary_mapping, split_query : list):
    try:
        postings_list1                  = vocabulary_mapping[split_query[0]]
    except:
        postings_list1                  = None

    try:
        postings_list2                  = vocabulary_mapping[split_query[1]]
    except:
        postings_list2                  = None
    
    common_postings_list            = []
    num_comparisons                 = 0
    ptr_1                           = postings_list1.head if postings_list1 else None
    ptr_2                           = postings_list2.head if postings_list2 else None
    
    while ptr_1 and ptr_2:
        if ptr_1.data == ptr_2.data:
            common_postings_list.append(ptr_1.data)
            ptr_1               = ptr_1.next
            ptr_2               = ptr_2.next
            num_comparisons     += 1   

        elif ptr_1.data > ptr_2.data:
            ptr_2               = ptr_2.next
            num_comparisons     += 1 

        else:
            ptr_1               = ptr_1.next
            num_comparisons     += 1
    
    return {
                'num_comparisons'   : num_comparisons,
                'num_docs'          : len(common_postings_list),
                'results'           : common_postings_list
            }



def daat_with_skip(vocabulary_mapping, split_query : list):
    try:
        postings_list1                  = vocabulary_mapping[split_query[0]]
    except:
        postings_list1                  = None

    try:
        postings_list2                  = vocabulary_mapping[split_query[1]]
    except:
        postings_list2                  = None
    
    common_postings_list            = []
    num_comparisons                 = 0
    ptr_1                           = postings_list1.head if postings_list1 else None
    ptr_2                           = postings_list2.head if postings_list2 else None
    
    while ptr_1 and ptr_2:
        if ptr_1.data == ptr_2.data:
            common_postings_list.append(ptr_1.data)
            ptr_1               = ptr_1.next
            ptr_2               = ptr_2.next
            num_comparisons     += 1   

        elif ptr_1.data < ptr_2.data:
            if ptr_1.skipped_to:
                if ptr_1.skipped_to.data < ptr_2.data:
                    ptr_1       = ptr_1.skipped_to
                else:
                    ptr_1       = ptr_1.next
            else:
                ptr_1       = ptr_1.next

            num_comparisons     += 1 

        else: 
            if ptr_2.skipped_to:
                if ptr_2.skipped_to.data < ptr_1.data:
                    ptr_2           = ptr_2.skipped_to
                else:
                    ptr_2           = ptr_2.next

            else:
                ptr_2           = ptr_2.next
            
            num_comparisons     += 1
    
    return {
                'num_comparisons'   : num_comparisons,
                'num_docs'          : len(common_postings_list),
                'results'           : common_postings_list
            }



