from nltk import word_tokenize
from collections import Counter
from nltk.corpus import stopwords

import numpy as np
import os
import xml.etree.ElementTree as ET
import html
from html.parser import HTMLParser
import re

stop = set(stopwords.words('english'))

# Lấy thông tin những từ từ matrix GloVe
def load_embedding_file(embed_file_name, word_set):
    ''' loads embedding file and returns a dictionary (word -> embedding) for the words existing in the word_set '''

    embeddings = {}
    with open(embed_file_name, 'r') as embed_file:
        for line in embed_file:
            content = line.strip().split()
            word = content[0]
            if word in word_set:
                embedding = np.array(content[1:], dtype=float)
                embeddings[word] = embedding

    # Thông tin matrix bao gồm các cặp giá trị Key = word, Value = row của matrix chứa từ đó
    return embeddings


def get_dataset_resources(data_file_name, sent_word2idx, target_word2idx, word_set, max_sent_len):
    ''' updates word2idx and word_set '''
    if len(sent_word2idx) == 0:
        sent_word2idx["<pad>"] = 0

    word_count = []
    sent_word_count = []
    target_count = []

    words = []
    sentence_words = []
    target_words = []

    # Mở file
    with open(data_file_name, 'r') as data_file:
        lines = data_file.read().split('\n')

        # Chỉ đọc những dòng chứa những review
        for line_no in range(0, len(lines) - 1, 3):
            sentence = lines[line_no]
            target = lines[line_no + 1]

            sentence.replace("$T$", "")
            sentence = sentence.lower()
            target = target.lower()
            max_sent_len = max(max_sent_len, len(sentence.split()))
            sentence_words.extend(sentence.split())
            target_words.extend([target])
            words.extend(sentence.split() + target.split())

        # Đếm số lần xuất hiện những từ trong câu,
        # những từ target,
        # và tất cả những từ trong câu bao gồm cả target
        sent_word_count.extend(Counter(sentence_words).most_common())
        target_count.extend(Counter(target_words).most_common())
        word_count.extend(Counter(words).most_common())

        # Tạo bộ từ điển các từ trong câu với Key = word, Value = index++
        for word, _ in sent_word_count:
            if word not in sent_word2idx:
                sent_word2idx[word] = len(sent_word2idx)

        # Tạo bộ từ điển các từ target với Key = word, Value = index++
        for target, _ in target_count:
            if target not in target_word2idx:
                target_word2idx[target] = len(target_word2idx)

        # Tạo bộ từ điển các từ trong câu gao gồm cả từ target với Key = word, Value = index++
        for word, _ in word_count:
            if word not in word_set:
                word_set[word] = 1

    # Trả lại độ dài của câu dài nhất
    return max_sent_len


def get_embedding_matrix(embeddings, sent_word2idx, target_word2idx, edim):
    ''' returns the word and target embedding matrix '''
    word_embed_matrix = np.zeros([len(sent_word2idx), edim], dtype=float)
    target_embed_matrix = np.zeros([len(target_word2idx), edim], dtype=float)

    for word in sent_word2idx:
        if word in embeddings:
            word_embed_matrix[sent_word2idx[word]] = embeddings[word]

    for target in target_word2idx:
        for word in target:
            if word in embeddings:
                target_embed_matrix[target_word2idx[target]] += embeddings[word]
        target_embed_matrix[target_word2idx[target]] /= max(1, len(target.split()))

    print(type(word_embed_matrix))
    return word_embed_matrix, target_embed_matrix


def get_dataset(data_file_name, sent_word2idx, target_word2idx, embeddings):
    ''' returns the dataset'''
    sentence_list = []
    location_list = []
    target_list = []
    polarity_list = []

    with open(data_file_name, 'r') as data_file:
        lines = data_file.read().split('\n')
        for line_no in range(0, len(lines) - 1, 3):
            sentence = lines[line_no].lower()
            target = lines[line_no + 1].lower()
            polarity = int(lines[line_no + 2])

            sent_words = sentence.split()
            target_words = target.split()
            try:
                # Lấy vị trí của từ target trong câu
                target_location = sent_words.index("$t$")
            except:
                print("sentence does not contain target element tag")
                exit()

            is_included_flag = 1
            id_tokenised_sentence = []
            location_tokenised_sentence = []

            for index, word in enumerate(sent_words):
                if word == "$t$":
                    continue
                try:
                    # vị trí của từ trong bộ từ điển vừa tạo
                    word_index = sent_word2idx[word]
                except:
                    print("id not found for word in the sentence")
                    exit()

                # lấy khoảng cách từ trong câu đối với từ target
                location_info = abs(index - target_location)

                # Lưu thông tin khoảng cách từ trong câu với từ target
                # Đồng thời cũng lưu vị trí từ đó trong từ điển vừa tạo
                if word in embeddings:
                    id_tokenised_sentence.append(word_index)
                    location_tokenised_sentence.append(location_info)

                # if word not in embeddings:
                #   is_included_flag = 0
                #   break

            # Nếu từ target có trong từ điển thì đánh dấu cờ flag = 1
            is_included_flag = 0
            for word in target_words:
                if word in embeddings:
                    is_included_flag = 1
                    break

            try:
                # Lấy vị trí của từ target trong từ điển vừa tạo
                target_index = target_word2idx[target]
            except:
                print(target)
                print("id not found for target")
                exit()

            # Nếu từ target đó không có trong từ điển thì bỏ qua
            if not is_included_flag:
                print(sentence)
                continue

            # Lưu thông tin các đánh giá review dưới dạng index trong từ điển
            # Lưu thông tin các khoảng các của từ trong câu đối với từ target
            # Lưu thông tin các từ target dưới dạng index trong từ điển
            # Lưu hướng đánh giá của người dừng đối với đánh giá, polarity in (0, 1, 2)
            sentence_list.append(id_tokenised_sentence)
            location_list.append(location_tokenised_sentence)
            target_list.append(target_index)
            polarity_list.append(polarity)

    return sentence_list, location_list, target_list, polarity_list
