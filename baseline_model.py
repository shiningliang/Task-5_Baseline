import codecs
import csv
import numpy as np
import pandas as pd
import nltk
from nltk import pos_tag
import pycrfsuite
import re
from sklearn.model_selection import train_test_split
import numpy as np
import ast
import numpy as np
from sklearn.metrics import classification_report
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
def get_antecedent_begin_end(sentence_token, antetoken):
    '''

    :param line: sentence
    :param antetoken: antecedent of this sentence
    :return: begin：the begin index of antecedent end:the end index of antecedent
    '''
    flag = 0
    j = 0
    begin = end = 0
    for index, words in enumerate(sentence_token):
        if words == antetoken[0] and flag == 0:
            begin = index
            j = 1
        elif j >= len(antetoken) - 1:
            end = index
            j = len(antetoken) - 1
            break
        elif words == antetoken[j] and j != 0:
            flag = 1
            j += 1
        else:
            flag = 0
    if begin != 0 and end == 0:
        for index, words in enumerate(sentence_token):
            if '-' in words:
                subgroup = words.split('-', 1)
                sentence_token.insert(int(index), subgroup[0])
                sentence_token.insert(int(index) + 1, subgroup[1])
                sentence_token.remove(words)
        flag = 0
        j = 0
        begin = end = 0
        for index, words in enumerate(sentence_token):
            if words == antetoken[0] and flag == 0:
                begin = index
                j = 1
            elif j >= len(antetoken) - 1:
                end = index
                j = len(antetoken) - 1
                break
            elif words == antetoken[j] and j != 0:
                flag = 1
                j += 1
            else:
                flag = 0
    return begin, end
def get_consequence_begin_end(sentence_token, consetoken):
    '''

    :param sentence_token: sentence
    :param consetoken: the consequence of sentence( if exists)
    :return: begin:the begin index of consequence end: the end index of consequence
    '''
    flag = 0
    j = 0
    begin = end = 0
    for index, words in enumerate(sentence_token):
        if words == consetoken[0] and flag == 0:
            begin = index
            j = 1
        elif j >= len(consetoken) - 1:
            end = index
            j = len(consetoken) - 1
            break
        elif words == consetoken[j] and j != 0:
            flag = 1
            j += 1

        else:
            flag = 0
    if begin != 0 and end == 0:
        for index, words in enumerate(sentence_token):
            if '-' in words:
                subgroup = words.split('-', 1)
                sentence_token.insert(int(index), subgroup[0])
                sentence_token.insert(int(index) + 1, subgroup[1])
                sentence_token.remove(words)
        flag = 0
        j = 0
        begin = end = 0
        for index, words in enumerate(sentence_token):
            if words == consetoken[0] and flag == 0:
                begin = index
                j = 1
            elif j >= len(consetoken) - 1:
                end = index
                j = len(consetoken) - 1
                break
            elif words == consetoken[j] and j != 0:
                flag = 1
                j += 1
            else:
                flag = 0
    return begin, end
def word2features(doc, i):
    word = doc[i][0]
    postag = doc[i][1]

    features = [
        'bias',
        'word.lower=' + word.lower(),
        'word[-3:]=' + word[-3:],
        'word[-2:]=' + word[-2:],
        'word.isupper=%s' % word.isupper(),
        'word.istitle=%s' % word.istitle(),
        'word.isdigit=%s' % word.isdigit(),
        'postag=' + postag
    ]


    if i > 0:
        word1 = doc[i - 1][0]
        postag1 = doc[i - 1][1]
        features.extend([
            '-1:word.lower=' + word1.lower(),
            '-1:word.istitle=%s' % word1.istitle(),
            '-1:word.isupper=%s' % word1.isupper(),
            '-1:word.isdigit=%s' % word1.isdigit(),
            '-1:postag=' + postag1
        ])
    else:

        features.append('BOS')



    if i < len(doc) - 1:
        word1 = doc[i + 1][0]
        postag1 = doc[i + 1][1]
        features.extend([
            '+1:word.lower=' + word1.lower(),
            '+1:word.istitle=%s' % word1.istitle(),
            '+1:word.isupper=%s' % word1.isupper(),
            '+1:word.isdigit=%s' % word1.isdigit(),
            '+1:postag=' + postag1
        ])
    else:
        features.append('EOS')

    return features
def extract_features(doc):
    return [word2features(doc, i) for i in range(len(doc))]
def get_labels(doc):
    return [label for (token, postag, label) in doc]
def is_correct_predict(y_sentence):
    cnt_Bant=0
    cnt_Bcon=0
    for word in y_sentence:
        if word=='B-Con':
            cnt_Bcon+=1
        if word=='B-Ant':
            cnt_Bant+=1
        else:
            continue
    if cnt_Bant>1 or cnt_Bcon>1:
        return False
    else:
        return True
def get_coordinate(x_data, y_data):


    conse_begin = 0
    conse_begin_mask=1
    conse_end = 0
    conse_end_mask=1
    ante_begin = 0
    ante_begin_mask=1
    ante_end = 0
    ante_end_mask=1
    coordinate_sentence = []
    coordinate = []
    for index_sentence, y_sentence in enumerate(y_data):
        for index_word, y_word in enumerate(y_sentence[1]):
            if is_correct_predict(y_sentence[1])==False:
                ante_begin=-1
                ante_end=-1
                conse_begin=-1
                conse_end=-1
            else:
                if y_word == "O":
                    conse_begin += (len(x_data[index_sentence][1][index_word][1][11:]) + 1)*conse_begin_mask
                    conse_end += (len(x_data[index_sentence][1][index_word][1][11:]) + 1)*conse_end_mask
                    ante_begin += (len(x_data[index_sentence][1][index_word][1][11:]) + 1)*ante_begin_mask
                    ante_end += (len(x_data[index_sentence][1][index_word][1][11:]) + 1)*ante_end_mask
                    continue
                elif y_word == 'B-Con':
                    conse_begin_mask=0

                    conse_end += (len(x_data[index_sentence][1][index_word][1][11:]) + 1) * conse_end_mask
                    ante_begin += (len(x_data[index_sentence][1][index_word][1][11:]) + 1) * ante_begin_mask
                    ante_end += (len(x_data[index_sentence][1][index_word][1][11:]) + 1) * ante_end_mask
                    continue
                elif y_word == 'I-Con' and ((index_word<(len(y_sentence[1])-1)) and (y_sentence[1][index_word + 1] == 'I-Con')):
                    conse_begin += (len( x_data[index_sentence][1][index_word][1][11:]) + 1) * conse_begin_mask
                    conse_end += (len(x_data[index_sentence][1][index_word][1][11:]) + 1) * conse_end_mask
                    ante_begin += (len(x_data[index_sentence][1][index_word][1][11:]) + 1) * ante_begin_mask
                    ante_end += (len(x_data[index_sentence][1][index_word][1][11:]) + 1) * ante_end_mask
                    continue
                elif y_word == 'I-Con' and ((index_word == (len(y_sentence[1]) - 1)) or (y_sentence[1][index_word + 1] != 'I-Con')):
                    conse_end += (len(x_data[index_sentence][1][index_word][1][11:]))*conse_begin_mask
                    conse_begin += (len( x_data[index_sentence][1][index_word][1][11:]) + 1) * conse_begin_mask
                    ante_begin += (len(x_data[index_sentence][1][index_word][1][11:]) + 1) * ante_begin_mask
                    ante_end += (len(x_data[index_sentence][1][index_word][1][11:]) + 1) * ante_end_mask
                    conse_end_mask=0
                    continue
                elif y_word == 'B-Ant':
                    ante_begin_mask=0
                    conse_begin += (len( x_data[index_sentence][1][index_word][1][11:]) + 1) * conse_begin_mask
                    conse_end += (len(x_data[index_sentence][1][index_word][1][11:]) + 1) * conse_end_mask
                    ante_end += (len(x_data[index_sentence][1][index_word][1][11:]) + 1) * ante_end_mask
                    continue
                elif y_word == 'I-Ant' and ((index_word < (len(y_sentence[1]) - 1)) and (y_sentence[1][index_word + 1] == 'I-Ant')):
                    conse_begin += (len(x_data[index_sentence][1][index_word][1][11:]) + 1) * conse_begin_mask
                    conse_end += (len(x_data[index_sentence][1][index_word][1][11:]) + 1) * conse_end_mask
                    ante_begin += (len(x_data[index_sentence][1][index_word][1][11:]) + 1) * ante_begin_mask
                    ante_end += (len(x_data[index_sentence][1][index_word][1][11:]) + 1) * ante_end_mask
                    continue
                elif y_word == 'I-Ant' and ((index_word == (len(y_sentence[1]) - 1)) or (y_sentence[1][index_word + 1] != 'I-Ant')):
                    conse_begin += (len(x_data[index_sentence][1][index_word][1][11:]) + 1) * conse_begin_mask
                    conse_end += (len(x_data[index_sentence][1][index_word][1][11:])+1) * conse_end_mask
                    ante_begin += (len(x_data[index_sentence][1][index_word][1][11:]) + 1) * ante_begin_mask
                    ante_end += (len(x_data[index_sentence][1][index_word][1][11:])) * ante_end_mask
                    ante_end_mask=0
                    continue
            if conse_end == conse_begin:
                conse_end = conse_begin = -1

        coordinate_sentence = [y_sentence[0],ante_begin, ante_end, conse_begin, conse_end]
        coordinate.append(coordinate_sentence)

        conse_begin = 0
        conse_end = 0
        ante_begin = 0
        ante_end = 0
        conse_begin_mask=1
        conse_end_mask=1
        ante_begin_mask=1
        ante_end_mask=1
    return coordinate
def compare_single_result(coordinate_predict,coordinate_true):
    precision_all = []
    recall_all = []
    f1_score_all = []
    cmp_result = []
    conse_true_set = set()
    conse_predict_set = set()
    for index, coordinate_predict_sentence in enumerate(coordinate_predict):
        if (coordinate_predict_sentence[1]<0 or coordinate_predict_sentence[2]<0 or coordinate_predict_sentence[1]>coordinate_predict_sentence[2]):
            precision=0
            recall=0
            f1_score=0
        else:
            ante_predict_set = set(range(coordinate_predict_sentence[1], coordinate_predict_sentence[2] + 1))
            ante_true_set = set(range(coordinate_true[index][1], coordinate_true[index][2] + 1))

            if coordinate_predict_sentence[3]*coordinate_predict_sentence[4]<0 or coordinate_predict_sentence[3]>coordinate_predict_sentence[4]:
                precision=0
                recall=0
                f1_score=0
            else:
                if coordinate_predict_sentence[3] != -1 and coordinate_predict_sentence[4] != -1:  # 非空
                    conse_predict_set = set(range(coordinate_predict_sentence[3], coordinate_predict_sentence[4] + 1))
                if coordinate_true[index][3] != -1 and coordinate_true[index][4] != -1:
                    conse_true_set = set(range(coordinate_true[index][3], coordinate_true[index][4] + 1))

                ante_intersection_set = set.intersection(ante_true_set, ante_predict_set)
                conse_intersection_set = set.intersection(conse_true_set, conse_predict_set)
                if len(ante_predict_set) == 0:
                    precision = f1_score = recall = 0
                else:
                    precision = (len(ante_intersection_set) + len(conse_intersection_set)) / (
                        len(ante_predict_set) + len(conse_predict_set))
                    recall = (len(ante_intersection_set) + len(conse_intersection_set)) / (
                        len(ante_true_set) + len(conse_true_set))
                    if precision == recall == 0:
                        f1_score = 0
                    else:
                        f1_score = (2 * precision * recall) / (precision + recall)
        result = [coordinate_predict_sentence[0], precision, recall, f1_score]
        cmp_result.append(tuple(result))
        conse_true_set.clear()
        conse_predict_set.clear()
    for index, result_sentence in enumerate(cmp_result):
        precision=result_sentence[1]
        recall=result_sentence[2]
        f1_score=result_sentence[3]
        precision_all.append(precision)
        recall_all.append(recall)
        f1_score_all.append(f1_score)
    return precision_all, recall_all, f1_score_all
docs_train = []
docs_test=[]
filepath_train = r"train_v2.csv"
filepath_test=r"test_v2.csv"

with open(filepath_train, 'r', encoding="utf-8",errors="ignore") as readFile:  # file.close()
    reader = csv.reader(readFile)  # csv reader

    lines = list(reader) #two-dimension matrix 和excel表里一个样式
    for line in lines:
        if line[1]!='sentence':
            sentence_index=line[0]
            subdoc=[]
            senten=line[1]
            label=[]
            sentence_token=nltk.tokenize.word_tokenize(senten)
            antf=0
            contf=0
            ant=line[3]
            con=line[4]

            if ant!='N/A':
                antf=1
            if antf!=0:
                anttoken=nltk.tokenize.word_tokenize(ant)


            if con!='N/A':
                conf=1
            if conf!=0:
                contoken=nltk.tokenize.word_tokenize(con)


            if (antf==1):
                ante_begin_index,ante_end_index=get_antecedent_begin_end(sentence_token=sentence_token,antetoken=anttoken)
            if (conf==1):
                conse_begin_index,conse_end_index=get_consequence_begin_end(sentence_token=sentence_token,consetoken=contoken)
            for i in range(0, len(sentence_token)):
                label.append('O')
            if antf != 0:
                label[ante_begin_index] = 'B-Ant'
                for i in range(1, len(anttoken)):
                    label[ante_begin_index + i] = 'I-Ant'
            if conf != 0:
                label[conse_begin_index] = 'B-Con'
                for i in range(1, len(contoken)):
                    label[conse_begin_index + i] = 'I-Con'

            for i in range(0, len(sentence_token)):
                subdoc.append(tuple((sentence_token[i], label[i])))

            subdoc.insert(0,sentence_index)

            docs_train.append(subdoc)
readFile.close()

with open(filepath_test, 'r', encoding="utf-8",errors="ignore") as readFile:  # file.close()
    reader = csv.reader(readFile)

    lines = list(reader)
    for line in lines:
        if line[1]!='sentence':
            sentence_index=line[0]
            subdoc=[]
            senten=line[1]
            label=[]
            sentence_token=nltk.tokenize.word_tokenize(senten)
            antf=0
            contf=0
            ant=line[3]
            con=line[4]

            if ant!='N/A':
                antf=1
            if antf!=0:
                anttoken=nltk.tokenize.word_tokenize(ant)


            if con!='N/A':
                conf=1
            if conf!=0:
                contoken=nltk.tokenize.word_tokenize(con)


            if (antf==1):
                ante_begin_index,ante_end_index=get_antecedent_begin_end(sentence_token=sentence_token,antetoken=anttoken)
            if (conf==1):
                conse_begin_index,conse_end_index=get_consequence_begin_end(sentence_token=sentence_token,consetoken=contoken)
            for i in range(0, len(sentence_token)):
                label.append('O')
            if antf != 0:
                label[ante_begin_index] = 'B-Ant'
                for i in range(1, len(anttoken)):
                    label[ante_begin_index + i] = 'I-Ant'
            if conf != 0:
                label[conse_begin_index] = 'B-Con'
                for i in range(1, len(contoken)):
                    label[conse_begin_index + i] = 'I-Con'

            for i in range(0, len(sentence_token)):
                subdoc.append(tuple((sentence_token[i], label[i])))

            subdoc.insert(0,sentence_index)

            docs_test.append(subdoc)
readFile.close()
data_train = []
data_test=[]
for i, doc in enumerate(docs_train):
    tokens = [t for t, label in doc[1:]]
    tagged = nltk.pos_tag(tokens)


    subdata=[(w, pos, label) for (w, label), (word, pos) in zip(doc[1:], tagged)]
    subdata.insert(0,doc[0])

    data_train.append(subdata)
for i, doc in enumerate(docs_test):
    tokens = [t for t, label in doc[1:]]
    tagged = nltk.pos_tag(tokens)

    subdata=[(w, pos, label) for (w, label), (word, pos) in zip(doc[1:], tagged)]
    subdata.insert(0,doc[0])

    data_test.append(subdata)
with open('train_v2_after_proecessing.csv','wt',encoding='utf-8',newline='') as save_data:
    cw=csv.writer(save_data)
    for row in data_train:
        cw.writerow(row)
with open('test_v2_after_proecessing.csv','wt',encoding='utf-8',newline='') as save_data:
    cw=csv.writer(save_data)
    for row in data_test:
        cw.writerow(row)


train_data_reader=[]
train_data_reader_line=[]
with open('train_v2_after_proecessing.csv','r',encoding='utf-8',errors="ignore") as csvfile:
    csv_reader = csv.reader(csvfile)
    for row in csv_reader:
        train_data_reader_line=[ast.literal_eval(word)for word in row[1:]]
        train_data_reader_line.insert(0,row[0])
        train_data_reader.append(train_data_reader_line)
test_data_reader=[]
test_data_reader_line=[]
with open('test_v2_after_proecessing.csv','r',encoding='utf-8',errors="ignore") as csvfile:
    csv_reader = csv.reader(csvfile)
    for row in csv_reader:
        test_data_reader_line=[ast.literal_eval(word)for word in row[1:]]
        test_data_reader_line.insert(0,row[0])
        test_data_reader.append(test_data_reader_line)

train_sentence_indexs=[doc[0] for doc in train_data_reader]
test_sentence_indexs=[doc[0] for doc in test_data_reader]
X_train = [extract_features(doc[1:]) for doc in train_data_reader]
y_train = [get_labels(doc[1:]) for doc in train_data_reader]
X_test = [extract_features(doc[1:]) for doc in test_data_reader]
y_test = [get_labels(doc[1:]) for doc in test_data_reader]

X_train=[[train_sentence_indexs[index],x_sentence] for index, x_sentence in enumerate(X_train)]
y_train=[[train_sentence_indexs[index],y_sentence] for index, y_sentence in enumerate(y_train)]
X_test=[[test_sentence_indexs[index],x_sentence] for index, x_sentence in enumerate(X_test)]
y_test=[[test_sentence_indexs[index],y_sentence] for index, y_sentence in enumerate(y_test)]

trainer = pycrfsuite.Trainer(verbose=True)
for xseq, yseq in zip(X_train, y_train):
    trainer.append(xseq[1], yseq[1])
trainer.set_params({
    'c1':1.0,
    'c2':1.0,
    'max_iterations': 200,
    'feature.possible_transitions': True
})
trainer.train(r"crf.model")
tagger1=pycrfsuite.Tagger()
tagger1.open("crf.model")
y_pred=[]
for xseq in X_test:
    y_pred_sentence = tagger1.tag(xseq[1])
    y_pred.append([xseq[0], y_pred_sentence])
coordinate_pred = get_coordinate(X_test, y_pred)
coordinate_true = get_coordinate(X_test, y_test)

precision_all,recall_all,f1_score_all=compare_single_result(coordinate_pred,coordinate_true)
precision=[]
recall=[]
f1_score=[]
precision.extend(precision_all)
recall.extend(recall_all)
f1_score.extend(f1_score_all)
print("{}: precision :{:.3f}\t recall:{:.3f}\t f1_score:{:.3f}".format("average",np.mean(precision_all),np.mean(recall_all),np.mean(f1_score_all)))
print("{}: precision :{:.3f}\t recall:{:.3f}\t f1_score:{:.3f}".format("average_all",np.mean(precision),np.mean(recall),np.mean(f1_score)))