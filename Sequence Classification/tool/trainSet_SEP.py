import torch
import random
import datetime
import en_core_web_sm

def trainFile2trianSenList(path):
    
    train_file = open(path, 'r')
    line_list = train_file.readlines()
    train_sen_list = []
    for str in line_list:
        sen = str.split('\t')[1]
        train_sen_list.append(sen)
    print('len(train_sen_list):',len(train_sen_list))
    return train_sen_list

def main():
   
    write_path = '../sen_after_replace_with_SEP_list.pt'
    sen_with_some_SEP = torch.load(write_path)

    
    train_path = '../dataset/MLDoc/english.train.10000'
    with open(train_path,'r') as f:
        b = f.readlines()
    lableList = []
    for i, line in enumerate(b):
        line = line.split("\t")
        lable = line[0] + '\t'
        lableList.append(lable)

    sen_with_lable_str = ''
    for i, sen in enumerate(sen_with_some_SEP):
        a = sen.__contains__('\n')
        if a == False:
            sen = sen + '\n'
        sen_with_lable = lableList[i] + sen
        sen_with_lable_str = sen_with_lable_str + sen_with_lable
    write_path = '../sen_with_some_SEP.train'
    with open(write_path, 'w', encoding='utf-8') as f:
        f.write(sen_with_lable_str)

    train_path = '../sen_with_some_SEP.train'
    with open(train_path, 'r') as f:
        sep_file = f.readlines()
    print('done')

def sen_data_with_SEP():
    f = "../dataset/MIXSC/en/opener_sents/train/strneg.txt"
    train_file = open(f, 'r')
    line_list = train_file.readlines()
    train_sen_list = []
    for str in line_list:
        # sen = str.split('\t')[1]
        sen = str
        train_sen_list.append(sen)
    print('len(train_sen_list):', len(train_sen_list))

if __name__ == '__main__':
    main()