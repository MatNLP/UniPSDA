import torch
import datetime
# from mbert_related_file.modeling_bert import BertModel
from transformers import BertModel, BertTokenizer, BertForMaskedLM, AdamW
import sys
sys.path.append("..")

def init_method():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bert = BertModel.from_pretrained(
        'bert-base-multilingual-cased', output_hidden_states=True,output_attentions=True)
    tokenizer = BertTokenizer.from_pretrained(
        'bert-base-multilingual-cased')
    return device, bert, tokenizer


def important_words_hidden( bert, tokenizer):
    write_path = '../senNumKey_posAndWordAndRootValue_sen_cla_combine.pt'
    senNumKey_posAndWordValue = torch.load(write_path)

    words_dict = senNumKey_posAndWordValue
    senNumKey_wordsHiddenListValue = {}
    for senNum in words_dict.keys():
        if senNum % 200 == 0:
            time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print("senNum: %d || %s " % (senNum, time))
        pos_words_dict = words_dict[senNum]
        words_hidden_list = []
        for i, word_str in enumerate(pos_words_dict['word']):
            para_of_bert = tokenizer(word_str, return_tensors='pt')
            input_ids = para_of_bert.input_ids
            # print('=======', input_ids.shape)
            input_ids = input_ids.tolist()[0]
            # print(input_ids)
            input_ids.pop()
            attention_mask = [1] * len(input_ids)
            if len(input_ids) > 10:
                input_ids = input_ids[:10]
                attention_mask = [1] * len(input_ids)
            while len(input_ids) <= 8:
                input_ids.append(1)
                attention_mask.append(0)
            input_ids.append(102)
            attention_mask.append(0)
            input_ids = torch.tensor(input_ids).unsqueeze(0)
            attention_mask = torch.tensor(attention_mask).unsqueeze(0)
            token_type_ids = torch.zeros_like(input_ids)
            # try:
            hidden_states_tuple = bert(input_ids=input_ids,
                          token_type_ids=token_type_ids,
                          attention_mask=attention_mask).hidden_states
            first_hidden = hidden_states_tuple[1]
            last_hidden = hidden_states_tuple[12]
            avg_first_last_hidden = (first_hidden + last_hidden) / 2
            avg_first_last_hidden = avg_first_last_hidden.detach()
            avg_first_last_hidden = avg_first_last_hidden.sum(1)/10

            words_hidden_list.append(avg_first_last_hidden)
        senNumKey_wordsHiddenListValue[senNum] = words_hidden_list
    print('len(senNumKey_wordsHiddenListValue):', len(senNumKey_wordsHiddenListValue))

    write_path = '../senNumKey_wordsHiddenListValue_sen_cla.pt'
    torch.save(senNumKey_wordsHiddenListValue, write_path)
    print('write done')
    return senNumKey_wordsHiddenListValue

def combine_senNumKey_dict_from_negAndpos():
   

    total_dict = {}
    var_train_set_list = ['strneg', 'neg', 'pos', 'strpos']
    index = 0
    for var_train_set in var_train_set_list:
        path = '../senNumKey_posAndWordAndRootValue_sen_cla_%s.pt'%var_train_set
        doc_data = torch.load(path)
        for key in doc_data.keys():
            total_dict[index] = doc_data[key]
            index += 1
    print(len(total_dict))
    write_path = '../senNumKey_posAndWordAndRootValue_sen_cla_combine.pt'
    torch.save(total_dict,write_path)

def important_word_type_index():
    write_path = '../senNumKey_posAndWordAndRootValue_sen_cla_combine.pt'
    senNumKey_posAndWordValue = torch.load(write_path)

def important_word_in_en_vocab_index():
    f = "../multilingual_vocab_dict.txt"
    train_file = open(f, 'r')
    line_list = train_file.readlines()
    en_vocab_list = eval(line_list[0])['en']
    write_path = '../senNumKey_posAndWordAndRootValue_sen_cla_combine.pt'
    senNumKey_posAndWordValue = torch.load(write_path)
    for senNum in senNumKey_posAndWordValue.keys():
        if senNum % 1000 == 0:
            time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print("senNum: %d || %s " % (senNum, time))
        sen_vocab_index = []
        for word_str in senNumKey_posAndWordValue[senNum]['word']:
            lower_word_str = word_str.lower()
            if lower_word_str not in en_vocab_list:
                vocab_index = 134880
            else:
                vocab_index = en_vocab_list.index(lower_word_str)
            sen_vocab_index.append(vocab_index)
        senNumKey_posAndWordValue[senNum]['vocab_index'] = sen_vocab_index
    path = '../senNumKey_posAndWordAndRootAndVocabIndexValue_sen_cla.pt'
    torch.save(senNumKey_posAndWordValue, path)

def important_word_in_cluster30000_typeIndex():
    path = '../senNumKey_posAndWordAndRootAndVocabIndexValue_sen_cla.pt'
    senNumKey_posAndWordAndRootAndVocabIndexValue = torch.load(path)
    lang = 'en'
    path1 = '../pt_for_weighted_cluster/type_30000_tensor_%s.pt' % lang
    type_30000_tensor_lang = torch.load(path1)
    path2 = '../pt_for_weighted_cluster/type_30000_index_of_big_dict_%s.pt' % lang
    type_30000_index_of_big_dict_lang = torch.load(path2)
    type_30000_tensor_lang_dict = {}
    for iter, tensor_value in enumerate(type_30000_tensor_lang):
        type_30000_tensor_lang_dict[iter] = tensor_value
    type_30000_index_of_big_dict_lang_dict = {}
    for iter, index_list in enumerate(type_30000_index_of_big_dict_lang):
        type_30000_index_of_big_dict_lang_dict[iter] = index_list

    en_index2type_30000 = {}
    for key in type_30000_index_of_big_dict_lang_dict.keys():
        for index in type_30000_index_of_big_dict_lang_dict[key]:
            en_index2type_30000[index] = key

    
    for senNum in senNumKey_posAndWordAndRootAndVocabIndexValue.keys():
        if senNum % 1000 == 0:
            time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print("senNum: %d || %s " % (senNum, time))
        cluster_30000_type_index_list = []
        for index in senNumKey_posAndWordAndRootAndVocabIndexValue[senNum]['vocab_index']:
            if index in en_index2type_30000.keys():
                cluster_30000_type_index = en_index2type_30000[index]
                cluster_30000_type_index_list.append(cluster_30000_type_index)
            else:
                cluster_30000_type_index = 30000
                cluster_30000_type_index_list.append(cluster_30000_type_index)
        senNumKey_posAndWordAndRootAndVocabIndexValue[senNum]['30000_type_index'] = cluster_30000_type_index_list
    path = '../senNumKey_posAndWordAndRootAndVocabIndexAnd30000typeIndexValue_sen_cla.pt'
    torch.save(senNumKey_posAndWordAndRootAndVocabIndexValue, path)
    print()

def find_same_cluster_in_de_en():
    lang = 'en'
    path2 = '../pt_for_weighted_cluster/type_30000_index_of_big_dict_%s.pt' % lang
    type_30000_index_of_big_dict_de_en = torch.load(path2)
    path = '../senNumKey_posAndWordAndRootAndVocabIndexAnd30000typeIndexValue_sen_cla.pt'
    senNumKey_posAndWordAndRootAndVocabIndexAnd30000typeIndexValue = torch.load(path)
    de_en_index2type_30000 = {}
    for iter, type_inner_ele_index in enumerate(type_30000_index_of_big_dict_de_en):
        for index in type_inner_ele_index:
            de_en_index2type_30000[index] = iter

    senNumKey_30000typeIndexValue = {}
    for senNum in senNumKey_posAndWordAndRootAndVocabIndexAnd30000typeIndexValue.keys():
        if senNum % 1000 == 0:
            time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print("senNum: %d || %s " % (senNum, time))
        cluster_30000_type_index_de_en_list = []
        for index in senNumKey_posAndWordAndRootAndVocabIndexAnd30000typeIndexValue[senNum]['30000_type_index']:
            if index in de_en_index2type_30000.keys():
                if index == 30000:
                    type_index = 60000
                else:
                    type_index = de_en_index2type_30000[index]
                cluster_30000_type_index_de_en_list.append(type_index)
        senNumKey_30000typeIndexValue[senNum] = cluster_30000_type_index_de_en_list
    path = '../senNumKey_cluster_30000_type_index_de_en_sen_cla.pt'
    torch.save(senNumKey_30000typeIndexValue, path)
    print()

def find_same_cluster_in_multi_lang():
    path2 = '../pt_for_weighted_cluster/type_30000_index_of_big_dict_multi_lang_real.pt'
    type_30000_index_of_big_dict_multi_lang = torch.load(path2)
    path = '../senNumKey_cluster_30000_type_index_de_en_sen_cla.pt'
    senNumKey_cluster_30000_type_index_de_en = torch.load(path)
    multi_lang_index2type_30000 = {}
    for iter, type_inner_ele_index in enumerate(type_30000_index_of_big_dict_multi_lang):
        for index in type_inner_ele_index:
            multi_lang_index2type_30000[index] = iter
    senNumKey_30000typeIndexValue_multi_lang = {}
    for senNum in senNumKey_cluster_30000_type_index_de_en.keys():
        if senNum % 1000 == 0:
            time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print("senNum: %d || %s " % (senNum, time))
        cluster_30000_type_index_multi_lang_list = []
        for index in senNumKey_cluster_30000_type_index_de_en[senNum]:
            if index in multi_lang_index2type_30000.keys():
                if index == 60000:
                    type_index = 140000
                else:
                    type_index = multi_lang_index2type_30000[index]
                cluster_30000_type_index_multi_lang_list.append(type_index)
        senNumKey_30000typeIndexValue_multi_lang[senNum] = cluster_30000_type_index_multi_lang_list
    path = '../senNumKey_30000typeIndexValue_multi_lang_sen_cla.pt'
    torch.save(senNumKey_30000typeIndexValue_multi_lang, path)
    print()

def find_important_word_candidate_tensor():
    path1 = '../pt_for_weighted_cluster/tensor_30000_collect_from_multi_lang_real.pt'
    tensor_30000_collect_from_multi_lang_real = torch.load(path1)
    path = '../senNumKey_30000typeIndexValue_multi_lang_sen_cla.pt'
    senNumKey_30000typeIndexValue_multi_lang = torch.load(path)
    senNumKey_candidateTensorValue = {}
    for senNum in senNumKey_30000typeIndexValue_multi_lang.keys():
        sen_candidateTensor = []
        for index in senNumKey_30000typeIndexValue_multi_lang[senNum]:
            if index == 140000:
                candidate_tensor = torch.zeros(1,768)
            else:
                candidate_tensor = tensor_30000_collect_from_multi_lang_real[index]
            sen_candidateTensor.append(candidate_tensor)
        senNumKey_candidateTensorValue[senNum] = sen_candidateTensor
    path = '../senNumKey_candidateTensorValue_sen_cla.pt'
    torch.save(senNumKey_candidateTensorValue, path)
if __name__ == '__main__':
    combine_senNumKey_dict_from_negAndpos()
    device, bert, tokenizer = init_method()
    important_words_hidden( bert, tokenizer)
    important_word_in_en_vocab_index()
    important_word_in_cluster30000_typeIndex()
    find_same_cluster_in_de_en()
    find_same_cluster_in_multi_lang()
    find_important_word_candidate_tensor()