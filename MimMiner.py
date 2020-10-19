# coding: utf-8

#-------------------------------------------------------------------------------
# Name:         Test
# Description:  
# Author:       JiashuaiZhang
# Date:         2020/5/19
#-------------------------------------------------------------------------------

import random
from collections import defaultdict
from collections import OrderedDict
import sys
import re
import os
import nltk
import numpy as np
import math
from sklearn.metrics.pairwise import cosine_similarity

import FileUtil

def sterm(record):
    del_list = ['of', 'the', 'and', 'in', 'at', 'to', 'by', '.', ',', '(', ')', 's', "'", ';',':','/','[',']','-','%']
    recordlist = []
    for word in record:
        if word not in del_list:
            recordlist.append(word)
    recordlist = [nltk.PorterStemmer().stem(word) for word in recordlist]
    return recordlist

def mesh_process(mesh_file):
    '''
    从mesh.xml文件中，获取tree结构和AC部分的ID和疾病名称
    :param mesh_file:
    :return:
    '''

    print('--------------选取Mesh中A和C的部分-------------------')
    selected_count = 0
    with open(mesh_file, 'r', encoding='utf-16') as mfile:
        flist = mfile.readlines()
        flist = [line.strip('\n') for line in flist]
        flist = [line.split('\t')[0:3] for line in flist]

    tokenizer = nltk.RegexpTokenizer(r"\w+(?:[-]\w+)*|'|[-.(]+|\S\w*")
    del_list = ['syndrom', 'disease', 'of', 'the', 'and', 'in', 'at', 'to', 'by', '.', ',', '(', ')', 's', "'", ';', ':']

    tree_id2mesh_id = OrderedDict()
    tree_id2synonym = OrderedDict()

    for line in flist:
        if line[0].startswith('A') | line[0].startswith('C'):
            selected_count += 1
            meshName = line[2:]
            terms = []
            for name in meshName:
                record = tokenizer.tokenize(name)
                record = sterm(record)
                record = [word.lower() for word in record]
                terms.append(record)

            tree_id2mesh_id[line[0]] = line[1]
            synonym = ""
            if len(terms) == 1 & (len(terms[0]) == 1) & (terms[0][0] in del_list):
                synonym  += terms[0][0]
                synonym  += ' '
            else:
                for term in terms:
                    for word in term:
                        if word not in del_list:
                            synonym  += word
                            synonym  += ' '
            tree_id2synonym[line[0]] = synonym.strip()

    print('选取了', selected_count, '个mesh term')

    mesh_info_file = "./data/mesh_process/after_mesh.txt"
    with open(mesh_info_file, 'w') as wfile:
        for tree_id in tree_id2mesh_id.keys():
            wfile.write(tree_id + "\t" + tree_id2mesh_id[tree_id] + "\t" + tree_id2synonym[tree_id] + "\n")
    print("mesh information写入了{}".format(mesh_info_file))

    return tree_id2mesh_id, tree_id2synonym

def process_omim(omim_file, omim_ids_file):
    '''
    从omim.txt文件中，获取omim对应的描述
    :param omim_file:
    :return:
    '''
    omim_record_dir = "./data/omim_record/"
    record_count = 0

    records = FileUtil.read_file2list(omim_file)
    file_path = ""
    omimID = ""
    for i in range(len(records)):
        line = records[i]
        if "*RECORD*" in line:
            omimID = records[i + 2]
            sys.stdout.write("\romim id -> {}".format(omimID))
            record_count += 1
        else:
            if "*FIELD*" in line:
                tagName = line.split()[1]
                if (os.path.exists(omim_record_dir + omimID + '/') == False):
                    os.mkdir(omim_record_dir + omimID + '/')
                file_path = omim_record_dir + omimID + '/' + omimID + '_' + tagName + '.txt'
            else:
                with open(file_path, 'a') as newfile:
                    newfile.write(line)
                    newfile.write('\n')

    print("\nomim文件中共有{}个".format(record_count))

    print('--------------合并CS TX---------------')
    omim_ids = FileUtil.read_file2list(omim_ids_file)
    process_dir = "./data/omim_process/"
    tokenizer = nltk.RegexpTokenizer(r"\w+(?:[-]\w+)*|'|[-.(.[./]+|\S\w*")
    del_list = ['syndrom', 'diseas', 'of', 'the', 'and', 'in', 'at', 'to', 'by', '.', ',', '(', ')', 's', "'", ';',
                ':', '[', ']', '/']

    omim2description = OrderedDict()
    for omim_id in omim_ids:
        cs_path = omim_record_dir + omim_id + '/' + omim_id + '_CS.txt'
        cs_content = ""
        if os.path.exists(cs_path):
            cs_content = FileUtil.read_file2str(cs_path)

        tx_content = ""
        tx_path = omim_record_dir + omim_id + '/' + omim_id + '_TX.txt'
        if os.path.exists(tx_path):
            tx_content = FileUtil.read_file2str(tx_path)

        cs_tx_content = cs_content + tx_content.strip()
        if cs_tx_content != "":
            filtered_text = tokenizer.tokenize(cs_tx_content)
            filtered_text = sterm(filtered_text)
            filtered_text = [word.lower() for word in filtered_text if word not in del_list]

            description = " ".join(filtered_text)
            omim2description[omim_id] = description

            write_path = process_dir + omim_id + '.txt'
            with open(write_path, 'w') as wfile:
                wfile.write(description)

            sys.stdout.write("\r{}的TX和CS文本写入到了{}".format(omim_id, write_path))

    print()
    return omim2description

'''
递归计算
'''
def calculate(treeid,treeID,mesh_count,isCalculate):
    treeidlist = treeid.split('.')
    treeidlen = len(treeidlist)
    sta_treeid = []
    childNodes = []
    actual_count = float(mesh_count[treeid])
    childContri = 0.0

    for stid in treeID:
        if stid.startswith(treeid) :
            sta_treeid.append(stid)

    for staid in sta_treeid:
        stalist = staid.split('.')
        if len(stalist) == (treeidlen+1):
            childNodes.append(staid)
    if len(childNodes) > 0:
        for childNode in childNodes:
            childContri += calculate(childNode,sta_treeid,mesh_count,isCalculate)
        result = actual_count + childContri/len(childNodes)
    else:
        result = actual_count

    mesh_count[treeid] = result
    isCalculate[treeid] = True

    return result

def calculate_similarity(omim2description, tree_id2synonym):
    '''
    根据表型的描述和同义词在mesh tree中的位置，计算表型之间的相似性
    :param omim2description:
    :param tree_id2synonym:
    :return:
    '''

    omim_ids = list(omim2description.keys())
    tree_ids = list(tree_id2synonym.keys())

    # ----------------------计算actual count-------------------------------
    actual_count = np.zeros((len(omim_ids), len(tree_ids)))
    for index_omim, omim_id in enumerate(omim_ids):
        sys.stdout.write("\ractual_count->{}".format(omim_id))
        description = omim2description[omim_id]
        for index_tree, tree_id in enumerate(tree_ids):
            name = tree_id2synonym[tree_id]
            actual_count[index_omim][index_tree] = description.count(name)

    print()
    np.savetxt("./data/statistic/actual_count.txt", actual_count, delimiter='\t', fmt="%d")

    # ----------------------计算hiera_count----------------------------------
    hiera_count = actual_count
    for index_omim, omim_id in enumerate(omim_ids):
        is_calculate = OrderedDict()
        tree_id_count = OrderedDict()
        sys.stdout.write("\rhiera_count->{}".format(omim_id))
        for index_tree, tree_id in enumerate(tree_ids):
            tree_id_count[tree_id] = hiera_count[index_omim][index_tree]
            is_calculate[tree_id] = False

        for tree_id in tree_ids:
            if is_calculate[tree_id] == False:
                calculate(tree_id, tree_ids, tree_id_count, is_calculate)

        for index_tree, tree_id in enumerate(tree_id_count.keys()):
            if is_calculate[tree_id] == True:
                hiera_count[index_omim][index_tree] = tree_id_count[tree_id]

    print()
    np.savetxt("./data/statistic/hiera_count.txt", hiera_count, delimiter='\t', fmt="%f")

    # ------------------------------计算weight_count-------------------------------------
    gwc_global = OrderedDict()
    for tree_id in tree_ids:
        gwc_global[tree_id] = 0

    mostCount = []
    for index_omim, omim_id in enumerate(omim_ids):
        most_occur = 0
        for index_tree, tree_id in enumerate(tree_ids):
            meshNum = float(actual_count[index_omim][index_tree])
            if meshNum > most_occur:
                most_occur = meshNum
            if meshNum > 0:
                gwc_global[tree_id] += 1

        mostCount.append(most_occur)

    for key in gwc_global.keys():
        recordNum = gwc_global[key]
        if recordNum > 0:
            gwc_global[key] = math.log2(len(omim_ids)/recordNum)
        else:
            gwc_global[key] = 0.0

    gwc = list(gwc_global.values())
    weight_count = np.zeros((len(omim_ids), len(tree_ids)))
    for index_omim, omim_id in enumerate(omim_ids):
        sys.stdout.write("\rweight_count->{}".format(omim_id))
        mf = mostCount[index_omim]
        cal_list = []
        for index_tree, tree_id in enumerate(tree_ids):
            cal_list.append(float(hiera_count[index_omim][index_tree]))
        gwc_cal = np.array(gwc) * np.array(cal_list)
        gwc_list = gwc_cal.tolist()
        cal_result = []
        for score in gwc_list:
            if score > 0:
                cal_result.append(0.5 + 0.5 * (score / mf))
            else:
                cal_result.append(score)
        for index_tree, tree_id in enumerate(tree_ids):
            weight_count[index_omim][index_tree] = cal_result[index_tree]

    print()
    np.savetxt("./data/statistic/weight_count.txt", weight_count, delimiter='\t', fmt="%f")

    # --------------------------计算disease similarity--------------------------------------
    print("---------------------calculate disease similarity---------------------------------")
    similarity_result = cosine_similarity(weight_count)
    dsim = {}
    for i in range(len(omim_ids)):
        for j in range(i+1, len(omim_ids)):
            dsim["{}\t{}".format(omim_ids[i], omim_ids[j])] = similarity_result[i][j]
    dsim_sorted = sorted(dsim.items(), key=lambda x: float(x[1]), reverse=True)

    similarity_result_file = "./data/statistic/result.txt"
    with open(similarity_result_file, "w") as file:
        for line in dsim_sorted:
            file.write("{}\t{:%f}\n".format(line[0], line[1]))


if __name__ == '__main__':

    omim_text_file = "./data/omim.txt"
    omim_ids_file = "./data/record_5080.txt"
    omim2description = process_omim(omim_text_file, omim_ids_file)

    mesh_file = "./data/mesh_process/MeshTreeHierarchy.csv"
    tree_id2mesh_id, tree_id2synonym = mesh_process(mesh_file)

    calculate_similarity(omim2description, tree_id2synonym)




    pass
