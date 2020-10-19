import os
import nltk
import re
from collections import Counter
from time import *


delelist = []


'''
读取omim.txt，将每个Record按照*Field*进行拆分，并保存至名为omim_id的文件夹下不同的文件中
'''
def omim_split_tag_from_file(omimsplittag):
    print('---------------拆分原始数据---------------')
    omimID = list()
    filePath = ''
    recordCount = 0
    with open('./data/omim.txt', 'r') as file:
        filelist = file.readlines()
        filelist = [line.strip('\n') for line in filelist]
        for i in range(len(filelist)):
            line = filelist[i]
            if "*RECORD*" in line:
                omimID = filelist[i + 2]
                recordCount += 1
            else:
                if "*FIELD*" in line:
                    tagName = line.split()[1]
                    if (os.path.exists(omimsplittag + omimID + '/') == False):
                        os.mkdir(omimsplittag + omimID + '/')
                    filePath = omimsplittag + omimID + '/' + omimID + '_' + tagName + '.txt'
                else:
                    with open(filePath, 'a') as newfile:
                        newfile.write(line)
                        newfile.write('\n')
    print('共提取',recordCount,'个record')


'''
从omim中选取遗传疾病相关记录的omim_id并保存
根据id选择相关记录病组织成mimminer_record并保存

# 1.将cs整体作为一个句子
# 2.对tx每个句子作为一行
'''
def write(rlist,path,model):
    with open(path,model) as mfile:
        for line in rlist:
            mfile.writelines(line)
            mfile.write('\n')


def read(path,model):
    if os.path.exists(path) == False:
        return []
    else:
        with open(path,model) as mfile:
            llist = mfile.readlines()
            llist = [line.strip('\n') for line in llist]
            return llist


def create_omimrecord_CS_TX(readPath,writePath):
    print('--------------合并CS TX---------------')
    rlist = read('./data/record_5080.txt','r')
    rlist = [line.strip('\n') for line in rlist]
    recordCount = len(rlist)
    recordlist = []
    for line in rlist:
        csPath = readPath + line + '/' + line+ '_CS.txt'
        txPath = readPath + line + '/' + line+ '_TX.txt'
        recordPath = writePath + line+ '.txt'
        cslist = read(csPath,'r')
        txlist = read(txPath,'r')
        cslist.extend(txlist)
        if len(cslist) < 3:
            recordlist.append(line)
            print(cslist)
        write(cslist,recordPath,'w')
    write(recordlist,writePath + 'nullRecord.txt','w')
    print('共选出',recordCount,'个record')
    print('共有',len(recordlist),'个为空')
    print(recordlist)


'''
1,提取词干
2,删除指定字符
'''
def sterm(record):
    del_list = ['of', 'the', 'and', 'in', 'at', 'to', 'by', '.', ',', '(', ')', 's', "'", ';',':','/','[',']','-','%']
    recordlist = []
    for word in record:
        if word not in del_list:
            recordlist.append(word)
    recordlist = [nltk.PorterStemmer().stem(word) for word in recordlist]
    return recordlist

'''
一次读取一个记录所有字符
读分词，提取词干，处理成小写，保存（每个单词后加‘ ’）
'''
def process_record(readpath,writepath):
    tokenizer = nltk.RegexpTokenizer(r"\w+(?:[-]\w+)*|'|[-.(.[./]+|\S\w*")
    del_list = ['syndrom','diseas','of', 'the', 'and', 'in', 'at', 'to', 'by', '.', ',', '(', ')', 's', "'", ';', ':','[',']','/']
    with open(readpath, 'r') as file:
        recordlist = file.read()
        recordlist = tokenizer.tokenize(recordlist)
        recordlist = sterm(recordlist)
        recordlist = [word.lower() for word in recordlist if word not in del_list]
    with open(writepath,'w') as wfile:
        for word in recordlist:
            wfile.write(word)
            wfile.write(' ')

'''
循环处理所有记录
'''
def records_process(readPath,writePath):
    print('---------对所有records：分词，提取词干，处理大小写-------------')
    with open('./data/record_5080.txt','r') as rfile:
        for line in rfile.readlines():
            line = line.strip('\n')
            readpath = readPath + line + '.txt'
            writepath = writePath + line + '.txt'
            process_record(readpath,writepath)





'''
提取Mesh中A和C的术语
'''
def select_meshterm(meshbasepath):

    print('--------------选取Mesh中A和C的部分-------------------')
    selected_count = 0
    with open(meshbasepath + 'MeshTreeHierarchy.csv', 'r', encoding='utf-16') as mfile:
        flist = mfile.readlines()
        flist = [line.strip('\n') for line in flist]
        flist = [line.split('\t')[0:3] for line in flist]
        with open(meshbasepath + 'AC_mesh.txt', 'w') as wfile:
            for line in flist:
                if line[0].startswith('A') | line[0].startswith('C'):
                    selected_count += 1
                    wfile.write(line[0] + '\t')
                    wfile.write(line[1] + '\t')
                    wfile.write(line[2])
                    wfile.write('\n')
    print('选取了',selected_count,'个mesh term')


'''
对多个词组的mesh term
'''
def pro_multi(meshbasepath):

    with open(meshbasepath + 'AC_mesh.txt', 'r') as rfile:
        rlist = rfile.readlines()
        rlist = [line.strip('\n') for line in rlist]
        with open(meshbasepath + 'AC_mesh_promulti.txt','w') as wfile:
            for line in rlist:
                terms = line.split('\t')
                wlist = terms[2].split(',')
                wfile.write(terms[0])
                wfile.write('\t')
                wfile.write(terms[1])
                for w in wlist:
                    wfile.write('\t')
                    wfile.write(w)
                wfile.write('\n')






'''
预处理mesh
读取A,C部分的mesh，对术语词部分进行分词，提取词干，改为小写。
保留整个record在del_list中的Mesh term，删除一个mesh term中某个出现在record中的词
层级关系和id间以‘ ’隔开，前两部分与name以'\t'隔开
mesh_name部分：每个word之间以' '隔开,删除出现在del_list中的词
'''
def mesh_process(meshbasepath):

    print('---------对所有Mesh term：分词，提取词干，处理大小写-------------')
    tokenizer = nltk.RegexpTokenizer(r"\w+(?:[-]\w+)*|'|[-.(]+|\S\w*")
    del_list = ['syndrom','disease','of', 'the', 'and', 'in', 'at', 'to', 'by', '.', ',', '(', ')', 's', "'", ';', ':']
    with open(meshbasepath + 'AC_mesh.txt','r') as rfile:
        with open(meshbasepath + 'mesh_promulti.txt', 'w') as wfile:
            for line in rfile.readlines():
                recordlist = line.strip('\n').split('\t')
                meshName = recordlist[2:]
                terms = []
                for name in meshName:
                    record = tokenizer.tokenize(name)
                    record = sterm(record)
                    record = [word.lower() for word in record]
                    terms.append(record)
                str = recordlist[0] + '\t' + recordlist[1] + '\t'
                if len(terms)== 1 & (len(terms[0]) == 1) & (terms[0][0] in del_list):
                    str += terms[0][0]
                    str += ' '
                else:
                    for term in terms:
                        for word in term:
                            if word not in del_list:
                                str += word
                                str += ' '
                        str += '\t'
                str.strip()
                wfile.write(str)
                wfile.write('\n')


if __name__ == '__main__':

    pass