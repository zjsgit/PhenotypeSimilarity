from collections import Counter
from collections import OrderedDict
import math
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

'''
读取Mesh并返回用于记录频率的字典和{id：name}，条目名为空（data_process中删除）的不考虑在内
'''
def readMesh(basePath):
    with open(basePath + 'mesh.txt','r') as mfile:
        meshlist = mfile.readlines()
        meshID_count = OrderedDict()
        meshIDNum_count = OrderedDict()
        meshID_name = OrderedDict()
        meshIDNum_name = OrderedDict()
        for mesh in meshlist:
            meshRecord = mesh.strip('\n').split('\t')
            if len(meshRecord[1]) > 0:
                meshID = meshRecord[0]
                meshID_count[meshID] = 0
                meshID_name[meshID] = meshRecord[1]
                meshIDNum_count[meshRecord[0].split()[1]] = 0
                meshIDNum_name[meshRecord[0].split()[1]] = meshRecord[1]
        return  meshID_count, meshID_name, meshIDNum_count,meshIDNum_name

'''
获取omim记录的ID
'''
def readomimID(basepath):
    omimPath = './data/record_5080.txt'
    with open(omimPath, 'r') as rfile:
        omimID = [line.strip('\n') for line in rfile.readlines()]
        return omimID

'''
根据omimID获取相应的Record的词频记录
'''
def readomimRecord(line,basepath):
    readpath = basepath + line + '.txt'
    with open(readpath,'r') as rfile:
        record = rfile.read()
    return record

'''
判断mesh term是否出现在Record中，若出现则计数+1,并记录mesh term在该记录中的次数
'''
def counter(meshID_count,meshID_name,omim_record):
    for meshID in meshID_count.keys():
        meshName = meshID_name[meshID]
        recordNum = omim_record.count(meshName)
        if recordNum > 0:
            meshID_count[meshID] += 1

'''
循环统计mesh在Records
'''
def statistic(meshbasepath,omimbasepath):
        meshID_count, meshID_name, meshIDNum_count, meshIDNum_name = readMesh(meshbasepath)
        omimID = readomimID(omimbasepath)
        for omimid in omimID:
            omim_record = readomimRecord(omimid,omimbasepath)
            counter(meshID_count,meshID_name,omim_record)
        return meshID_count,meshID_name,meshIDNum_count, meshIDNum_name




'''
获取mesh中term在每一个record的actual_count，并保存
'''
def actualCounter(meshbasepath,process,statisticbasepath):
    print('--------------计算actual_count------------------')
    with open(meshbasepath + 'mesh_promulti.txt', 'r') as rfile:
        meshlist = rfile.readlines()
        omimID = readomimID(process)
        treeID_Name = OrderedDict()
        with open(statisticbasepath + 'actual_count.txt', 'w') as wfile:
            wfile.write('ID')
            for mesh in meshlist:
                meshterm = mesh.strip('\n').split('\t')
                treeID_Name[meshterm[0]] = meshterm[2:]
                wfile.write('\t')
                wfile.write(meshterm[0]+'|'+meshterm[1])
            wfile.write('\n')
            for omimid in omimID:
                record = readomimRecord(omimid,process)
                wfile.write(omimid)
                for key in treeID_Name.keys():
                    meshName = treeID_Name[key]
                    minCount = float('inf')
                    for name in meshName:
                        num = record.count(name)
                        if num < minCount:
                            minCount = num
                    wfile.write('\t')
                    wfile.write(str(minCount))
                wfile.write('\n')


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

'''
计算hierarchy count
'''
def hierarchy_counter(statisticbasepath):

    print('--------------计算hiera_count------------------')
    with open(statisticbasepath + 'actual_count.txt', 'r') as rfile:
        with open(statisticbasepath +'hiera_count.txt', 'w') as wfile:
            actual_count = rfile.readlines()
            wfile.write(actual_count[0])
            meshID = actual_count[0].strip('\n').split('\t')[1:]
            treeID = [line.split('|')[0] for line in meshID]
            actual_record = actual_count[1:]
            treeid_count = OrderedDict()
            isCalculate = OrderedDict()
            for record in actual_record:
                record_count = record.strip('\n').split('\t')[1:]
                for i in range(len(treeID)):
                    treeid_count[treeID[i]] = record_count[i]
                    isCalculate[treeID[i]] = False
                for treeid in treeID:
                    if isCalculate[treeid] == False:
                        calculate(treeid,treeID,treeid_count,isCalculate)
                recordid = record.split('\t')[0]
                wfile.write(recordid)
                wfile.write('\t')
                for key in treeid_count.keys():
                    if isCalculate[key] == True:
                        wfile.write(str(treeid_count[key]))
                        wfile.write('\t')
                    else:
                        wfile.write('False')
                        wfile.write('\t')
                wfile.write('\n')

'''
计算gw_c和r_mf
'''
def global_local(statisticbasepath):
    with open(statisticbasepath + 'actual_count.txt', 'r') as rfile:
        rlist = rfile.readlines()
        meshID = rlist[0].strip('\n').split('\t')[1:]
        treeID = [line.split('|')[0] for line in meshID]
        # print('treeID-------',len(treeID))
        gwc_global = OrderedDict()
        for treeid in treeID:
            gwc_global[treeid] = 0
        mostCount = []
        for record in rlist[1:]:
            most_occur = 0
            recordlist = record.strip('\n').split('\t')[1:]
            for i in range(len(treeID)):
                meshNum = float(recordlist[i])
                if meshNum > most_occur:
                    most_occur = meshNum
                if meshNum > 0:
                    gwc_global[treeID[i]] += 1
            mostCount.append(most_occur)
    for key in gwc_global.keys():
        recordNum = gwc_global[key]
        if recordNum > 0:
            gwc_global[key] = math.log2(5080.0/recordNum)
        else:
            gwc_global[key] = 0.0

    most_zero = []
    for i in range(len(mostCount)):
        if mostCount[i] == 0:
            mostCount[i] = 1
            most_zero.append(i)
    print("共有",len(most_zero),"个记录未匹配到任何Mesh term")
    print(most_zero)
    with open('./data/statistic/glow_weight.txt', 'w') as wfile:
        for w in gwc_global:
            wfile.write(str(w))
            wfile.write('\n')
    return gwc_global, mostCount,most_zero

'''
实现公式二和公式三
'''
def getCount():
    with open('./data/record_5080.txt', 'r') as rfile:
        rlist = rfile.readlines()
        rlist = [line.strip('\n') for line in rlist]
        recordlen = OrderedDict()
        for record in rlist:
            with open('./data/process/' + record + '.txt', 'r') as sfile:
                lens = len(sfile.read())
                recordlen[record] = lens
    return recordlen

def complish_weight(statisticbasepath):

    print('---------------计算权重----------------')
    gwc_global, mostCount, most_zero = global_local(statisticbasepath)
    #recordsCount = getCount()
    # print('mostCount:',mostCount)
    gwc = []
    for key in gwc_global.keys():
        gwc.append(gwc_global[key])
    with open(statisticbasepath +'hiera_count.txt', 'r') as rfile:
        hlist = rfile.readlines()
        records = hlist[1:]
        with open(statisticbasepath + 'weight_count.txt', 'w') as wfile:
            wfile.write(hlist[0])
            for i in range(len(records)):
                mf = mostCount[i]
                meshlist = records[i].split('\t')
                wfile.write(meshlist[0])
                cal_list = meshlist[1:-1]
                cal_list = [float(num) for num in cal_list]
                gwc_cal = np.array(gwc) * np.array(cal_list)
                gwc_list = gwc_cal.tolist()
                #if recordsCount[meshlist[0]] > 1500:
                cal_result = []
                for score in gwc_list:
                    if score > 0:
                        cal_result.append(0.5 + 0.5 * (score/mf))
                    else:
                        cal_result.append(score)
                gwc_list = cal_result
                for resul in gwc_list:
                    wfile.write('\t')
                    wfile.write(str(resul))
                wfile.write('\n')

'''
统计actual_count的信息
'''
def selectNumid(statisticbasepath):
    with open(statisticbasepath + 'actual_count.txt', 'r') as rfile:
        flist = rfile.readlines()
        meshidlist = flist[0].strip('\n').split('\t')[1:]
        treeidlist = [line.split('|')[0] for line in meshidlist]
        records_count = flist[1:]
        treeid_numid = OrderedDict()
        treeid_count = OrderedDict()
        for meshid in meshidlist:
            treeid = meshid.split('|')[0]
            numid = meshid.split('|')[1]
            treeid_numid[treeid] = numid
            treeid_count[treeid] = 0
        for record in records_count:
            recordlist = record.strip('\n').split('\t')[1:]
            for i in range(len(recordlist)):
                if float(recordlist[i]) > 0 :
                    treeid = treeidlist[i]
                    treeid_count[treeid] += 1
        with open(statisticbasepath + 'numid_selected.txt','w') as wfile:
            numidlist = []
            for key in treeid_count.keys():
                if treeid_count[key] > 0:
                    numid = treeid_numid[key]
                    if numid not in numidlist:
                        wfile.write(key)
                        wfile.write('\t')
                        wfile.write(numid)
                        wfile.write('\n')
                        numidlist.append(numid)

'''
选取每个记录中在records中出现的mesh term记录并保存
'''
def pre_cal_similarity(mesh_process,statisticbasepath):

    print('-------------选择参与计算的mesh term----------------------')
    selectNumid(statisticbasepath)
    with open(statisticbasepath + 'numid_selected.txt', 'r') as rfile:
        rlist = rfile.readlines()
        tree_num = OrderedDict()
        for tr_num in rlist:
            term = tr_num.strip('\n').split('\t')
            tree_num[term[0]] = term[1]
        keylist = tree_num.keys()
        with open(statisticbasepath + 'weight_count.txt', 'r') as calfile:
            with open(statisticbasepath + 'pre_calsimilarity.txt', 'w') as wfile:
                wfile.write('numid')
                for key in tree_num.keys():
                    wfile.write('\t')
                    wfile.write(tree_num[key])
                wfile.write('\n')
                records = calfile.readlines()
                idlist = records[0].strip('\n').split('\t')[1:]
                treeidlist = [line.split('|')[0] for line in idlist]
                print(len(treeidlist),treeidlist)
                records = records[1:]
                for record in records:
                    recordlist = record.strip('\n').split('\t')
                    wfile.write(recordlist[0])
                    numlist = recordlist[1:]
                    for i in range(len(numlist)):
                        if treeidlist[i] in keylist:
                            wfile.write('\t')
                            wfile.write(numlist[i])
                    wfile.write('\n')

'''
计算余弦相似性
'''
def cal_similarity(statisticbasepath):

    print('-----------------计算余弦相似性---------------------')
    with open(statisticbasepath +'pre_calsimilarity.txt', 'r') as rfile:
        with open(statisticbasepath + 'similarity_result.txt', 'w') as wfile:
            records= rfile.readlines()[1:]
            omimID = []
            allrecord = []
            for record_a in records:
                recordlist_a = record_a.strip('\n').split('\t')
                omimID_a = recordlist_a[0]
                omimID.append(omimID_a)
                recordlist_a = recordlist_a[1:]
                allrecord.append(recordlist_a)

            allarray = np.array(allrecord)
            similarity_resul = cosine_similarity(allarray)
            similarity_list = similarity_resul.tolist()
            wfile.write('omimID')
            for omimid in omimID:
                wfile.write('\t')
                wfile.write(omimid)
            wfile.write('\n')
            for i in range(len(omimID)):
                wfile.write(omimID[i])
                omimlist = similarity_list[i]
                for omim_score in omimlist:
                    wfile.write('\t')
                    wfile.write(str(omim_score))
                wfile.write('\n')

def similarity_sort(statisticbasepath):

    with open(statisticbasepath + 'similarity_result.txt', 'r') as rfile:
        dsim = {}
        rlist = rfile.readlines()
        omimID = rlist[0].split('\t')[1:-1]
        records_list = rlist[1:]
        for i in range(len(records_list)):
            record = records_list[i]
            recordlist = record.split('\t')[0:-1]
            omimid_a = recordlist[0]
            recordlist = recordlist[1:]
            recordscore = recordlist[i+1:]
            recordlist_b = omimID[i+1:]
            for j in range(len(recordlist_b)):
                ida_idb = omimid_a + '\t' + recordlist_b[j]
                dsim[ida_idb] = recordscore[j]
        dsim_sorted = sorted(dsim.items(), key=lambda id: float(id[1]), reverse=True)
        with open(statisticbasepath + 'similarity_sorted.txt', 'w') as wfile:
            for tup in dsim_sorted:
                wfile.write(tup[0] + '\t' + tup[1])
                wfile.write('\n')

if __name__ == '__main__':

    pass
