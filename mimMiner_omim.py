import process.data_process as data_process
import process.mesh_match as mesh_match

mesh_process = './data/mesh_process/'
omim_record = './data/omim_record/'
statisticbasepath = './data/statistic/'
omim_split_tag = './data/omim_split_tag/'
process = './data/process/'
base_path = "./data/"

def omimprocess():

    data_process.omim_split_tag_from_file(omim_split_tag)
    data_process.create_omimrecord_CS_TX(omim_split_tag, omim_record)
    data_process.records_process(omim_record,process)

def meshprocess():

    data_process.select_meshterm(mesh_process)

    data_process.mesh_process(mesh_process)

def calculateresult():

    mesh_match.actualCounter(mesh_process,process,statisticbasepath)

    mesh_match.hierarchy_counter(statisticbasepath)

    mesh_match.complish_weight(statisticbasepath)

    mesh_match.pre_cal_similarity(mesh_process,statisticbasepath)

    mesh_match.cal_similarity(statisticbasepath)

    mesh_match.similarity_sort(statisticbasepath)


if __name__ == "__main__":

    omimprocess()
    meshprocess()
    calculateresult()

