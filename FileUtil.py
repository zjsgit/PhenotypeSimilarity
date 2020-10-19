# coding: utf-8

#-------------------------------------------------------------------------------
# Name:         FileUtil
# Description:  
# Author:       JiashuaiZhang
# Date:         2020/10/18
#-------------------------------------------------------------------------------

import os
from collections import defaultdict

def read_file2list(file_path):

    with open(file_path, "r") as file:
        lines = file.readlines()
        file_content = [line.strip('\n') for line in lines]
        return file_content

def write_list2file(write_content, file_path):

    with open(file_path, "w", encoding="utf-8") as file:
        for line in write_content:
            file.write(line + "\n")

def read_file2str(file_path):

    with open(file_path, "r") as file:
        lines = file.readlines()
        content_str = ""
        for line in lines:
            if line != " " and line != "\n":
                content_str += line.strip("\n") + " "
    return content_str

if __name__ == '__main__':
    pass