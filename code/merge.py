#coding:utf-8
import os
#获取目标文件夹的路径  
meragefiledir = os.getcwd()+'/results1'  
#获取当前文件夹中的文件名称列表  
filenames=os.listdir(meragefiledir)  
filenames.sort()
#获取输出文件的路径 
file=open('result1.txt','w')  
#向文件中写入字符  
   
for filename in filenames:  
    print(filename)
    filepath=meragefiledir+'/'+filename    
    for line in open(filepath):  
        file.writelines(line[-7:]) 

file.close()  
