
# coding: utf-8

# In[1]:

import glob
#Merge sample files to create bigger sameple
def merge_sample_files(sample_path, merge_count):
    #Make copy
    head, tail = os.path.split(sample_path)
    copy_path = os.path.join(head,tail+'_copy')
    if os.path.isdir(copy_path):
        shutil.rmtree(copy_path)
    shutil.copytree(sample_path, copy_path)
    #delete all the file and recreate folder
    shutil.rmtree(sample_path)
    os.makedirs(sample_path, exist_ok=True)
    file_number = 1
    count = 0
    filenames = sorted(glob.glob(os.path.join(copy_path,'*')),  key=os.path.getmtime)
    for filename in filenames:
        if count == 0:
            df = pd.read_csv(filename, index_col=0)
            count += 1
        else:
            temp_df = pd.read_csv(filename, index_col=0)
            df = df.append(temp_df)
            count += 1
        if count == merge_count:
            df.to_csv(os.path.join(sample_path,str(file_number)))
            df = df.drop(df.index, inplace=True)
            count = 0
            file_number +=1


# In[ ]:



