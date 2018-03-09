
import os
import subprocess
import glob
from shutil import move, copy

program_folder = './ThesisPrograms/'
os.makedirs(program_folder, exist_ok=True)

notebook_folder = program_folder + './notebooks/'
os.makedirs(notebook_folder, exist_ok=True)

python_folder = program_folder +'./python/'
os.makedirs(python_folder, exist_ok=True)

#pdf_folder = program_folder +'./pdf/'
#os.makedirs(pdf_folder, exist_ok=True)

notebook_list = glob.glob('*.ipynb')
for filename in notebook_list:
    subprocess.check_call(['jupyter', 'nbconvert', '--to', 'script', filename])
    #subprocess.call(['jupyter', 'nbconvert', '--to', 'pdf', filename])

for filename in notebook_list:
    try:
        copy(filename, notebook_folder)
        print('copied:', filename)
        filename = filename.split('.')[0]+'.py'
        move(filename, os.path.join(python_folder, filename))
        print('moved:', filename)
        #filename = filename.split('.')[0]+'.pdf'
        #move(filename, pdf_folder)
        #print('moved:', filename)
        print('******************')
    except Exception as err:
        print('**Excepton:',err)

#Copy this file also
this_file = os.path.basename(__file__)
copy(this_file, program_folder)
print('copied this file:', this_file)      

os.chdir(program_folder)
subprocess.check_call('git add .', shell=True)
subprocess.check_call('git commit -m "New updates"', shell=True)
subprocess.check_call('git push -u origin master', shell=True)

