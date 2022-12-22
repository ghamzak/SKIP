import os
import glob
import subprocess
import pandas as pd

# def getLastFile(pathtofolder):
#     list_of_files = glob.glob(pathtofolder+'/*') # * means all if need specific format then *.csv
#     latest_file = max(list_of_files, key=os.path.getctime)
#     return latest_file


def format_fix(predictions_file_address):
    df = pd.read_csv(predictions_file_address, header=None, sep='\t', names=['pid', 'sid', 'entity', 'cos', 's1', 's2'])
    new_df_dict = df.to_dict()
    
    for i in range(len(new_df_dict['pid'])):
        if type(new_df_dict['s1'][i]) != str:
            new_df_dict['s1'][i]= '?'
        if new_df_dict['cos'][i] == 'MOVE' and new_df_dict['s1'][i] == '?' and type(new_df_dict['s2'][i]) != str:
            new_df_dict['s2'][i] = '??'
        else:
            if type(new_df_dict['s2'][i]) != str:
                new_df_dict['s2'][i]= '?'        
    new_df = pd.DataFrame(new_df_dict)
    x = predictions_file_address.split('/')
    new_predictions_file_address = '/'.join(x[:-1]) + x[-1].split('.')[0] + '_fixed.tsv'
    new_df.to_csv(new_predictions_file_address, index=False, header=False, sep='\t')
    


if __name__ == '__main__':
    # split = input('which split? ')
    parent_dir = os.path.dirname(os.path.realpath(__file__))#os.path.join(os.path.dirname(os.path.realpath(__file__)), 'aristo-leaderboard/propara')
    evaluator_file_address = os.path.join(parent_dir, 'evaluator/evaluator.py')
    predictions_file = os.path.join('/data/ghazaleh/neuralsymbolic/predict/', 'prediction.tsv')
    format_fix(predictions_file)
    x = predictions_file.split('/')
    fixed_prediction_file = '/'.join(x[:-1]) + x[-1].split('.')[0] + '_fixed.tsv' 
    answers_file = os.path.join('/home/ghazaleh/dissertation/symbolic/lexis2/aristo-leaderboard/propara/data/test/', 'answers.tsv')
    # subprocess.run(['python', evaluator_file_address, '-p', predictions_file, '-a', answers_file])
    subprocess.run(['python', evaluator_file_address, '-p', fixed_prediction_file, '-a', answers_file])