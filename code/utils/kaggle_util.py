import pandas as pd
import numpy as np

# Any results you write to the current directory are saved as output.
from time import time
from tqdm import tqdm_notebook as tqdm
from IPython.display import HTML

from collections import Counter
from scipy import stats
import lightgbm as lgb
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import KFold, StratifiedKFold
import gc
import json
pd.set_option('display.max_columns', 1000)

def read_data():
    print('Reading train.csv file....')
    train = pd.read_csv('../data/raw_data/train.csv')
    print('Training.csv file have {} rows and {} columns'.format(train.shape[0], train.shape[1]))

    print('Reading test.csv file....')
    test = pd.read_csv('../data/raw_data/test.csv')
    print('Test.csv file have {} rows and {} columns'.format(test.shape[0], test.shape[1]))

    print('Reading train_labels.csv file....')
    train_labels = pd.read_csv('../data/raw_data/train_labels.csv')
    print('Train_labels.csv file have {} rows and {} columns'.format(train_labels.shape[0], train_labels.shape[1]))

    print('Reading specs.csv file....')
    specs = pd.read_csv('../data/raw_data/specs.csv')
    print('Specs.csv file have {} rows and {} columns'.format(specs.shape[0], specs.shape[1]))

    print('Reading sample_submission.csv file....')
    sample_submission = pd.read_csv('../data/raw_data/sample_submission.csv')
    print('Sample_submission.csv file have {} rows and {} columns'.format(sample_submission.shape[0], sample_submission.shape[1]))
    return train, test, train_labels, specs, sample_submission

def encode_title(train, test, train_labels):
    # encode title
    # map function:在對一陣列的元素做運算時使用，
    
    train['title_event_code'] = list(map(lambda x, y: str(x) + '_' + str(y), train['title'], train['event_code']))
    test['title_event_code'] = list(map(lambda x, y: str(x) + '_' + str(y), test['title'], test['event_code']))
    all_title_event_code = list(set(train["title_event_code"].unique()).union(test["title_event_code"].unique()))
    # make a list with all the unique 'titles' from the train and test set
    list_of_user_activities = list(set(train['title'].unique()).union(set(test['title'].unique())))
    # make a list with all the unique 'event_code' from the train and test set
    list_of_event_code = list(set(train['event_code'].unique()).union(set(test['event_code'].unique())))
    list_of_event_id = list(set(train['event_id'].unique()).union(set(test['event_id'].unique())))
    # make a list with all the unique worlds from the train and test set
    list_of_worlds = list(set(train['world'].unique()).union(set(test['world'].unique())))
    # create a dictionary numerating the titles
    activities_map = dict(zip(list_of_user_activities, np.arange(len(list_of_user_activities))))
    activities_labels = dict(zip(np.arange(len(list_of_user_activities)), list_of_user_activities))
    activities_world = dict(zip(list_of_worlds, np.arange(len(list_of_worlds))))
    activities_world = {'NONE':0, 'MAGMAPEAK':1, 'TREETOPCITY':2, 'CRYSTALCAVES':3}
    
    assess_titles = list(set(train[train['type'] == 'Assessment']['title'].value_counts().index).union(set(test[test['type'] == 'Assessment']['title'].value_counts().index)))
    # replace the text titles with the number titles from the dict
    train['title'] = train['title'].map(activities_map)
    test['title'] = test['title'].map(activities_map)
    train['world'] = train['world'].map(activities_world)
    test['world'] = test['world'].map(activities_world)
    train_labels['title'] = train_labels['title'].map(activities_map)
    win_code = dict(zip(activities_map.values(), (4100*np.ones(len(activities_map))).astype('int')))
    # then, it set one element, the 'Bird Measurer (Assessment)' as 4110, 10 more than the rest
    win_code[activities_map['Bird Measurer (Assessment)']] = 4110
    
    # convert text into datetime
    train['timestamp'] = pd.to_datetime(train['timestamp'])
    test['timestamp'] = pd.to_datetime(test['timestamp'])
    
    train['timestamp_weekday'] = pd.DatetimeIndex(train.timestamp).weekday
    train['timestamp_daytime'] = pd.DatetimeIndex(train.timestamp).hour*60 + pd.DatetimeIndex(train.timestamp).minute
    test['timestamp_weekday'] = pd.DatetimeIndex(test.timestamp).weekday
    test['timestamp_daytime'] = pd.DatetimeIndex(test.timestamp).hour*60 + pd.DatetimeIndex(test.timestamp).minute
    
    return train, test, train_labels, win_code, list_of_user_activities, list_of_event_code, activities_labels, assess_titles, list_of_event_id, all_title_event_code

def get_data(user_sample, win_code, list_of_user_activities, list_of_event_code, activities_labels, assess_titles, list_of_event_id, all_title_event_code, test_set=False):
    ''' 
    The user_sample is a DataFrame from train or test where the only one 
    installation_id is filtered
    And the test_set parameter is related with the labels processing, that is only requered
    if test_set=False
    '''
    # Constants and parameters declaration            
    all_assessments = []
    accumulated_accuracy_group = 0
    accumulated_accuracy = -1
    accumulated_correct_attempts = 0 
    accumulated_incorrect_attempts = 0
    accumulated_actions = 0
    counter = 0
    assessment_durations = []
    accuracy = -1
    
    """
    User acc history
    """
    ## acmu acc each title
    acmu_correct_attempt = {'acmu_correct_attempt_' + title: 0 for title in assess_titles}
    acmu_incorrect_attempt = {'acmu_incorrect_attempt_' + title: 0 for title in assess_titles}
    acmu_accuracy_attempt = {'acmu_accuracy_attempt_' + title: -1 for title in assess_titles}
    
    ## acmu accgp total
    accuracy_groups = {'acmu_acc_gp_0':0, 'acmu_acc_gp_1':0, 'acmu_acc_gp_2':0, 'acmu_acc_gp_3':0}
    
    ## last acc each title
    last_accuracy_title = {'last_acc_' + title: -1 for title in assess_titles}
 

    # new features: time spent in each activity
    last_session_time_sec = 0
    
    """
    learning history
    """
    type_count = {'acmu_type_Clip':0, 'acmu_type_Activity': 0, 'acmu_type_Assessment': 0, 'acmu_type_Game':0}
    world_count = {'acmu_world_NONE':0, 'acmu_world_MAGMAPEAK': 0, 'acmu_world_TREETOPCITY': 0, 'acmu_world_CRYSTALCAVES':0}
    world_map = {0:'NONE', 1:'MAGMAPEAK', 2:'TREETOPCITY', 3:'CRYSTALCAVES'}
    
    event_code_count = {'acmu_ev_code_'+str(eve): 0 for eve in list_of_event_code}
    event_id_count = {'acmu_ev_id_'+str(eve): 0 for eve in list_of_event_id}
    title_count = {'acmu_title_'+str(eve): 0 for eve in activities_labels.values()} 
    title_event_code_count = {'acmu_ev_title_'+str(eve): 0 for eve in all_title_event_code}
#     event_code_count: Dict[str, int] = {'acmu_ev_code_'+str(eve): 0 for eve in list_of_event_code}
#     event_id_count: Dict[str, int] = {'acmu_ev_id_'+str(eve): 0 for eve in list_of_event_id}
#     title_count: Dict[str, int] = {'acmu_title_'+str(eve): 0 for eve in activities_labels.values()} 
#     title_event_code_count: Dict[str, int] = {'acmu_ev_title_'+str(eve): 0 for eve in all_title_event_code}
    

        
    ## timestamp
    duration_title = {'duration_title_' + title: [] for title in activities_labels.values()}
    duration_title2 = {'duration_title_' + title: 0 for title in activities_labels.values()}
    
    ## 種類統計
    title_class_count = set()
    type_class_count = set()
    world_class_count = set()
    event_id_class_count = set()
    event_code_class_count = set()
    
    ##
    session_total_times = 0
    
    # itarates through each session of one instalation_id
    for i, session in user_sample.groupby('game_session', sort=False):
        # i = game_session_id
        # session is a DataFrame that contain only one game_session
        
        # get some sessions information
        session_type = session['type'].iloc[0]
        session_title = session['title'].iloc[0]
        session_title_text = activities_labels[session_title]
        session_world = session['world'].iloc[0]
        session_weekday = session['timestamp_weekday'].iloc[0]
        session_daytime = session['timestamp_daytime'].iloc[0]
        session_timestamp = session['timestamp'].iloc[0]

        ## modified timestamp
        ## minus session time when diff time between events is more than 10 min
        x = session.timestamp
        x = (x - x.shift(1)).astype('timedelta64[s]')
        x = np.sum(x[x>10*60])
        session_time = (session.iloc[-1, 2] - session.iloc[0, 2] ).seconds - x
        
        session_total_times += session_time
        
        # for each assessment, and only this kind off session, the features below are processed
        # and a register are generated
        if (session_type == 'Assessment') & (test_set or len(session)>1):
            # search for event_code 4100, that represents the assessments trial
            all_attempts = session.query(f'event_code == {win_code[session_title]}')
            # then, check the numbers of wins and the number of losses
            true_attempts = all_attempts['event_data'].str.contains('true').sum()
            false_attempts = all_attempts['event_data'].str.contains('false').sum()
            
            features = dict()
            features['installation_id'] = session['installation_id'].iloc[-1]
            features['game_session'] = session['game_session'].iloc[-1]
            features['title'] = session_title
            features['world'] = session_world
            features['timestamp'] = session_timestamp
            features['timestamp_weekday'] = session_weekday
            features['timestamp_daytime'] = session_daytime
            features['total_times'] = session_total_times
            l = [v for v in event_code_count.values()]
            features['total_event_count'] = np.sum(l)            
            
            features.update(type_count.copy())
            features.update(world_count.copy())
            features.update(event_code_count.copy())
            features.update(event_id_count.copy())
            features.update(title_count.copy())
            features.update(title_event_code_count.copy())
            features.update(last_accuracy_title.copy())
            features['last_acc_this_title_lasttime'] = last_accuracy_title.copy()['last_acc_' + session_title_text]
            features['last_acc_all'] = accuracy
            features.update(accuracy_groups)
            features.update(acmu_correct_attempt)
            features.update(acmu_incorrect_attempt)
            features.update(acmu_accuracy_attempt)
            features.update(duration_title2)
            
            features['title_class_count'] = len(title_class_count)
            features['type_class_count'] = len(type_class_count)
            features['world_class_count'] = len(world_class_count)
            features['event_id_class_count'] = len([x for x in list(event_id_count.values()) if x > 0]) 
            features['event_code_class_count'] = len([x for x in list(event_code_count.values()) if x > 0]) 

            features['accumulated_correct_attempts'] = accumulated_correct_attempts
            features['accumulated_incorrect_attempts'] = accumulated_incorrect_attempts
            
            # the time spent in the app so far
            if assessment_durations == []:
                features['assessment_duration_mean'] = -1
            else:
                features['assessment_duration_mean'] = np.mean(assessment_durations)
           
            # the accurace is the all time wins divided by the all time attempts
            features['accumulated_accuracy'] = accumulated_accuracy/counter if counter > 0 else -1
            accuracy = true_attempts/(true_attempts+false_attempts) if (true_attempts+false_attempts) != 0 else -1
            accumulated_accuracy += accuracy
            
            # a feature of the current accuracy categorized
            # it is a counter of how many times this player was in each accuracy group
            if accuracy == 0:
                features['accuracy_group'] = 0
            elif accuracy == 1:
                features['accuracy_group'] = 3
            elif accuracy == 0.5:
                features['accuracy_group'] = 2
            else:
                features['accuracy_group'] = 1

            # mean of the all accuracy groups of this player
            features['accumulated_accuracy_group'] = accumulated_accuracy_group/counter if counter > 0 else -1
            # how many actions the player has done so far, it is initialized as 0 and updated some lines below
            features['accumulated_actions'] = accumulated_actions
            
            ## update
            accumulated_correct_attempts += true_attempts 
            accumulated_incorrect_attempts += false_attempts
            acmu_correct_attempt['acmu_correct_attempt_'+session_title_text] += true_attempts
            acmu_incorrect_attempt['acmu_incorrect_attempt_'+session_title_text] += false_attempts
            acmu_all = acmu_correct_attempt['acmu_correct_attempt_'+session_title_text] +\
                       acmu_incorrect_attempt['acmu_incorrect_attempt_'+session_title_text] 
            acmu_accuracy_attempt['acmu_accuracy_attempt_'+session_title_text] = -1 if acmu_all==0 else\
            acmu_correct_attempt['acmu_correct_attempt_'+session_title_text]/acmu_all

            assessment_durations.append(session_time)
            accuracy_groups['acmu_acc_gp_'+str(features['accuracy_group'])] += 1
            accumulated_accuracy_group += features['accuracy_group']
            last_accuracy_title['last_acc_' + session_title_text] = accuracy
            
            # there are some conditions to allow this features to be inserted in the datasets
            # if it's a test set, all sessions belong to the final dataset
            # it it's a train, needs to be passed throught this clausule: session.query(f'event_code == {win_code[session_title]}')
            # that means, must exist an event_code 4100 or 4110
            if test_set:
                all_assessments.append(features)
            elif true_attempts+false_attempts > 0:
                all_assessments.append(features)
                
            counter += 1
        
        ## update
        # this piece counts how many actions was made in each event_code so far
        def update_counters(counter: dict, col: str, perfix: str):
                num_of_session_count = Counter(session[col])
                for k in num_of_session_count.keys():
                    x = k
                    if col == 'title':
                        x = activities_labels[k]
                    counter['acmu_'+perfix+str(x)] += num_of_session_count[k]
                return counter
            
        type_count['acmu_type_'+str(session_type)] += 1
        world_count['acmu_world_'+world_map[session_world]] += 1   
        event_code_count = update_counters(event_code_count, "event_code","ev_code_")
        event_id_count = update_counters(event_id_count, "event_id","ev_id_")
        title_count = update_counters(title_count, 'title',"title_")
        title_event_code_count = update_counters(title_event_code_count, 'title_event_code',"ev_title_")
        
        duration_title['duration_title_' + activities_labels[session_title]].append(session_time)
        duration_title2 = duration_title.copy()
        for key in duration_title2:
            if len(duration_title2[key])==0:
                duration_title2[key] = 0
            else:
                duration_title2[key] = np.mean(duration_title2[key])
        
        # counts how many actions the player has done so far, used in the feature of the same name
        accumulated_actions += len(session)
        
        ## class update
        title_class_count.add(session_title)
        type_class_count.add(session_type)
        world_class_count.add(session_world)
        event_id_class_count.add(session_world)
        event_code_class_count.add(session_world)
        
    ## create user assessment 
#     if create_psuedo_assessment:
#         for key in world_count:
#             if world_count[key]>10:
#                 create_session(world=key[11:])
        
        
    # if it't the test_set, only the last assessment must be predicted, the previous are scraped
    if test_set:
        return all_assessments[-1]
    # in the train_set, all assessments goes to the dataset
    return all_assessments

def get_type_duration(train,type_to_title,activities_labels):
    for i in range(type_to_title.shape[0]):
        train['duration_type_{}'.format(type_to_title.iloc[i,0])] = 0
        for t in type_to_title.iloc[i,1]:        
            train['duration_type_{}'.format(type_to_title.iloc[i,0])] += train['duration_title_'+activities_labels[t]]
    return train

def get_world_duration(train,world_to_title,activities_labels):
    for i in range(world_to_title.shape[0]):
        train['duration_world_{}'.format(world_to_title.iloc[i,0])] = 0
        for t in world_to_title.iloc[i,1]:        
            train['duration_world_{}'.format(world_to_title.iloc[i,0])] += train['duration_title_'+activities_labels[t]]
    return train

## TODO
def get_this_title_type_world_duration(train):
    train['duration_this_world'] = 0
    train['duration_this_type'] = 0
    train['duration_this_title'] = 0
    for i in range(train.shape[0]):
        train.loc[i,'duration_this_world'] = train.loc[i,'duration_world_{}'.format(train.loc[i,'world'])]
        train.loc[i,'duration_this_type'] = train.loc[i,'duration_type_{}'.format(train.loc[i,'type'])]
        train.loc[i,'duration_this_title'] = train.loc[i,'duration_title_{}'.format(train.loc[i,'title'])]
    return train

## TODO
def get_this_world_type_count(train):
    return train

def get_train_and_test(train, test, win_code, list_of_user_activities, list_of_event_code,\
activities_labels, assess_titles, list_of_event_id, all_title_event_code):
    compiled_train = []
    compiled_test = []
    for i, (ins_id, user_sample) in tqdm(enumerate(train.groupby('installation_id', sort = False)), total =\
                                         train.installation_id.nunique()):
        user_data = get_data(user_sample, win_code, list_of_user_activities, list_of_event_code, activities_labels, assess_titles, list_of_event_id, all_title_event_code)
        compiled_train += user_data
    for ins_id, user_sample in tqdm(test.groupby('installation_id', sort = False), total = test.installation_id.nunique()):
        test_data = get_data(user_sample, win_code, list_of_user_activities, list_of_event_code, activities_labels, assess_titles, list_of_event_id, all_title_event_code, test_set = True)
        compiled_test.append(test_data)
    reduce_train = pd.DataFrame(compiled_train)
    reduce_test = pd.DataFrame(compiled_test)
    
    ## duration_type
    type_to_title = train.groupby(['type'])['title'].unique().reset_index()
    world_to_title = train.groupby(['world'])['title'].unique().reset_index()
    reduce_train = get_type_duration(reduce_train,type_to_title,activities_labels)
    reduce_test = get_type_duration(reduce_test,type_to_title,activities_labels)
    reduce_train = get_world_duration(reduce_train,world_to_title,activities_labels)
    reduce_test = get_world_duration(reduce_test,world_to_title,activities_labels)
    
    ## get "this" feature
    ## this duration, acc, history
    
    categoricals = ['title']
    return reduce_train, reduce_test, categoricals

