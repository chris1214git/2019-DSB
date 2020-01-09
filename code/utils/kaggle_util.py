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
    
    return train, test, train_labels, win_code, list_of_user_activities, list_of_event_code, activities_labels, assess_titles, list_of_event_id, all_title_event_code, activities_map

def get_data_0106(user_sample, win_code, list_of_user_activities, list_of_event_code, activities_labels, assess_titles, list_of_event_id, all_title_event_code, activities_map, test_set=False, create_psuedo_assessment=False):
    ''' 
    The user_sample is a DataFrame from train or test where the only one 
    installation_id is filtered
    And the test_set parameter is related with the labels processing, that is only requered
    if test_set=False
    '''
    
    ## Set origin parameters
    
    ## constants and parameters declaration            
    all_assessments = []
    pseudo_train = []
    accumulated_accuracy_group = 0
    accumulated_accuracy = -1
    accumulated_correct_attempts = 0 
    accumulated_incorrect_attempts = 0
    accumulated_actions = 0
    counter = 0
    counter_all_session = 0
    
    assessment_durations = []
    accuracy = -1
    
    #  User acc history
    ## acmu acc each title
    acmu_correct_attempt = {'acmu_correct_attempt_' + title: 0 for title in assess_titles}
    acmu_incorrect_attempt = {'acmu_incorrect_attempt_' + title: 0 for title in assess_titles}
    acmu_accuracy_attempt = {'acmu_accuracy_attempt_' + title: -1 for title in assess_titles}
    ## acmu accgp total
    accuracy_groups = {'acmu_acc_gp_0':0, 'acmu_acc_gp_1':0, 'acmu_acc_gp_2':0, 'acmu_acc_gp_3':0}
    acmu_accuracy_groups_title = {'acmu_accuracy_groups_' + title: [] for title in assess_titles}
    ## last acc each title
    last_accuracy_title = {'last_acc_' + title: -1 for title in assess_titles}
    last2_accuracy_title = {'last2_acc_' + title: -1 for title in assess_titles}
    last3_accuracy_title = {'last3_acc_' + title: -1 for title in assess_titles}
 
    # new features: time spent in each activity
    last_session_time_sec = 0
    
    ## learning history
    type_count = {'acmu_type_Clip':0, 'acmu_type_Activity': 0, 'acmu_type_Assessment': 0, 'acmu_type_Game':0}
    world_count = {'acmu_world_NONE':0, 'acmu_world_MAGMAPEAK': 0, 'acmu_world_TREETOPCITY': 0, 'acmu_world_CRYSTALCAVES':0}
    event_code_count = {'acmu_ev_code_'+str(eve): 0 for eve in list_of_event_code}
    event_id_count = {'acmu_ev_id_'+str(eve): 0 for eve in list_of_event_id}
    title_count = {'acmu_title_'+str(eve): 0 for eve in activities_labels.values()} 
    title_event_code_count = {'acmu_ev_title_'+str(eve): 0 for eve in all_title_event_code}
   
    ## timestamp
    duration_title = {'duration_title_' + title: [] for title in activities_labels.values()}
    duration_title2 = {'duration_title_' + title: 0 for title in activities_labels.values()}

    ## media sequence change
    media_seq_change = 0
    
    ## map
    world_map = {0:'NONE', 1:'MAGMAPEAK', 2:'TREETOPCITY', 3:'CRYSTALCAVES'}
    assessment_to_title ={"Mushroom Sorter (Assessment)":['Tree Top City - Level 1', 'Ordering Spheres', 'All Star Sorting', 'Costume Box',\
                                                         'Fireworks (Activity)', '12 Monkeys', 'Tree Top City - Level 2', \
                                                         'Flower Waterer (Activity)', "Pirate's Tale", 'Mushroom Sorter (Assessment)'],
                            "Bird Measurer (Assessment)":['Air Show', 'Treasure Map', 'Tree Top City - Level 3', 'Crystals Rule', 'Rulers',\
                                                          'Bug Measurer (Activity)', 'Bird Measurer (Assessment)'],
                          "Cauldron Filler (Assessment)":['Magma Peak - Level 1', 'Sandcastle Builder (Activity)', 'Slop Problem',\
                                                          'Scrub-A-Dub', 'Watering Hole (Activity)', 'Magma Peak - Level 2', 'Dino Drink',\
                                                          'Bubble Bath', 'Bottle Filler (Activity)', 'Dino Dive', \
                                                          'Cauldron Filler (Assessment)'],
                          "Cart Balancer (Assessment)":['Crystal Caves - Level 1', 'Chow Time', 'Balancing Act',\
                                                        'Chicken Balancer (Activity)', 'Lifting Heavy Things', 'Crystal Caves - Level 2',\
                                                        'Honey Cake', 'Happy Camel', 'Cart Balancer (Assessment)'],
                          "Chest Sorter (Assessment)":['Leaf Leader', 'Crystal Caves - Level 3', 'Heavy, Heavier, Heaviest', 'Pan Balance',\
                                                       'Egg Dropper (Activity)', 'Chest Sorter (Assessment)']}

    title_sequence = ['Welcome to Lost Lagoon!', 'Tree Top City - Level 1', 'Ordering Spheres', 'All Star Sorting', 'Costume Box',\
                      'Fireworks (Activity)', '12 Monkeys', 'Tree Top City - Level 2', 'Flower Waterer (Activity)', "Pirate's Tale",\
                      'Mushroom Sorter (Assessment)', 'Air Show', 'Treasure Map', 'Tree Top City - Level 3', 'Crystals Rule', 'Rulers',\
                      'Bug Measurer (Activity)', 'Bird Measurer (Assessment)', 'Magma Peak - Level 1', 'Sandcastle Builder (Activity)',\
                      'Slop Problem', 'Scrub-A-Dub', 'Watering Hole (Activity)', 'Magma Peak - Level 2', 'Dino Drink', 'Bubble Bath',\
                      'Bottle Filler (Activity)', 'Dino Dive', 'Cauldron Filler (Assessment)', 'Crystal Caves - Level 1', 'Chow Time',\
                      'Balancing Act', 'Chicken Balancer (Activity)', 'Lifting Heavy Things', 'Crystal Caves - Level 2', 'Honey Cake',\
                      'Happy Camel', 'Cart Balancer (Assessment)', 'Leaf Leader', 'Crystal Caves - Level 3', 'Heavy, Heavier, Heaviest',\
                      'Pan Balance', 'Egg Dropper (Activity)', 'Chest Sorter (Assessment)']
    title_sequence = {title:i for i,title in enumerate(title_sequence)}
    
    session_total_times = 0

    last_type = 'None'
    last_title = -1
    last_assess_title = 'None'
    last_title_sequence = 0
    reset_cnt = 0
    session_timestamp = user_sample['timestamp'].iloc[0]
    session_title_sequence = 0
    
    # itarates through each session of one instalation_id
    for i, session in user_sample.groupby('game_session', sort=False):
        # i = game_session_id
        # session is a DataFrame that contain only one game_session
        counter_all_session+=1
        
        # get some sessions information
        session_type = session['type'].iloc[0]
        session_title = session['title'].iloc[0]
        session_title_text = activities_labels[session_title]
        session_world = session['world'].iloc[0]
        session_weekday = session['timestamp_weekday'].iloc[0]
        session_daytime = session['timestamp_daytime'].iloc[0]
        last_session_timestamp_diff = (pd.Timedelta(session['timestamp'].iloc[0] - session_timestamp).seconds)// 60
        session_timestamp = session['timestamp'].iloc[0]
        
        last_title_sequence_diff = title_sequence[session_title_text] - session_title_sequence
        media_seq_change += abs(last_title_sequence_diff)
        session_title_sequence = title_sequence[session_title_text]
        
        ## modified timestamp
        ## minus session time when diff time between events is more than 10 min
        x = session.timestamp
        x = (x - x.shift(1)).astype('timedelta64[s]')
        x = np.sum(x[x>1*60])
        session_time = (session.iloc[-1, 2] - session.iloc[0, 2] ).seconds - x
        
        # for each assessment, and only this kind off session, the features below are processed
        # and a register are generated
        if (session_type == 'Assessment') & (test_set or len(session)>1):
            # search for event_code 4100, that represents the assessments trial
            all_attempts = session.query(f'event_code == {win_code[session_title]}')
            # then, check the numbers of wins and the number of losses
            true_attempts = all_attempts['event_data'].str.contains('true').sum()
            false_attempts = all_attempts['event_data'].str.contains('false').sum()
            # titles related to this assessment
            this_assess_title = assessment_to_title[activities_labels[session_title]]
            
            # default features
            features = dict()
            features['installation_id'] = session['installation_id'].iloc[-1]
            features['game_session'] = session['game_session'].iloc[-1]
            features['title'] = session_title
            features['world'] = session_world
            features['timestamp'] = session_timestamp
            features['timestamp_weekday'] = session_weekday
            features['timestamp_daytime'] = session_daytime    
            features['last_session_timestamp_diff'] = last_session_timestamp_diff    
            
            # counting features
            features.update(type_count)
            features.update(world_count)
            features.update(event_code_count)
            features.update(event_id_count)
            features.update(title_count)
            features.update(title_event_code_count)
            
            # counting features class num
            features['title_class_count'] = len(list(filter(lambda x: x > 0, title_count.values())))
            features['type_class_count'] = len(list(filter(lambda x: x > 0, type_count.values())))
            features['world_class_count'] = len(list(filter(lambda x: x > 0, world_count.values())))
            features['event_id_class_count'] = len(list(filter(lambda x: x > 0, event_id_count.values())))
            features['event_code_class_count'] = len(list(filter(lambda x: x > 0, event_code_count.values())))
            features['title_event_code_class_count'] = len(list(filter(lambda x: x > 0, title_event_code_count.values())))
            
            ## This assessment related features
            
            # this world
            features['this_world_title_count'] = world_count['acmu_world_'+world_map[session_world]]
            # this title count(same assessment)
            features['this_title_count'] = title_count['acmu_title_'+session_title_text]
            
            # this title count(all title related to this assessment)
            features['this_assessment_title_count'] = 0
            for title in this_assess_title:
                features['this_assessment_title_count'] += title_count['acmu_title_'+title]
            features['this_assessment_title_class_count'] = 0
            features['this_assessment_title_class_count_standardize'] = 0
            
             # this title class count(all title related to this assessment)
            for title in this_assess_title:
                if title_count['acmu_title_'+title]>0:
                    features['this_assessment_title_class_count'] += 1
            features['this_assessment_title_class_count_standardize'] = features['this_assessment_title_class_count']/len(this_assess_title)
            features['last_assessment_title_same'] = session_title_text == last_assess_title
 
            # last acc
            features.update(last_accuracy_title)
            features.update(last2_accuracy_title)
            features.update(last3_accuracy_title)
            features['last_acc_this_title'] = last_accuracy_title['last_acc_' + session_title_text]
            features['last2_acc_this_title'] = last2_accuracy_title['last2_acc_' + session_title_text]
            features['last3_acc_this_title'] = last3_accuracy_title['last3_acc_' + session_title_text]
            features['last12_acc_this_titl_same'] = features['last_acc_this_title']==features['last2_acc_this_title']
            features['last123_acc_this_titl_same'] = (features['last_acc_this_title']==features['last2_acc_this_title']) &\
                                                    (features['last_acc_this_title']==features['last3_acc_this_title'])
            features['last12_acc_this_titl_mean'] = (features['last_acc_this_title']+features['last2_acc_this_title'])/2
            features['last123_acc_this_titl_mean'] = (features['last_acc_this_title']+features['last2_acc_this_title']+\
                                                     features['last3_acc_this_title'])/3
            features['last_acc_all'] = accuracy
            
            # title sequence diff
            features['last_title_sequence_diff'] = last_title_sequence_diff
            features['media_sequence_change'] = media_seq_change/counter_all_session
            
            # acmu acc
            features.update(accuracy_groups)
            features.update(acmu_correct_attempt)
            features.update(acmu_incorrect_attempt)
            features.update(acmu_accuracy_attempt)
            
            features.update(duration_title2)
            features['this_assessment_title_duration'] = 0
            for title in this_assess_title:
                features['this_assessment_title_duration'] += duration_title2['duration_title_'+title]

            features['accumulated_correct_attempts'] = accumulated_correct_attempts
            features['accumulated_incorrect_attempts'] = accumulated_incorrect_attempts
            
            features['total_times'] = session_total_times            
            features['total_event_count'] = sum(event_code_count.values())     
            
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
            
            acmu_acgp_this = np.mean(acmu_accuracy_groups_title['acmu_accuracy_groups_'+session_title_text]) if \
            len(acmu_accuracy_groups_title['acmu_accuracy_groups_'+session_title_text])>0 else -1
            features['acmu_accuracy_group_this_title'] = acmu_acgp_this 
            
            ## update
            accumulated_correct_attempts += true_attempts 
            accumulated_incorrect_attempts += false_attempts
            acmu_correct_attempt['acmu_correct_attempt_'+session_title_text] += true_attempts
            acmu_incorrect_attempt['acmu_incorrect_attempt_'+session_title_text] += false_attempts
            acmu_all = acmu_correct_attempt['acmu_correct_attempt_'+session_title_text] +\
                       acmu_incorrect_attempt['acmu_incorrect_attempt_'+session_title_text] 
            acmu_accuracy_attempt['acmu_accuracy_attempt_'+session_title_text] = -1 if acmu_all==0 else\
            acmu_correct_attempt['acmu_correct_attempt_'+session_title_text]/acmu_all
            
            acmu_accuracy_groups_title['acmu_accuracy_groups_'+session_title_text].append(features['accuracy_group'])
            
            
            assessment_durations.append(session_time)
            accuracy_groups['acmu_acc_gp_'+str(features['accuracy_group'])] += 1
            accumulated_accuracy_group += features['accuracy_group']
            last3_accuracy_title['last3_acc_' + session_title_text] = last2_accuracy_title['last2_acc_' + session_title_text]
            last2_accuracy_title['last2_acc_' + session_title_text] = last_accuracy_title['last_acc_' + session_title_text]
            last_accuracy_title['last_acc_' + session_title_text] = accuracy
            
            last_assess_title = session_title_text
            
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
        
        if last_title!=session_title: 
            type_count['acmu_type_'+str(session_type)] += 1
            world_count['acmu_world_'+world_map[session_world]] += 1   
            title_count = update_counters(title_count, 'title',"title_")            
        last_title = session_title
        
        event_code_count = update_counters(event_code_count, "event_code","ev_code_")
        event_id_count = update_counters(event_id_count, "event_id","ev_id_")
        title_event_code_count = update_counters(title_event_code_count, 'title_event_code',"ev_title_")
        
        duration_title['duration_title_' + activities_labels[session_title]].append(session_time)
        duration_title2 = duration_title.copy()
        for key in duration_title2:
            if len(duration_title2[key])==0:
                duration_title2[key] = 0
            else:
                duration_title2[key] = np.mean(duration_title2[key])
        
        session_total_times += session_time
        # counts how many actions the player has done so far, used in the feature of the same name
        accumulated_actions += len(session)
        
        
    # if it't the test_set, only the last assessment must be predicted, the previous are scraped
    if test_set:
        return all_assessments[-1]
    # in the train_set, all assessments goes to the dataset
    return all_assessments, pseudo_train

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

def get_this_title_world_duration(train,activities_labels):
    train['duration_this_world'] = 0
    train['duration_this_title'] = 0
#     world_map = {0:'NONE', 1:'MAGMAPEAK', 2:'TREETOPCITY', 3:'CRYSTALCAVES'}
    for i in range(train.shape[0]):
#         train.loc[i,'duration_this_world'] = train.loc[i,'duration_world_{}'.format(world_map[train.loc[i,'world']])]
        train.loc[i,'duration_this_world'] = train.loc[i,'duration_world_{}'.format(train.loc[i,'world'])]
        train.loc[i,'duration_this_title'] = train.loc[i,'duration_title_{}'.format(activities_labels[train.loc[i,'title']])]
    return train

## TODO
def get_this_world_type_count(train):
    return train

## 計算前1~5次session是多久以前(單位:hr)
def add_time_diff_hour(reduce_train):
    reduce_train['time_diff_hour_session1']=0
    reduce_train['time_diff_hour_session2']=0
    reduce_train['time_diff_hour_session3']=0
    reduce_train['time_diff_hour_session4']=0
    reduce_train['time_diff_hour_session5']=0
    iid_list = reduce_train['installation_id'].unique()
    cnt=0
    print('add_time_diff_hour ...')
    for iid in iid_list:
        cnt+=1
        if cnt%300==0:
            pass
#             print(cnt)
        idx = reduce_train['installation_id']==iid
        reduce_train.loc[idx,'time_diff_hour_session1'] = reduce_train.loc[idx,'timestamp'].diff(1).fillna(pd.Timedelta('-1 days')).apply(lambda x: x.total_seconds()//60)
        reduce_train.loc[idx,'time_diff_hour_session2'] = reduce_train.loc[idx,'timestamp'].diff(2).fillna(pd.Timedelta('-1 days')).apply(lambda x: x.total_seconds()//60)
        reduce_train.loc[idx,'time_diff_hour_session3'] = reduce_train.loc[idx,'timestamp'].diff(3).fillna(pd.Timedelta('-1 days')).apply(lambda x: x.total_seconds()//60)
        reduce_train.loc[idx,'time_diff_hour_session4'] = reduce_train.loc[idx,'timestamp'].diff(4).fillna(pd.Timedelta('-1 days')).apply(lambda x: x.total_seconds()//60)
        reduce_train.loc[idx,'time_diff_hour_session5'] = reduce_train.loc[idx,'timestamp'].diff(5).fillna(pd.Timedelta('-1 days')).apply(lambda x: x.total_seconds()//60)
    return reduce_train

## 計算前30min~7day有多少個session
def add_session_cnt(reduce_train):
    iid_list = reduce_train['installation_id'].unique()
    reduce_train['session_cnt_10min']=0
    reduce_train['session_cnt_30min']=0
    reduce_train['session_cnt_1hour']=0
    reduce_train['session_cnt_1day']=0
    reduce_train['session_cnt_7day']=0
    print('add_session_cnt ... ...')
    cnt=0
    reduce_train2 = reduce_train.copy().set_index('timestamp')
    for iid in iid_list:
        cnt+=1
        if cnt%300==0:
            pass
#             print(cnt)
        idx = reduce_train['installation_id']==iid
        idx2 = reduce_train2['installation_id']==iid

        reduce_train.loc[idx,'session_cnt_10min'] = reduce_train2.loc[idx2,'title'].rolling('10min').count().values
        reduce_train.loc[idx,'session_cnt_30min'] = reduce_train2.loc[idx2,'title'].rolling('30min').count().values
        reduce_train.loc[idx,'session_cnt_1hour'] = reduce_train2.loc[idx2,'title'].rolling('60min').count().values
        reduce_train.loc[idx,'session_cnt_1day'] = reduce_train2.loc[idx2,'title'].rolling('1D').count().values
        reduce_train.loc[idx,'session_cnt_7day'] = reduce_train2.loc[idx2,'title'].rolling('7D').count().values 

    return reduce_train
        

def get_train_and_test(train, test, win_code, list_of_user_activities, list_of_event_code,\
activities_labels, assess_titles, list_of_event_id, all_title_event_code, activities_map, create_psuedo_assessment=False):
    compiled_train = []
    compiled_psuedo_train = []
    compiled_test = []
    for i, (ins_id, user_sample) in tqdm(enumerate(train.groupby('installation_id', sort = False)), total =\
                                         train.installation_id.nunique()):
        user_data, pseudo_data = get_data_0106(user_sample, win_code, list_of_user_activities, list_of_event_code, activities_labels, assess_titles, list_of_event_id, all_title_event_code, activities_map,create_psuedo_assessment=create_psuedo_assessment)
        compiled_train += user_data
        compiled_psuedo_train += pseudo_data
    for ins_id, user_sample in tqdm(test.groupby('installation_id', sort = False), total = test.installation_id.nunique()):
        test_data = get_data_0106(user_sample, win_code, list_of_user_activities, list_of_event_code, activities_labels, assess_titles, list_of_event_id, all_title_event_code, activities_map, test_set = True)
        compiled_test.append(test_data)
    reduce_train = pd.DataFrame(compiled_train)
    reduce_psuedo_train = pd.DataFrame(compiled_psuedo_train)
    reduce_test = pd.DataFrame(compiled_test)
    
    ## duration_type
    type_to_title = train.groupby(['type'])['title'].unique().reset_index()
    world_to_title = train.groupby(['world'])['title'].unique().reset_index()
    
    reduce_train = get_type_duration(reduce_train,type_to_title,activities_labels)
    reduce_test = get_type_duration(reduce_test,type_to_title,activities_labels)
    
    reduce_train = get_world_duration(reduce_train,world_to_title,activities_labels)
    reduce_test = get_world_duration(reduce_test,world_to_title,activities_labels)
    
    reduce_train = get_this_title_world_duration(reduce_train,activities_labels)
    reduce_test = get_this_title_world_duration(reduce_test,activities_labels)
    
    reduce_train = add_time_diff_hour(reduce_train)
    reduce_test = add_time_diff_hour(reduce_test)
    
    reduce_train = add_session_cnt(reduce_train)
    reduce_test = add_session_cnt(reduce_test)   
    
    if create_psuedo_assessment:
        reduce_psuedo_train = get_type_duration(reduce_psuedo_train,type_to_title,activities_labels)
        reduce_psuedo_train = get_world_duration(reduce_psuedo_train,world_to_title,activities_labels)
        reduce_psuedo_train = get_this_title_world_duration(reduce_psuedo_train,activities_labels)
        reduce_psuedo_train = add_time_diff_hour(reduce_psuedo_train)
        reduce_psuedo_train = add_session_cnt(reduce_psuedo_train)

    
    ## get "this" feature
    
    
    categoricals = ['title']
    return reduce_train, reduce_psuedo_train, reduce_test, categoricals


