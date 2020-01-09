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


def reset_param_get_data(assess_titles,list_of_event_code,list_of_event_id,activities_labels,all_title_event_code):
    ## constants and parameters declaration            
    all_assessments = []
    accumulated_accuracy_group = 0
    accumulated_accuracy = -1
    accumulated_correct_attempts = 0 
    accumulated_incorrect_attempts = 0
    accumulated_actions = 0
    counter = 0
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
 
    # new features: time spent in each activity
    last_session_time_sec = 0
    
    ## learning history
    type_count = {'acmu_type_Clip':0, 'acmu_type_Activity': 0, 'acmu_type_Assessment': 0, 'acmu_type_Game':0}
    world_count = {'acmu_world_NONE':0, 'acmu_world_MAGMAPEAK': 0, 'acmu_world_TREETOPCITY': 0, 'acmu_world_CRYSTALCAVES':0}
    world_map = {0:'NONE', 1:'MAGMAPEAK', 2:'TREETOPCITY', 3:'CRYSTALCAVES'}
    event_code_count = {'acmu_ev_code_'+str(eve): 0 for eve in list_of_event_code}
    event_id_count = {'acmu_ev_id_'+str(eve): 0 for eve in list_of_event_id}
    title_count = {'acmu_title_'+str(eve): 0 for eve in activities_labels.values()} 
    title_event_code_count = {'acmu_ev_title_'+str(eve): 0 for eve in all_title_event_code}
   
    ## timestamp
    duration_title = {'duration_title_' + title: [] for title in activities_labels.values()}
    duration_title2 = {'duration_title_' + title: 0 for title in activities_labels.values()}
    
    ## 種類統計
    title_class_count = set()
    type_class_count = set()
    world_class_count = set()
    event_id_class_count = set()
    event_code_class_count = set()
    
    session_total_times = 0
    
    return (all_assessments,
            accumulated_accuracy_group,
            accumulated_accuracy,
            accumulated_correct_attempts,
            accumulated_incorrect_attempts,
            accumulated_actions,
            counter,
            assessment_durations,
            accuracy,
            acmu_correct_attempt,
            acmu_incorrect_attempt,
            acmu_accuracy_attempt,
            accuracy_groups,
            acmu_accuracy_groups_title,
            last_accuracy_title,
            last2_accuracy_title,
            last_session_time_sec,
            type_count,
            world_count,
            world_map,
            event_code_count,
            event_id_count,
            title_count,
            title_event_code_count,
            duration_title,
            duration_title2,
            title_class_count,
            type_class_count,
            world_class_count,
            event_id_class_count,
            event_code_class_count,
            session_total_times)
    
def get_data_reset(user_sample, win_code, list_of_user_activities, list_of_event_code, activities_labels, assess_titles, list_of_event_id, all_title_event_code, activities_map, test_set=False, create_psuedo_assessment=False):
    ''' 
    The user_sample is a DataFrame from train or test where the only one 
    installation_id is filtered
    And the test_set parameter is related with the labels processing, that is only requered
    if test_set=False
    '''
    
#     ## Set origin parameters
    
#     ## constants and parameters declaration            
#     all_assessments = []
#     accumulated_accuracy_group = 0
#     accumulated_accuracy = -1
#     accumulated_correct_attempts = 0 
#     accumulated_incorrect_attempts = 0
#     accumulated_actions = 0
#     counter = 0
#     assessment_durations = []
#     accuracy = -1
    
#     #  User acc history
#     ## acmu acc each title
#     acmu_correct_attempt = {'acmu_correct_attempt_' + title: 0 for title in assess_titles}
#     acmu_incorrect_attempt = {'acmu_incorrect_attempt_' + title: 0 for title in assess_titles}
#     acmu_accuracy_attempt = {'acmu_accuracy_attempt_' + title: -1 for title in assess_titles}
#     ## acmu accgp total
#     accuracy_groups = {'acmu_acc_gp_0':0, 'acmu_acc_gp_1':0, 'acmu_acc_gp_2':0, 'acmu_acc_gp_3':0}
#     acmu_accuracy_groups_title = {'acmu_accuracy_groups_' + title: [] for title in assess_titles}
#     ## last acc each title
#     last_accuracy_title = {'last_acc_' + title: -1 for title in assess_titles}
#     last2_accuracy_title = {'last2_acc_' + title: -1 for title in assess_titles}
 
#     # new features: time spent in each activity
#     last_session_time_sec = 0
    
#     ## learning history
#     type_count = {'acmu_type_Clip':0, 'acmu_type_Activity': 0, 'acmu_type_Assessment': 0, 'acmu_type_Game':0}
#     world_count = {'acmu_world_NONE':0, 'acmu_world_MAGMAPEAK': 0, 'acmu_world_TREETOPCITY': 0, 'acmu_world_CRYSTALCAVES':0}
#     world_map = {0:'NONE', 1:'MAGMAPEAK', 2:'TREETOPCITY', 3:'CRYSTALCAVES'}
#     event_code_count = {'acmu_ev_code_'+str(eve): 0 for eve in list_of_event_code}
#     event_id_count = {'acmu_ev_id_'+str(eve): 0 for eve in list_of_event_id}
#     title_count = {'acmu_title_'+str(eve): 0 for eve in activities_labels.values()} 
#     title_event_code_count = {'acmu_ev_title_'+str(eve): 0 for eve in all_title_event_code}
   
#     ## timestamp
#     duration_title = {'duration_title_' + title: [] for title in activities_labels.values()}
#     duration_title2 = {'duration_title_' + title: 0 for title in activities_labels.values()}
    
#     ## 種類統計
#     title_class_count = set()
#     type_class_count = set()
#     world_class_count = set()
#     event_id_class_count = set()
#     event_code_class_count = set()
    
#     session_total_times = 0
    
    
    (all_assessments,
    accumulated_accuracy_group,
    accumulated_accuracy,
    accumulated_correct_attempts,
    accumulated_incorrect_attempts,
    accumulated_actions,
    counter,
    assessment_durations,
    accuracy,
    acmu_correct_attempt,
    acmu_incorrect_attempt,
    acmu_accuracy_attempt,
    accuracy_groups,
    acmu_accuracy_groups_title,
    last_accuracy_title,
    last2_accuracy_title,
    last_session_time_sec,
    type_count,
    world_count,
    world_map,
    event_code_count,
    event_id_count,
    title_count,
    title_event_code_count,
    duration_title,
    duration_title2,
    title_class_count,
    type_class_count,
    world_class_count,
    event_id_class_count,
    event_code_class_count,
    session_total_times) = reset_param_get_data(assess_titles,list_of_event_code,list_of_event_id,activities_labels,all_title_event_code)
    
    last_type = 'None'
    last_title = -1
    reset_cnt = 0
    
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
        
        ## reset parameter if title restart
        if session_title_text=='Welcome to Lost Lagoon!':
            reset_cnt+=1
            if reset_cnt==3:
                reset_cnt = 0
                (all_assessments,
                accumulated_accuracy_group,
                accumulated_accuracy,
                accumulated_correct_attempts,
                accumulated_incorrect_attempts,
                accumulated_actions,
                counter,
                assessment_durations,
                accuracy,
                acmu_correct_attempt,
                acmu_incorrect_attempt,
                acmu_accuracy_attempt,
                accuracy_groups,
                acmu_accuracy_groups_title,
                last_accuracy_title,
                last2_accuracy_title,
                last_session_time_sec,
                type_count,
                world_count,
                world_map,
                event_code_count,
                event_id_count,
                title_count,
                title_event_code_count,
                duration_title,
                duration_title2,
                title_class_count,
                type_class_count,
                world_class_count,
                event_id_class_count,
                event_code_class_count,
                session_total_times) = reset_param_get_data(assess_titles,list_of_event_code,list_of_event_id,activities_labels,all_title_event_code)
        
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
            features.update(last2_accuracy_title.copy())
            features['last_acc_this_title_lasttime'] = last_accuracy_title.copy()['last_acc_' + session_title_text]
            features['last2_acc_this_title_lasttime'] = last2_accuracy_title.copy()['last2_acc_' + session_title_text]
            
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
            last2_accuracy_title['last2_acc_' + session_title_text] = last_accuracy_title['last_acc_' + session_title_text]
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
        
#         if last_title!=session_title: 
        type_count['acmu_type_'+str(session_type)] += 1
        world_count['acmu_world_'+world_map[session_world]] += 1   
        title_count = update_counters(title_count, 'title',"title_")
        
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
        
        ## class update
        title_class_count.add(session_title)
        type_class_count.add(session_type)
        world_class_count.add(session_world)
        event_id_class_count.add(session_world)
        event_code_class_count.add(session_world)
    
    
    ## create user assessment
    pseudo_train=[]
    if create_psuedo_assessment:
        for key in world_count:
            title = []
            world = 0
            if key[11:]=='MAGMAPEAK':
                title = [activities_map['Cauldron Filler (Assessment)']]
                world = 1
            elif key[11:]=='TREETOPCITY':
                title = [activities_map['Mushroom Sorter (Assessment)'], activities_map['Bird Measurer (Assessment)']]
                world = 2
            elif key[11:]=='CRYSTALCAVES':
                title = [activities_map['Chest Sorter (Assessment)'], activities_map['Cart Balancer (Assessment)']]
                world = 3
            else:
                continue
            if world_count[key]>10:
                session_type = 'Assessment'
                session_title = title[0]
                session_title_text = activities_labels[session_title]
                print(session_title,session_title_text)
                session_world = world
                session_weekday = user_sample['timestamp_weekday'].iloc[-1]
                session_daytime = user_sample['timestamp_daytime'].iloc[-1]
                session_timestamp = user_sample['timestamp'].iloc[-1]

                features = dict()
                features['installation_id'] = user_sample['installation_id'].iloc[-1]
                features['game_session'] = user_sample['game_session'].iloc[-1]+'_pseudo'
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
                pseudo_train.append(features)
        
    # if it't the test_set, only the last assessment must be predicted, the previous are scraped
    if test_set:
        return all_assessments[-1]
    # in the train_set, all assessments goes to the dataset
    return all_assessments, pseudo_train