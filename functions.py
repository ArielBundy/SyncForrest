#!/usr/bin/env python
# coding: utf-8

# In[ ]:
def load_csv(csv_path):
    """
    """
    # imports
    import os
    import csv 
    # initialization
    file = open(csv_path)
    reader = csv.reader(file, delimiter="\t")
    demo,labels,time_arr,value_arr = [],[],[],[]
    # use a statements to collect demographics and data seperatly
    collect_data = False
    collect_demo = True
    ended = False
    for row in reader:
        line = row[0].split(',')
        if 'ended' in row[0]:
            ended = True
            collect_data = False  
        if collect_demo:
            demo.append(line)
        if collect_data:
            time_arr.append(float(line[0]))
            value_arr.append(float(line[1])) 
        if 'Time,X' in row[0]:
            labels = line
            collect_data = True
            collect_demo = False   
    demo = demo[:-1]# delete labels from last line of demo 
    if ended:
        demo.append(['experiment did not finish'])
    return demo,labels,time_arr,value_arr

def verify(demo,labels,time_arr,value_arr):
    """
    """
    # valid is true until proven otherwise
    valid = True
    # check if ended before time
    if demo[-1][0] == 'experiment did not finish':
        valid = False
        print('experiment did not finish')  
    # check labels
    if labels[0] != 'Time' or labels[1]!= 'X':
        valid = False
        print('check labels')  
    # check length of time and value arrays (have to match)
    if len(time_arr) != len(value_arr):
        valid = False
        print('time and value arrays are not of same size')
    return valid
# In[ ]:
def plotInterpolation(rawX,rawY,newX,newY):
    """
    """
    # imports
    import matplotlib.pyplot as plt
    # find first correction
    counter = 0
    for x1,x2 in zip(rawY,newY):
        if x1 != x2:
            counter += 1
            break
        else:
            counter+=1
    # overlook before and after
    fig, axes = plt.subplots(2, 1, figsize=(14, 6))
    axes[0].set_title('interpolation overlook')
    axes[0].set_ylabel('rating')
    axes[0].plot(rawX,rawY,'k',label='before')
    axes[0].plot(newX,newY,'r',label='after')
    axes[0].legend()
    # plot first interpolation incident
    axes[1].set_title('first incident zoom-in')
    axes[1].set_ylabel('rating')
    axes[1].set_xlabel('time(sec)')
    axes[1].plot(rawX[counter-5:counter+10],rawY[counter-5:counter+10],'k',label='before')
    axes[1].plot(newX[counter-5:counter+10],newY[counter-5:counter+10],'ro--',label='after')
    axes[1].legend()
    plt.show()

def timeInterpolation(time_arr,value_arr,samp_rate):
    """
    """
    # import
    import numpy as np
    # initialization
    x = np.array(time_arr)
    y = np.array(value_arr)
    start = round(time_arr[0] * 10)
    end = round(time_arr[-1] * 10)
    s = round(10*samp_rate) 
    # make new x axis 
    t =  [x/10.0 for x in range(start, end+s, s)]
    newX = np.array(t)
    # linear interpolation using numpy.interp
    newY = np.interp(newX,x,y)
    # plot interpolation
    plotInterpolation(x,y,newX,newY)
    return newX,newY

def smoothing(x,y,win_size):
    """
    this function preforms sliding window averginge and returns 
    same length smoothed output
    """
    # import
    import numpy as np
    kernel = np.ones(win_size) / win_size
    ySmooth = np.convolve(y, kernel, mode='same')
    plotCheck(x,y,ySmooth,title = 'smooth vs unsmooth',yAx ="Rating [a.u]")
    return ySmooth

def plotCheck(x,y,yNew,title,yAx,plotAll=False):
    """
    """
    # import
    import matplotlib.pyplot as plt
    # find first time difference
    counter = 0
    for x1,x2 in zip(y,yNew):
        if x1 != x2:
            counter += 1
            break
        else:
            counter+=1     
    start = counter
    end = counter + 100 # 10 seconds
    # overlook before and after
    fig, axes = plt.subplots(1, 1, figsize=(7, 3))
    axes.set_title(title)
    axes.set_ylabel(yAx)
    axes.set_xlabel('time [sec]')
    if not plotAll:
        axes.plot(x[start:end],y[start:end],'k',label='before')
        axes.plot(x[start:end],yNew[start:end],'r--',label='after')
        axes.legend()
    else:
        axes.plot(x,y,color='k',label='before')
        axes.plot(x,yNew,color='r--',label='after')
        axes.legend()
    plt.show()

def demeanArray(array):
    '''
    this function takes in a numpy array and returnes a fisher transformed array
    '''
    # import
    import numpy as np
    #print('mean before z transformation =',np.mean(array),'std =',np.std(array))
    z = (array - np.mean(array)) / np.std(array)
    #print('mean after z transformation =',np.mean(z),'std =',np.std(z),'\n')
    return z

def makeSameSize(runList,tList):
    """
    this function takes in a list of arrays (ratings) and a list of time arrays, finds the shortest, corrects
    all arrays to be in same length and picks the shortest time array. 
    it returns a matrix of runs x ratings and the time array that match matrix row dimenson
    """
    # import
    import numpy as np
    # initialization of a large number for first iteration
    shortest = 999999999
    # iterate over runs in runList
    for i,run in enumerate(runList):
        if run.size <= shortest:
            # assign size and time array
            shortest = run.size
            t = tList[i]
            # correct in case the run before is shorter
            if i !=0:
                if runList[i-1].size > shortest:
                    runList[i-1] = runList[i-1][:shortest]
        # shorten run if it is longer than 'shortest'    
        else:
            runList[i] = runList[i][:shortest]
            t = tList[i][:shortest]
    # loop to make a matrix
    for i, row in enumerate(runList):
        if i == 0:
            runMat = np.array(row)
        else:
            runMat = np.vstack([runMat, row])
    # create empty array if no runs
    if not runList:
        runMat = np.array([])
        t = np.empty(0)
    return runMat,t

def plotAllSubs(x,runMat,title=''):
    """
    """
    # import
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    if runMat.size == 0:
        return(print('not enough data for',title))
    
    elif runMat.shape == (len(runMat),):
        fig, axes = plt.subplots(1, 1, figsize=(15, 3))
        t = np.arange(0,x.size/600,1/600)
        axes.plot(t,runMat,label='sub1')
        titleStr = 'single subject rating for ' + title
        axes.set_title(titleStr)
        axes.set_ylabel('Rating\n [Z-score]')
        axes.set_xlabel('time [min]')
        axes.legend()
        axes.legend(bbox_to_anchor=(1.05, 1),loc='upper left', borderaxespad=0.,ncol=3)
        
        output_dir = os.path.join(os.getcwd(),"processed")
        plt.savefig(output_dir+"\\"+title+'_oneSub.jpg')
    else:
        fig, axes = plt.subplots(1, 1, figsize=(15, 3))
        t = np.arange(0,x.size/600,1/600) # x axis will be in minutes, otherwise use x when plotting to see in seconds
        axes.plot(t,runMat.mean(0),'k',label='mean')
        for i,run in enumerate(runMat):
            sub = 'sub'+str(i+1)
            axes.plot(t,run,label=sub,alpha=0.25) 
        titleStr = 'all ratings and group mean for ' + title
        axes.set_title(titleStr)
        axes.set_ylabel('Rating\n [Z-score]')
        axes.set_xlabel('time [min]')
        axes.legend()
        axes.legend(bbox_to_anchor=(1.05, 1),loc='upper left', borderaxespad=0.,ncol=3)
        plt.tight_layout()
        
        output_dir = os.path.join(os.getcwd(),"processed")
        plt.savefig(output_dir+"\\"+title+'_allSubs.jpg')

def runDataFrame(runsX,title):
    """
    """
    # import 
    import pandas as pd
    import numpy as np
    if runsX.size < 2 or runsX.size == len(runsX):
        print(title,'cannot create data frame if less than 2 subs')
        return []
    else:
        # initialization, create dictionary with keys = subs and values are runs
        data = {}
        for i,run in enumerate(runsX):
            key = 'sub'+str(i+1)
            data[key] = run
        # create dataframe
        df = pd.DataFrame(data)
    return df

def plotCorrMat(df,title,output_dir):
    """
    """
    # import 
    import os
    import pandas as pd
    import seaborn as sns
    import numpy as np
    import matplotlib.pyplot as plt
    m = df.corr()
    mask_ut=np.triu(np.ones(m.shape)).astype(bool)
    # plot
    fig, axes = plt.subplots(1, 1, figsize=(9, 7))
    titleStr = title + ' all participants correlation matrix'
    axes.set_title(titleStr)
    sns.heatmap(m,annot=True,cmap=sns.color_palette("viridis", as_cmap=True),
                fmt=".1f",mask=mask_ut,vmin=-1, vmax=1) # use mask=mask_ut for lower t
    plt.tight_layout()
    output_dir = os.path.join(os.getcwd(),"processed")
    plt.savefig(output_dir+"\\"+title+"_CorrMat.jpg")
    return m

def makePredictorCSV(x,y,runTitle):
    """
    writes a CSV predictor
    """
    import os
    import pandas as pd 
    import numpy as np
    runData = pd.DataFrame(x)
    duration = np.full(len(x), 0.1) # adding a column of sample rate
     # creating a 'mean' column
    runData = runData.assign(duration=duration)
    runData = runData.assign(mean=y)
    runData.columns =['onset', 'duration','mean']
    # outputs
    output_dir = os.path.join(os.getcwd(),"predictors")
    try:    
        os.makedirs(output_dir)
    except:
        pass
    filename = output_dir+ r"\\" +runTitle + '.tsv'
    runData.to_csv(filename,sep='\t',index=False)
    return runData

def makePredictorCSV_var(x,y,runTitle):
    """
    writes a CSV predictor
    """
    import os
    import pandas as pd 
    import numpy as np
    runData = pd.DataFrame(x)
    duration = np.full(len(x), 0.1) # adding a column of sample rate
     # creating a 'mean' column
    runData = runData.assign(duration=duration)
    runData = runData.assign(var=y)
    runData.columns =['onset', 'duration','var']
    # outputs
    output_dir = os.path.join(os.getcwd(),"predictors")
    try:    
        os.makedirs(output_dir)
    except:
        pass
    filename = output_dir+ r"\\" +'var_'+runTitle + '.tsv'
    runData.to_csv(filename,sep='\t',index=False)
    return runData

def dropOutliers(df,subStr_list):
    """
    """ 
    import numpy as np
    # loop to remove outliers
    for sub in subStr_list:
        df.drop(sub, inplace=True, axis=1)
    
    # recalculate group mean
    df = df.assign(mean=np.mean(df,1))
    return df

def plotGroupRatings(data,t,title):
    """
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    x = np.arange(0,t.size/600,1/600)
    fig, axes = plt.subplots(1, 1, figsize=(15, 3))
    for col in data.columns[:]:
        if col == 'mean':
            axes.plot(x,data[col],'k',label=col)
        else:
            axes.plot(x,data[col],label=col,alpha=0.25)  

    titleStr = 'Outliers Removed: all ratings and group mean for ' + title 
    axes.set_title(titleStr)
    axes.set_ylabel('Rating\n [Z-score]')
    axes.set_xlabel('time [min]')
    axes.legend()
    axes.legend(bbox_to_anchor=(1.05, 1),loc='upper left', borderaxespad=0.,ncol=3)
    plt.tight_layout()
    # save new plot
    output_dir = os.path.join(os.getcwd(),"processed")
    plt.savefig(output_dir+"\\outliers_removed_"+title+'_allSubs.jpg')

def plotMeanRatingWithSTD(data,t,title):
    """
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    meanR = np.mean(data,axis=1)
    stdR =  np.std(data,axis=1)
    x = np.arange(0,t.size/600,1/600)
    # plot
    fig, ax = plt.subplots(1, figsize=(15, 4))
    ax.plot(x, meanR, lw=2, label='mean rating', color='black')
    ax.fill_between(x, meanR+stdR, meanR-stdR, facecolor='gray', alpha=0.5,label='STD')
    ax.set_title(title + r' predictor and STD')
    ax.legend(loc='upper left')
    ax.set_xlabel('time [min]')
    ax.set_ylabel('Rating')
    ax.set_ylim((-3.5, 3.5)) # for same scaling between runs. check if plots are intact 
    ax.grid()
    plt.tight_layout()
    # save new plot
    output_dir = os.path.join(os.getcwd(),"processed")
    plt.savefig(output_dir+"\\mean_and_STD_"+title+'.jpg')
    return

def pairwise_mean_vec_from_corr_mat(correlation_matrix):
    """"
    """
    import pandas as pd
    import numpy as np
    # Check if the input is a pandas dataframe
    if not isinstance(correlation_matrix, pd.DataFrame):
        raise ValueError("Input should be a pandas dataframe")
    # Check if the matrix is square
    if correlation_matrix.shape[0] != correlation_matrix.shape[1]:
        raise ValueError("Input correlation matrix should be a square dataframe")
    # Get the size of the square matrix
    matrix_size = correlation_matrix.shape[0]
    # Calculate the row means excluding the diagonal elements
    row_means = []
    for i in range(matrix_size):
        row = correlation_matrix.iloc[i].values
        row_mean = np.mean(np.delete(row, i))  # Exclude the i-th element (diagonal)
        row_means.append(row_mean)
    return row_means

def label_volumes(arr):
    import numpy as np
    import math
    arrDownsampled = arr[:-2:20] # downsampled to TR (10 per sec, tr=2 sec)
    # Create a boolean array of the same shape as the input array
    result = np.zeros_like(arrDownsampled, dtype=int)
    # Iterate over each pair of values
    for i in range(arrDownsampled.shape[0]):
            # Check if either value is greater than 1.5
        if arrDownsampled[i] > 0:
                result[i] = 1
        else:
            result[i] = 0
    return result

def label_segments(arr):
    import numpy as np
    segment_size = len(arr) // 6
    segments = np.array_split(arr, 6)
    segment_std = np.array([np.mean(segment) for segment in segments])
    sorted_std = np.sort(segment_std[::-1])
    top_3_std = sorted_std[:3]
    result = [std in top_3_std for std in segment_std]
    return result

def preprocessOutput(input_dir,output_dir):
    """
    """
    import os

    # initialization
    csvList = [os.path.join(input_dir,file) for file in os.listdir(input_dir)]
    infoList = []
    reject = []
    
    # first movie (fg_seg3)
    gaze_run1,touch_run1,affect_run1,general_run1 = [],[],[],[] 
    xGaze_run1,xTouch_run1,xAffect_run1,xGeneral_run1 = [],[],[],[]
    # second movie (fg_seg7)
    gaze_run2,touch_run2,affect_run2,general_run2 = [],[],[],[]
    xGaze_run2,xTouch_run2,xAffect_run2,xGeneral_run2 = [],[],[],[]
    # loop - for every csv
    for csvFile in csvList:
        demographics, labels, time_raw, rating_raw = load_csv(csvFile) 
        ver = verify(demographics, labels, time_raw, rating_raw) # check if verified
        if ver:
            print(csvFile,'\nis verified\n')
            # interpolation
            x,y = timeInterpolation(time_raw, rating_raw,0.1)
            print('origin time samples =',len(time_raw),'\ninterp time samples =',x.size,'\ndifference:', x.size-len(time_raw))
            # smoothing
            ySmooth = smoothing(x,y,5)
            # normalizing rating on individual run level
            z = demeanArray(ySmooth)
            # get session info
            info = [i[1] for i in demographics]
            infoList.append(info)   
            movie = info[7] # 8th line in log is movie
            exp = info[8]   # 9th line in log is experiment
            # sort to different lists per experiment per run
            if exp == 'Gaze Synchrony':
                if movie == 'fg_av_eng_seg3.mkv':
                    gaze_run1.append(z)
                    xGaze_run1.append(x)
                elif movie == 'fg_av_eng_seg7.mkv':
                    gaze_run2.append(z)
                    xGaze_run2.append(x)    
            elif exp == 'Touch Synchrony':
                if movie == 'fg_av_eng_seg3.mkv':
                    touch_run1.append(z)
                    xTouch_run1.append(x)
                elif movie == 'fg_av_eng_seg7.mkv':
                    touch_run2.append(z)
                    xTouch_run2.append(x)
            elif exp == 'Affect Synchrony':
                if movie == 'fg_av_eng_seg3.mkv':
                    affect_run1.append(z)
                    xAffect_run1.append(x)
                elif movie == 'fg_av_eng_seg7.mkv':
                    affect_run2.append(z)
                    xAffect_run2.append(x)
            elif exp == 'General Synchrony':
                if movie == 'fg_av_eng_seg3.mkv':
                    general_run1.append(z)
                    xGeneral_run1.append(x)
                elif movie == 'fg_av_eng_seg7.mkv':
                    general_run2.append(z)
                    xGeneral_run2.append(x)              
        # if not verified
        else:
            print(csvFile,'\nis not verified!\n')
            reject.append(csvFile)
    # continue group level
    # run1
    gaze1,tGaze1 = makeSameSize(gaze_run1,xGaze_run1)
    touch1,tTouch1= makeSameSize(touch_run1,xTouch_run1)
    affect1,tAffect1 = makeSameSize(affect_run1,xAffect_run1)
    general1,tGeneral1 = makeSameSize(general_run1,xGeneral_run1)
    # run 2
    gaze2,tGaze2 = makeSameSize(gaze_run2,xGaze_run2)
    touch2,tTouch2 = makeSameSize(touch_run2,xTouch_run2)
    affect2,tAffect2 = makeSameSize(affect_run2,xAffect_run2)
    general2,tGeneral2 = makeSameSize(general_run2,xGeneral_run2)
    # a list of all runs
    allRunList = [gaze1,gaze2,touch1,touch2,affect1,affect2,general1,general2]
    allTimeList = [tGaze1,tGaze2,tTouch1,tTouch2,tAffect1,tAffect2,tGeneral1,tGeneral2]
    allTitles = ['Gaze_run-1','Gaze_run-2','Touch_run-1','Touch_run-2',
                 'Affect_run-1','Affect_run-2','General_run-1','General_run-2']
    # plot all runs
    for i,runs in enumerate(allRunList):
        plotAllSubs(allTimeList[i],runs,allTitles[i])   
    return allRunList,allTimeList,infoList,allTitles,reject

def calculateGrid(subNumByConds):
    """
    takes in a vector representing the number of subjects in each condition
    by the following order (index):
        0 = gaze
        1 = touch
        2 = affect
        3 = general
    an returns the gridlines to be displayed when plotting the RDM of a specific run
    """
    # take in gaze number and raise it by 0.5
    gridList = []
    a = -0.5 
    for subs in subNumByConds:
        a+= subs
        gridList.append(a)
    return gridList[:-1]

def generate_random_rdm(n):
    import random
    import numpy as np
    matrix = np.zeros((n, n))  # Initialize matrix with zeros
    vector = []
    for i in range(n):
        for j in range(i + 1, n):  # Iterate over upper triangle elements
            value = random.uniform(0, 2)  # Generate a random value between 0 and 2
            matrix[i][j] = value  # Assign value to current element
            matrix[j][i] = value  # Assign value to symmetric element
            vector.append(matrix[i][j]) 
    return matrix, np.array(vector)

def BasicModelRDM(rdm):
    import rsatoolbox
    # get rdm matrix (shape NxN)
    n = rdm.get_matrices().shape[0]
    random_rdm = generate_random_rdm(n)
    return random_rdm

def behaveRSA(x):
    """
    """
    import os
    import rsatoolbox
    import matplotlib.pyplot as plt
    import numpy as np
    import rsatoolbox.data as rsd 
    output_dir = os.path.join(os.getcwd(),"processed")
    # initialize dataset dimensions (number of participants per condition)
    # create a vector for each run, representing the number of subjects for each condition
    run1_subNumVec=[x[0].shape[0],x[2].shape[0],x[4].shape[0],x[6].shape[0]]
    run2_subNumVec=[x[1].shape[0],x[3].shape[0],x[5].shape[0],x[7].shape[0]]
    # concatanate all runs to create the dataset:
    compDataset1 = np.concatenate((x[0],x[2],x[4],x[6])) # all conditions, run1.
    compDataset2 = np.concatenate((x[1],x[3],x[5],x[7])) # all conditions, run2.
    
    # create rating RDMs for both runs
    ################### run-1 ####################
    measurements1 = compDataset1
    nTrials1 = measurements1.shape[0]
    nTP1 = measurements1.shape[1]
    des1 = {'session': 1, 'subj': 1}
    obs_des1 = {'trials1': np.array(['Ga_' + str(x+1) for x in np.arange(run1_subNumVec[0])]+
                                    ['To_' + str(x+1) for x in np.arange(run1_subNumVec[1])]+
                                    ['Af_' + str(x+1) for x in np.arange(run1_subNumVec[2])]+
                                    ['Ge_' + str(x+1) for x in np.arange(run1_subNumVec[3])])}
    chn_des1 = {'TP': np.array(['TP_' + str(x) for x in np.arange(nTP1)])}
    data1 = rsd.Dataset(measurements=measurements1,
                               descriptors=des1,
                               obs_descriptors=obs_des1,
                               channel_descriptors=chn_des1)
    # calculate RDMs
    rdms1 = rsatoolbox.rdm.calc_rdm(data1, method = 'correlation')
    # visualize RDMs
    fig1, ax, ret_val = rsatoolbox.vis.show_rdm(rdms1, 
                                                rdm_descriptor='run-1', 
                                                figsize = (20,20),
                                                cmap = 'viridis_r',
                                                gridlines = calculateGrid(run1_subNumVec),
                                                vmin = 0, vmax = 2,
                                                show_colorbar='panel',pattern_descriptor='trials1')
    
    # save figure
    plt.savefig(output_dir+"\\RDM_run-1.png",dpi=900)

    ################### run-2 ####################
    measurements2 = compDataset2
    nTrials2 = measurements2.shape[0]
    nTP2 = measurements2.shape[1]
    des2 = {'session': 1, 'subj': 1}
    obs_des2 = {'trials2': np.array(['Ga_' + str(x+1) for x in np.arange(run2_subNumVec[0])]+
                                    ['To_' + str(x+1) for x in np.arange(run2_subNumVec[1])]+
                                    ['Af_' + str(x+1) for x in np.arange(run2_subNumVec[2])]+
                                    ['Ge_' + str(x+1) for x in np.arange(run2_subNumVec[3])])}

    chn_des2 = {'TP': np.array(['TP_' + str(x) for x in np.arange(nTP2)])}
    data2 = rsd.Dataset(measurements=measurements2,
                        descriptors=des2,
                        obs_descriptors=obs_des2,
                        channel_descriptors=chn_des2)
    # calculate RDMs
    rdms2 = rsatoolbox.rdm.calc_rdm(data2, method = 'correlation')
    # visualize RDMs
    fig2, ax, ret_val = rsatoolbox.vis.show_rdm(rdms2, 
                                                rdm_descriptor='run-2', 
                                                figsize=(20,20),
                                                cmap = 'viridis_r',
                                                gridlines=calculateGrid(run2_subNumVec),
                                                vmin = 0, vmax = 2,
                                                show_colorbar='panel',pattern_descriptor='trials2')
    # save figure
    plt.savefig(output_dir+"\\RDM_run-2.png",dpi=900)
    
    # create model RDMs for both runs, based on hypothetical assumptions
    ############# run-1 #######################
    
    # get matrix of run 1
    # mat = rdm1.get_matrices()
    
    return rdms1,rdms2

def createPredictorLabels(data,t,title):
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    x = np.arange(0,t.size/600,1/600)
    meanR = demeanArray(np.array(data['mean'])) # DEMEANED again for labeling
    stdR = np.array(data['STD'])
    volume_labels = label_volumes(meanR)
    # plot prediction by volume
    # create scatter groups
    scatters = []
    for i in volume_labels:
        if i == 1:
            scatters.append(2)
        else:
            scatters.append(-2)   
    # plot run and segments
    fig, ax = plt.subplots(1, figsize=(20, 4))
    ax.plot(x, meanR, lw=2, label='mean rating', color='black')
    ax.set_title(title + r' predictor and label per volume')
    ax.scatter(x[:-2:20],scatters, alpha = 0.5,c = 'm', label = 'labels')
    ax.legend(loc = 'lower left')
    ax.set_xlabel('time [min]')
    ax.set_ylabel('Rating')
    ax.set_ylim((-3.5, 3.5)) # for same scaling between runs. check if plots are intact 
    plt.tight_layout()
    # save new plot
    output_dir = os.path.join(os.getcwd(),"processed")
    plt.savefig(output_dir+"\\prediction of_"+title+'_by_volume.jpg')
    # create labels for 6 segments
    segment_labels = label_segments(meanR)
    xcoords = np.linspace(0, x[-1], 7)
    tcoords = np.linspace(0, t[-1], 7)
    # plot run and segments
    fig, ax = plt.subplots(1, figsize=(15, 4))
    ax.plot(x, meanR, lw=2, label='mean rating', color='black')
    ax.set_title(title + r' predictor and label per segment')
    ax.set_xlabel('time [min]')
    ax.set_ylabel('Rating')
    ax.set_ylim((-3.5, 3.5)) # for same scaling between runs. check if plots are intact 
    plt.tight_layout()
    for xc in xcoords:
        ax.axvline(xc, color = 'm')
    #ax.fill_between(x, meanR+stdR, meanR-stdR, facecolor='gray', alpha=0.5,label='STD')
    return volume_labels, segment_labels

