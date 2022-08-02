"""
@author: kaustubhprabhu
"""
import pandas as pd
import numpy as np
from os import listdir
from multiprocessing import Pool
from subprocess import call
import sys
import math

tthreshold=300 # threshold: 60 = 60 sec, one sample can only represent 1 mins stay
z=0
epochtime = 1420372800 # 2015/01/03  12:00 am, used to calculate hour of the day quickly
eps = 0.0001#:0.0001:0.001
minpts = 5 #2:1:30
#Read data - for all users one by one
re=[]
#for zz = 1:size(gpsData,2) # Last column has users id
kmp = 1
if (kmp == 1  ): #for all files

        tmpdata = pd.read_csv('Users/kaustubhprabhu/Documents/GitHub/Opioid-LifeRhythm/dbscan/peopleData/p2/189538_p2.csv')
        tmpdata.iloc[:,1] = tmpdata.iloc[:,1]/1000
        tmpdata.insert(4,'k', np.floor((tmpdata.iloc[:,1]-epochtime)/86400))
        wd = np.mod(tmpdata.iloc[:,4],7)+1
        tmpdata.insert(5,'l', wd)
        days = tmpdata.iloc[:,4].nunique()

        th=12 # %12km/hour
        lenth = len(tmpdata)
        tmpdata["m"] = np.nan
        for l in range(2,lenth):
            fP1Lat = tmpdata.iloc[l-1,2]
            fP1Lon = tmpdata.iloc[l-1,3]
            fP2Lat = tmpdata.iloc[l,2]
            fP2Lon = tmpdata.iloc[l,3]
            fRadLon1 = (np.pi/180)*(fP1Lon)
            fRadLon2 = (np.pi/180)*(fP2Lon)
            fRadLat1 = (np.pi/180)*(fP1Lat)
            fRadLat2 = (np.pi/180)*(fP2Lat)
            fD1 = abs(fRadLat1 - fRadLat2)
            fD2 = abs(fRadLon1 - fRadLon2)
            fP = (math.sin(fD1/2)**2) + math.cos(fRadLat1) * math.cos(fRadLat2) * (math.sin(fD2/2)**2)
            d = 6371.137 * 2 * math.asin(math.sqrt(fP))
            t = (tmpdata.iloc[l,1]-tmpdata.iloc[l-1,1])
            if (d*3600/t >th):
                tmpdata.iloc[l,6]=1
                tmpdata.iloc[l-1,6]=1

        for day in range(1,len(days)):
                z=z+1
                userData = tmpdata[find(tmpdata.iloc[:,4]==days(day)),:]
                if (len(userData) == 0): #% if no data found, ignoremove one
                    continue

            #in case the time is in microsecond
                periodts = userData[1,2];

                gpsFeatures[z,13] = userData[1,1]#; % uid
                gpsFeatures[z,1] = userData[1,2]#; % start time
                userData = [userData[:,2], userData[:,3], userData[:,4],  userData[:,6], userData[:,7]] #% [timestamp, lat, long, weekday, moving state]
            #%% Time spent calculation (preprocessing step)
            #% Calculate how much time user spent on a perticular location
            #% Traverse through all location samples
                timeSpent = [] #% clear timeSpent
                timeSpent[1,1] = tthreshold
                for i in range( 2,userData.size[1]):
                    latDiff =abs(userData[i-1,2] - userData[i,2])
                    longDiff = abs(userData[i-1,3] - userData[i,3])
                    timeSpent[i,1] = userData[i,1] - userData[i-1,1]
                    if timeSpent[i,1] < 0 :
                        print('something is fishy, see code line 59')
                        pause(5)# % pause, in case timespent seems incorrect
                    print('something is fishy, see code line 59')
                    if timeSpent[i,1] > tthreshold: #% missing data case
                        timeSpent[i,1] = tthreshold# % android: 10 minutes  ios: 1 minute
            #% end for - this will give us time spent b/w consecutive long/lat
                userData.insert('p',timeSpent)# % add as a column % [timestamp, lat, long, weekday, mins_between_traces]

                # %% Feature 1 = Variance
                # % First feature - Location variance - that measures the variability in the
                # % Calculate statistical variance of longitude and latitude
                varLat = np.var(userData.iloc[:,2])
                varLong = np.var(userData.iloc[:,3])
                locVar = math.log(varLat + varLong)#; % Nautral Log
                gpsFeatures[z,2] = locVar
                ''' #%% Feature 2 = Entropy
                #% Entropy - to measure the variability of the time the subject spent at a
                #% location cluster
                # % Clustering used = dbScan
                [class,locationType]= dbscan([userData(:,2:3)], minpts, eps); % Apply dbscan without time dimension
                userData(:,end+1) = class#'; % Cluster label  % [timestamp, lat, long, moving/stat, weekday mins_between_traces cluster#]
                eind1 = size(userData,2); % This index have cluster number (-1 = outlier)
            userData(:,end+1) = type#'; % Boundry or outlier etc % [timestamp, lat, long, moving/stat, weekday mins_between_traces cluster# type]
            eind2 = size(userData,2); % This index have cluster type info
            % Remove noise/outlier points i.e., type = -1
            indClus = find(userData(:,eind1) ~= -1); % -1 Out - ASMA UNCOMMENTED
            userData = userData(indClus,:);
            if (length(userData) == 0) % if no data found, ignore and move one
                continue
            end

                #% Calculate Entropy
                ent = 0
                t = [zeros(len(userData[:,1]),3) mod((userData[:,1]*1000-epochtime)/3600000,24)]
                col = userData.size[2]
                userData(:,end+1) = t(:,4)

            # %% Feature 4 = Number of Unique Clusters
            #% Number of unique clusters
            uniClus = unique(userData(:,eind1))#; % Cluster number % ASMA CHANGED
            I = size(uniClus,1)#; % Number of unique clusters
            gpsFeatures(z,10) = I; #% outlier already removed from cluster

            if I > 1 % if atleast 2 clusters found
                maxa = 0;
                    f = 1; % Variable that finally will have home cluster information
                    nightcluster{z}=[];
                    for i = 1:I % total time spend in the cluster
                        % if uniClus(i) > -1
                        tmesptind = find( userData(:,eind1) == uniClus(i)); % find the cluster index
                        totTimeClus = sum(userData(:, 6)); % total time spend @ "all" meaningful clusters
                        tsumTime(1,i) = (sum(userData(tmesptind, 6))); % total time spent in a perticular cluster i.e., ith cluster
                        sumTime(1,i) = (sum(userData(tmesptind, 6)))/totTimeClus; % Percentage of time spent @ cluster i
                        ent = ent + ( sumTime(1,i) * (log(sumTime(1,i)))); % calculate entropy

                        if uniClus(i) ~= -1
                            idTime = find(userData(tmesptind,end) >= 0 & userData(tmesptind,end) < 6); % Time between 0 and 6
                            idset = userData(tmesptind(idTime), 6);
                            if sum(userData(tmesptind,6))/sum(userData(:,6)) > 0.1
                                hometime = sum(idset);

                                if hometime > maxa
                                    maxa = hometime;
                                    f = uniClus(i); % By the end of iterations, f will have the cluster marked as home
                                end
                            end
                        else
                            gpsFeatures(z,16) = tmesptind/length(userData(:,1));
                        end

                    end
                    %% Feature 2 = Entropy
                    ent = 0;
                    es = sort(sumTime(1,:),1);
                    for nes =1:length(es(1,:))
                       ent = ent + ( es(1,nes) * (log(es(1,nes))));
                    end
                    ent = -ent;

                    gpsFeatures(z,5) = ent;
                    %% Feature 3 = Normalized Entropy
                    gpsFeatures(z,6) = gpsFeatures(z,5)/(log(I));  % I is the total number of clusters

                    %% Feature 5 = Home stay - cluster which have the time between 12am to 6am
                    if uniClus(1) == -1 % Never happens, just in case
                      gpsFeatures(z,7) = (sumTime(1,f+1));%/sum(sumTime); % WE DONT NEED TO SUM IT
                    else
                      gpsFeatures(z,7) = (sumTime(1,f));%/sum(sumTime);
                    end
                else
                    gpsFeatures(z,5) = 0;
                    gpsFeatures(z,6) = 0;
                    gpsFeatures(z,7) = 1;
                end % End if - we are done with clustering


                '''
                '''#%% Feature 6 = Transition time
                #% Transition time = number of samples in transition/ total number of
                #% samples
                movingSamples = sum(userData(find(userData(:,5) == 1),6))
                AllSamples = sum(userData(:,6));
                gpsFeatures(z,8) = movingSamples/ AllSamples;
                %% Feature 7 = Total Distance
                % Total distance traveled in km
                sumdistkm = 0;
                for i = 1:(size(userData,1)-1)
                    latLong1 = [userData(i,2) userData(i,3)];
                    latLong2 = [userData(i+1,2) userData(i+1,3)];
                    [d1km d2km]=lldistkm(latLong1,latLong2);
                     sumdistkm = sumdistkm + d1km;
                end
                gpsFeatures(z,9) = sumdistkm; %unit: km
                %% Feaure 8 = Average Moving Speed
                gpsFeatures(z,3) = sumdistkm/((userData(end,1)-userData(1,1)));  %average moving speed  unit:km/s
                clearvars -except days tmpdata epochtime nday gpsonly os interval starttime eps minpts c tthreshold z gpsData gpsFeatures re %tsumTime centers sumTime userData timeSpent latDiff longDiff ind varLat varLong locVar LongLatStat idx C clususerData clus1Ind clus2Ind clus3Ind timeClus1 timeClus2 timeClus3 i gpsTime homeTime t sumdistkm movingSamples col ent latLong1 latLong2 d1km d2km AllSamples
        end
    end
    end
end
'''

'''
tthreshold=60; % threshold: 60 = 60 sec, one sample can only represent 1 mins stay
z = 0
epochtime = 1420372800 % 2015/01/03  12:00 am, used to calculate hour of the day quickly
for eps = 0.0001%:0.0001:0.001
    for minpts = 5 %2:1:30
    %%
    % Read data - for all users one by one
    re=[];
    for zz = 1:size(gpsData,2) % Last column has users id
        tmpdata = gpsData{zz};
        tmpdata(:,2) = tmpdata(:,2)/1000;
        tmpdata(:,5) = floor((tmpdata(:,2) - epochtime)/86400);
        wd = mod(tmpdata(:,5),7)+1;
        tmpdata(:,6) = wd;
        days = unique(tmpdata(:,5))

        th=1; %1km/hour
        len = length(tmpdata);
        for l = 2:len
            fP1Lat = tmpdata(l-1,3);
            fP1Lon = tmpdata(l-1,4);
            fP2Lat = tmpdata(l,3);
            fP2Lon = tmpdata(l,4);
            fRadLon1 = (pi/180).*(fP1Lon);
            fRadLon2 = (pi/180).*(fP2Lon);
            fRadLat1 = (pi/180).*(fP1Lat);
            fRadLat2 = (pi/180).*(fP2Lat);
            fD1 = abs(fRadLat1 - fRadLat2);
            fD2 = abs(fRadLon1 - fRadLon2);
            fP = power (sin(fD1/2), 2) + cos(fRadLat1) * cos(fRadLat2) * power (sin(fD2/2), 2);
            d = 6371.137 * 2 * asin(sqrt(fP)) ;
            t = (tmpdata(l,2)-tmpdata(l-1,2));
            if (d*3600/t >th)
                tmpdata(l,7)=1;
                tmpdata(l-1,7)=1;
            end
        end


    for day = 1:length(days)
            z=z+1
            userData = tmpdata(find(tmpdata(:,5)==days(day)),:);
            if (length(userData) == 0) % if no data found, ignore and move one
                continue;
            end
             % in case the time is in microsecond
            periodts = userData(1,2);

            gpsFeatures(z,13) = userData(1,1); % uid
            gpsFeatures(z,1) = userData(1,2); % start time
            userData = [userData(:,2) userData(:,3) userData(:,4)  userData(:,6) userData(:,7)]; % [timestamp, lat, long, weekday, moving state]
            %% Time spent calculation (preprocessing step)
            % Calculate how much time user spent on a perticular location
            % Traverse through all location samples
            timeSpent = []; % clear timeSpent
            timeSpent(1,1) = tthreshold;
            for i = 2:size(userData,1)
                latDiff =abs(userData(i-1,2) - userData(i,2));
                longDiff = abs(userData(i-1,3) - userData(i,3));
                timeSpent(i,1) = userData(i,1) - userData(i-1,1) ;
                if timeSpent(i,1) < 0
                    disp('something is fishy, see code line 59');
                    pause(5); % pause, in case timespent seems incorrect
                end
                if timeSpent(i,1) > tthreshold % missing data case
                    timeSpent(i,1) = tthreshold; % android: 10 minutes  ios: 1 minute
                end
            end % end for - this will give us time spent b/w consecutive long/lat

            userData(:, end+1) = timeSpent; % add as a column % [timestamp, lat, long, weekday, mins_between_traces]

            %% Feature 1 = Variance
            % First feature - Location variance - that measures the variability in the
            % Calculate statistical variance of longitude and latitude
            varLat = var(userData(:,2));
            varLong = var(userData(:,3));
            locVar = log(varLat + varLong); % Nautral Log
            gpsFeatures(z,2) = locVar;
            %% Feature 2 = Entropy
            % Entropy - to measure the variability of the time the subject spent at a
            % location cluster
            % Clustering used = dbScan
            [class,type]= dbscan([userData(:,2:3)], minpts, eps); % Apply dbscan without time dimension
            userData(:,end+1) = class'; % Cluster label  % [timestamp, lat, long, moving/stat, weekday mins_between_traces cluster#]
            eind1 = size(userData,2); % This index have cluster number (-1 = outlier)
            userData(:,end+1) = type'; % Boundry or outlier etc % [timestamp, lat, long, moving/stat, weekday mins_between_traces cluster# type]
            eind2 = size(userData,2); % This index have cluster type info
            % Remove noise/outlier points i.e., type = -1
            indClus = find(userData(:,eind1) ~= -1); % -1 Out - ASMA UNCOMMENTED
            userData = userData(indClus,:);
            if (length(userData) == 0) % if no data found, ignore and move one
                continue;
            end
            % Calculate Entropy
            ent = 0;
            t = [zeros(length(userData(:,1)),3) mod((userData(:,1)*1000-epochtime)/3600000,24)];
            col = size(userData,2);
            userData(:,end+1) = t(:,4);

            %% Feature 4 = Number of Unique Clusters
            % Number of unique clusters
            uniClus = unique(userData(:,eind1)); % Cluster number % ASMA CHANGED
            I = size(uniClus,1); % Number of unique clusters
            gpsFeatures(z,10) = I; % outlier already removed from cluster

            if I > 1 % if atleast 2 clusters found
                maxa = 0;
                    f = 1; % Variable that finally will have home cluster information
                    nightcluster{z}=[];
                    for i = 1:I % total time spend in the cluster
                        % if uniClus(i) > -1
                        tmesptind = find( userData(:,eind1) == uniClus(i)); % find the cluster index
                        totTimeClus = sum(userData(:, 6)); % total time spend @ "all" meaningful clusters
                        tsumTime(1,i) = (sum(userData(tmesptind, 6))); % total time spent in a perticular cluster i.e., ith cluster
                        sumTime(1,i) = (sum(userData(tmesptind, 6)))/totTimeClus; % Percentage of time spent @ cluster i
                        ent = ent + ( sumTime(1,i) * (log(sumTime(1,i)))); % calculate entropy

                        if uniClus(i) ~= -1
                            idTime = find(userData(tmesptind,end) >= 0 & userData(tmesptind,end) < 6); % Time between 0 and 6
                            idset = userData(tmesptind(idTime), 6);
                            if sum(userData(tmesptind,6))/sum(userData(:,6)) > 0.1
                                hometime = sum(idset);

                                if hometime > maxa
                                    maxa = hometime;
                                    f = uniClus(i); % By the end of iterations, f will have the cluster marked as home
                                end
                            end
                        else
                            gpsFeatures(z,16) = tmesptind/length(userData(:,1));
                        end

                    end
                    %% Feature 2 = Entropy
                    ent = 0;
                    es = sort(sumTime(1,:),1);
                    for nes =1:length(es(1,:))
                       ent = ent + ( es(1,nes) * (log(es(1,nes))));
                    end
                    ent = -ent;

                    gpsFeatures(z,5) = ent;
                    %% Feature 3 = Normalized Entropy
                    gpsFeatures(z,6) = gpsFeatures(z,5)/(log(I));  % I is the total number of clusters

                    %% Feature 5 = Home stay - cluster which have the time between 12am to 6am
                    if uniClus(1) == -1 % Never happens, just in case
                      gpsFeatures(z,7) = (sumTime(1,f+1));%/sum(sumTime); % WE DONT NEED TO SUM IT
                    else
                      gpsFeatures(z,7) = (sumTime(1,f));%/sum(sumTime);
                    end
                else
                    gpsFeatures(z,5) = 0;
                    gpsFeatures(z,6) = 0;
                    gpsFeatures(z,7) = 1;
                end % End if - we are done with clustering



                %% Feature 6 = Transition time
                % Transition time = number of samples in transition/ total number of
                % samples
                movingSamples = sum(userData(find(userData(:,5) == 1),6));
                AllSamples = sum(userData(:,6));
                gpsFeatures(z,8) = movingSamples/ AllSamples;
                %% Feature 7 = Total Distance
                % Total distance traveled in km
                sumdistkm = 0;
                for i = 1:(size(userData,1)-1)
                    latLong1 = [userData(i,2) userData(i,3)];
                    latLong2 = [userData(i+1,2) userData(i+1,3)];
                    [d1km d2km]=lldistkm(latLong1,latLong2);
                     sumdistkm = sumdistkm + d1km;
                end
                gpsFeatures(z,9) = sumdistkm; %unit: km
                %% Feaure 8 = Average Moving Speed
                gpsFeatures(z,3) = sumdistkm/((userData(end,1)-userData(1,1)));  %average moving speed  unit:km/s
                clearvars -except days tmpdata epochtime nday gpsonly os interval starttime eps minpts c tthreshold z gpsData gpsFeatures re %tsumTime centers sumTime userData timeSpent latDiff longDiff ind varLat varLong locVar LongLatStat idx C clususerData clus1Ind clus2Ind clus3Ind timeClus1 timeClus2 timeClus3 i gpsTime homeTime t sumdistkm movingSamples col ent latLong1 latLong2 d1km d2km AllSamples
        end
    end
    end
end

gpsFeatures = gpsFeatures(~any(isnan(gpsFeatures),2),:);
gpsFeatures( ~any(gpsFeatures,2), : ) = [];
gpsFeatures( find(isnan(gpsFeatures(:,5))), : ) = [];

% var, Average Moving Speed, ent, logent, Transition time, Total Distance, Number of Unique Clusters
dSet = gpsFeatures(:,[2,3,5,6,7,8,9,10,13,1]);
dSet = dSet(~any(isnan(dSet),2),:);
dSet( find(dSet(:,2)==0),:) = [];
dSet( ~any(dSet,2), : ) = [];  % Remove any row that have a Null value

save(['dSet_' os num2str(time_d) '.mat'],'dSet');

'''

