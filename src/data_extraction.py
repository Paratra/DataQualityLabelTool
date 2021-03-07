#!/usr/bin/env python3

from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
from scipy import fft
from sklearn import preprocessing

from scipy.signal import butter, lfilter
from statsmodels.tsa.stattools import acf
from numpy import array

from matplotlib.backends.backend_pdf import PdfPages

import csv
import matplotlib.pyplot as plt
import pywt
from influxdb import InfluxDBClient
import operator
import scipy.signal as sg
import scipy as sp
import pytz
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")


def calculate_derivatives(x, n=1):
    result = []
    for j in range(n):
        for ind in range(x.shape[0]):
            if ind == x.shape[0]-1:
                result.append(0)
            else:
                result.append(x[ind+1] - x[ind])

        x = np.asarray(result)
        result = []

    return np.asarray(x)

def peak_reserve(data_arr, window_size):
    result_arr = data_arr.copy()
    # peaks_ind = []
    for ind in range(result_arr.shape[0]):
        if ind+window_size <= data_arr.shape[0]:
            max_ind = np.where(result_arr[ind:ind+window_size]==max(result_arr[ind:ind+window_size]))
            ind_arr = np.asarray(list(range(ind,ind+window_size)))
        else:
            max_ind = np.where(result_arr[ind:]==max(result_arr[ind:]))
            ind_arr = np.asarray(list(range(ind,result_arr.shape[0])))

        if max(result_arr[ind:ind+window_size]) != 0:

            # peaks_ind.append(ind_arr[max_ind])
            # print(ind_arr)
            # import pdb; pdb.set_trace()
            remain_ind = np.delete(ind_arr, max_ind)
            result_arr[remain_ind] = 0
        # import pdb; pdb.set_trace()
    return result_arr#, np.asarray(peaks_ind)



def detect_peaks(x, mph=None, mpd=1, threshold=0, edge='rising',
                 kpsh=False, valley=False, show=False, ax=None):

    x = np.atleast_1d(x).astype('float64')
    if x.size < 3:
        return np.array([], dtype=int)
    if valley:
        x = -x
        if mph is not None:
            mph = -mph
    # find indices of all peaks
    dx = x[1:] - x[:-1]
    # handle NaN's
    indnan = np.where(np.isnan(x))[0]
    if indnan.size:
        x[indnan] = np.inf
        dx[np.where(np.isnan(dx))[0]] = np.inf
    ine, ire, ife = np.array([[], [], []], dtype=int)
    if not edge:
        ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
    else:
        if edge.lower() in ['rising', 'both']:
            ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
        if edge.lower() in ['falling', 'both']:
            ife = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0))[0]
    ind = np.unique(np.hstack((ine, ire, ife)))
    # handle NaN's
    if ind.size and indnan.size:
        # NaN's and values close to NaN's cannot be peaks
        ind = ind[np.in1d(ind, np.unique(np.hstack((indnan, indnan-1, indnan+1))), invert=True)]
    # first and last values of x cannot be peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size-1:
        ind = ind[:-1]
    # remove peaks < minimum peak height
    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]
    # remove peaks - neighbors < threshold
    if ind.size and threshold > 0:
        dx = np.min(np.vstack([x[ind]-x[ind-1], x[ind]-x[ind+1]]), axis=0)
        ind = np.delete(ind, np.where(dx < threshold)[0])
    # detect small peaks closer than minimum peak distance
    if ind.size and mpd > 1:
        ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                # keep peaks with the same height if kpsh is True
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
                    & (x[ind[i]] > x[ind] if kpsh else True)
                idel[i] = 0  # Keep current peak
        # remove the small peaks and sort back the indices by their occurrence
        ind = np.sort(ind[~idel])

    if show:
        if indnan.size:
            x[indnan] = np.nan
        if valley:
            x = -x
            if mph is not None:
                mph = -mph
        _plot(x, mph, mpd, threshold, edge, valley, ax, ind)

    return ind


def checkOnBedCR(signal, onBedThreshold=10):
#       arrSignal = array( signal )
       signalFiltered = butter_bandstop_filter(signal, 17 , 22 , 100, 5)
       arrSignal = array( signalFiltered )
       auto = acf(arrSignal, unbiased=False, nlags=100, qstat=False, fft=False)

       peaks = detect_peaks(auto, show=False)
       correpk = len(peaks)
       #saveResults('corrStatus', 'bs10' ,str(correpk), time)
       if correpk <= onBedThreshold:
           return True
       else:
           return False
       # return correpk

def butter_bandstop_filter(data, lowcut, highcut, fs, order):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq

        i, u = butter(order, [low, high], btype='bandstop')
        y = lfilter(i, u, data)
        return y



# This function converts the time string to epoch time xxx.xxx (second.ms).
# Example: time = "2020-08-13T02:03:00.200", zone = "UTC" or "America/New_York"
# If time = "2020-08-13T02:03:00.200Z" in UTC time, then call timestamp = local_time_epoch(time[:-1], "UTC"), which removes 'Z' in the string end
def local_time_epoch(time, zone):
    local_tz = pytz.timezone(zone)
    try:
        localTime = datetime.strptime(time, "%Y-%m-%dT%H:%M:%S.%f")
    except:
        localTime = datetime.strptime(time, "%Y-%m-%dT%H:%M:%S")

    local_dt = local_tz.localize(localTime, is_dst=None)
    # utc_dt = local_dt.astimezone(pytz.utc)
    epoch = local_dt.timestamp()
    # print("epoch time:", epoch) # this is the epoch time in seconds, times 1000 will become epoch time in milliseconds
    # print(type(epoch)) # float
    return epoch

# This function converts the epoch time xxx.xxx (second.ms) to time string.
# Example: time = "2020-08-13T02:03:00.200", zone = "UTC" or "America/New_York"
# If time = "2020-08-13T02:03:00.200Z" in UTC time, then call timestamp = local_time_epoch(time[:-1], "UTC"), which removes 'Z' in the string end
def epoch_time_local(epoch, zone):
    local_tz = pytz.timezone(zone)
    time = datetime.fromtimestamp(epoch).astimezone(local_tz).strftime("%Y-%m-%dT%H:%M:%S.%f")
    return time

# This function converts the grafana URL time to epoch time. For exmaple, given below URL
# https://sensorweb.us:3000/grafana/d/OSjxFKvGk/caretaker-vital-signs?orgId=1&var-mac=b8:27:eb:6c:6e:22&from=1612293741993&to=1612294445244
# 1612293741993 means epoch time 1612293741.993; 1612294445244 means epoch time 1612294445.244
def grafana_time_epoch(time):
    return time/1000

def extract_data_from_influxdb(ip, port, unit, user_id, pass_word, db_name, stampIni, stampEnd):

    client = InfluxDBClient(ip, port, user_id, pass_word, db_name, ssl=True)

    # query = 'SELECT "value" FROM Z WHERE ("location" = \''+unit+'\') and time >= \''+stampIni+'\' and time <= \''+stampEnd+'\''
    query = 'SELECT "value" FROM Z WHERE ("location" = \''+unit+'\')  and time >= '+stampIni+' and time <= '+stampEnd

    result = client.query(query)

    points = list(result.get_points())
    values = map(operator.itemgetter('value'), points)
    times = map(operator.itemgetter('time'), points)
    data = np.array(list(values))
    time_point = np.array(list(times))

    fs = 100 # for vibration data

    if(len(data) == 0):

        print("No data in the chosen time range!")

        return [0]

    # print('Vib data from PiShake %s from %s to %s is fetched!!'%(unit, stampIni, stampEnd))

    return np.asarray([time_point, data]).T


def extract_range_data_from_influxdb(ip, port, unit_label, bed_unit, user_id, pass_word, db_name_label, mode, low_range, high_range):

    client = InfluxDBClient(ip, port, user_id, pass_word, db_name_label, ssl=True)

    if mode == 'sbp' or mode == 'SBP':

        # query = 'SELECT "heartrate","respiratoryrate","systolic","diastolic" FROM "caretaker4" WHERE ("bed" = \''+bed_unit+'\') and systolic >= '+low_range+' and systolic <= '+high_range+''
        query = 'SELECT "heartrate","respiratoryrate","systolic","diastolic" FROM caretaker4 WHERE "bed" = \''+bed_unit+'\' and systolic >= '+low_range+' and systolic <= '+high_range+''

    elif mode == 'dbp' or mode == 'DBP':

        query = 'SELECT "heartrate","respiratoryrate","systolic","diastolic" FROM caretaker4 WHERE "bed" = \''+bed_unit+'\' and diastolic >= '+low_range+' and diastolic <= '+high_range+''


    result = client.query(query)

    # import pdb; pdb.set_trace()

    points = list(result.get_points())

    values_hr = map(operator.itemgetter('heartrate'), points)
    values_rr = map(operator.itemgetter('respiratoryrate'), points)
    values_sbp = map(operator.itemgetter('systolic'), points)
    values_dbp = map(operator.itemgetter('diastolic'), points)
    times = map(operator.itemgetter('time'), points)

    data_hr = np.array(list(values_hr))
    data_rr = np.array(list(values_rr))
    data_sbp = np.array(list(values_sbp))
    data_dbp = np.array(list(values_dbp))

    time_point = np.array(list(times))
    time_point_list = []
    #
    if(len(data_hr) == 0):
        print("No hr data!")
        return [0]

    if(len(data_rr) == 0):
        print("No rr data!")
        return [0]

    if(len(data_sbp) == 0):
        print("No sbp data!")
        return [0]

    if(len(data_dbp) == 0):
        print("No dbp data!")
        return [0]

    print('Label data from caretaker4 %s is fetched!!'%(unit_label))

    return np.asarray([time_point, data_hr, data_rr, data_sbp, data_dbp]).T
    # return np.asarray([time_point, data_sbp, data_dbp]).T


def window_std(data, win_len):

    len_ = len(data)
    result = []
    for ind in range(len_):
        if ind < win_len-1 :
            # result.append(np.std( data[:ind+1]) )
            continue
        else:
            result.append(np.std( data[ind-win_len+1:ind+1]) )

    return result


def par(data, win_len):
    len_ = len(data)
    result = []
    for ind in range(len_):
        if ind < win_len-1 :
            # result.append(np.max( data[:ind+1]) / np.mean( data[:ind+1]) )
            continue
        else:
            result.append(np.max( data[ind-win_len+1:ind+1]) / np.mean( data[ind-win_len+1:ind+1]) )


        # if ind+n > len_ :
        #     result.append(np.mean( x[ind:]) )
        # else:
        #     result.append(np.mean( x[ind:ind+n]) )

    return result

def data_good(data):

    data = data/abs(np.max(data))
    # import pdb; pdb.set_trace()

    derivatived_data = calculate_derivatives(data,5)
    normalised_data = derivatived_data/max(abs(derivatived_data))
    normalised_data[np.where(abs(normalised_data)<0.1)] = 0

    removed_data = peak_reserve(normalised_data, window_size=55)
    index_line = np.where(removed_data>0)

    diff_sequence = np.diff(index_line).squeeze()
    diff_std = np.std(diff_sequence)

    win_std_list = window_std(data, 30)
    win_std = np.std(win_std_list)

    win_par_list = par(data, 100)
    win_par_std = np.std(win_par_list)
    # import pdb; pdb.set_trace()


    # plt.plot(data)
    # plt.vlines(index_line, min(data), max(data),'r')
    # plt.show()
    # import pdb; pdb.set_trace()
    ON_BED = checkOnBedCR(list(data), onBedThreshold=10)
    if ON_BED and (diff_std<25) and (win_std>0.02) and (win_std<0.2) and (win_par_std<2.0):
        # import pdb; pdb.set_trace()
        return True, index_line
    else:
        # print(f'ON_BED: {ON_BED}')
        # print(f'diff_std: {diff_std}')
        # print(f'win_std: {win_std}')
        # print(f'win_par_std: {win_par_std}')
        return False, index_line





MODE_RANGE = [
                # ['sbp',80,90],
                # ['sbp',100,110],
                ['sbp',115,125],
                # ['sbp',140,200],
                # ['dbp',60,70],
                # ['dbp',70,80],
                # ['dbp',80,90],
                # ['dbp',90,100],
            ]


info_list = []
for ind, mr in enumerate(MODE_RANGE):
    print(ind)
    mode = mr[0]
    low_range = mr[1]
    high_range = mr[2]


    each_data = extract_range_data_from_influxdb('sensorweb.us', '8086', 'ca:re:ta:ke:r0:00', 'b8:27:eb:23:4b:2b;b8:27:eb:80:1c:cf',
                            'test', 'sensorweb', 'caretaker4_ctru', mode,
                            str(low_range), str(high_range))

    for j, each_gt in enumerate(each_data):
        # import pdb; pdb.set_trace()
        time = local_time_epoch(each_gt[0].replace('Z',''), "utc")
        info_list.append(['b8:27:eb:23:4b:2b',int(time*1000),each_gt[3],each_gt[4]])

    # raw_data = extract_data_from_influxdb('sensorweb.us', '8086', each_time_unit[2],
    #                         'test', 'sensorweb', 'shake_ctru',
    #                         sec2time_whole(int(each_distributed_result[0].split('.')[0])-30).replace('000000Z',each_distributed_result[0].split('.')[1])+'000Z', sec2time_whole(int(each_distributed_result[0].split('.')[0])).replace('000000Z',each_distributed_result[0].split('.')[1])+'000Z')

    # import pdb; pdb.set_trace()
    np.random.shuffle(info_list)
    info_list = info_list[15130:16130]
    # info_list[0] = ['b8:27:eb:23:4b:2b', 1612297552000]



# the time string may be converted to epoch time with local_time_epoch()
# do not use "Z" in the end, because that means UTC time
# info_list = [
#             ['b8:27:eb:23:4b:2b', '2021-01-11T14:41:10.000'],  # 141.,  73.
#             ['b8:27:eb:23:4b:2b', '2020-11-24T14:49:20.000'], # 119.,  64
#             ['b8:27:eb:23:4b:2b', '2020-12-10T10:34:00.000'],  # 88., 50
#             ['b8:27:eb:23:4b:2b', '2021-02-02T15:25:52.000'],
#             ]



# the grafana time may be converted to epoch time with grafana_time_epoch()
# info_list = [
#             ['b8:27:eb:23:4b:2b', 1610394070000],  # 141.,  73.
#             ['b8:27:eb:23:4b:2b', 1606247360000], # 119.,  64
#             ['b8:27:eb:23:4b:2b', 1607614440000],  # 88., 50
#             ['b8:27:eb:23:4b:2b', 1612297552000],
#             ]




duration = 10 # seconds

plt.figure(figsize=(20, 8))

filtered_data_list = []

# pdf_not_good = PdfPages(f'../{low_range}_{high_range}_not_good.pdf')
# good_data_list = []
for ind, info in enumerate(info_list):

    ip = "sensorweb.us"
    unit = info[0] # BedJ
    # start_epoch = local_time_epoch(info[1], "America/New_York")
    start_epoch = grafana_time_epoch(info[1])

    end_epoch = start_epoch + duration # info[2]

    stampIni = str(int(start_epoch*10e8))
    stampEnd = str(int(end_epoch*10e8))
    print("Start:",  start_epoch,  stampIni, "End:", end_epoch, stampEnd)

    client = InfluxDBClient(ip, "8086", "test", "sensorweb", "shake_ctru", True)
    query = 'SELECT "value" FROM Z WHERE ("location" = \''+unit+'\')  and time >= '+stampIni+' and time <= '+stampEnd
    # query = 'SELECT "value" FROM Z WHERE ("location" = \''+unit+'\') and time >= \''+stampIni+'\' and time <= \''+stampEnd+'\''

    result = client.query(query)
    points = list(result.get_points())
    values =  map(operator.itemgetter('value'), points)
    times  =  map(operator.itemgetter('time'),  points)
    data = np.array(list(values))
    fs = 100 # for vibration data

    if(len(data) == 0):
        print("No data in the chosen time range!")
        continue

    if(len(data) != duration*100):
        continue



    # GOOD_DATA, index_line = data_good(data)
    # if not GOOD_DATA:
    #     print("Not good data")
    #     continue

    # good_data_list.append(['b8:27:eb:23:4b:2b', info[1]])

    try:

        client = InfluxDBClient(ip, "8086", "test", "sensorweb", "caretaker4_ctru", True)
        unit = "ca:re:ta:ke:r0:00"
        query = 'SELECT mean("systolic") FROM caretaker4 WHERE ("location" = \''+unit+'\') and time >= ' +stampIni+' and time <= '+stampEnd
        # query = 'SELECT mean("systolic") FROM caretaker4 WHERE ("location" = \''+unit+'\') and time >= \''+stampIni+'\' and time <= \''+stampEnd+'\''
        result = client.query(query)
        points = list(result.get_points())
        values =  map(operator.itemgetter('mean'), points)
        times  =  map(operator.itemgetter('time'),  points)
        # import pdb; pdb.set_trace()
        systolic = int(np.array(list(values)))

        if (systolic<low_range) or (systolic>high_range):
            print('Out of designed range!')
            continue

        print(systolic)



        client = InfluxDBClient(ip, "8086", "test", "sensorweb", "caretaker4_ctru", True)
        unit = "ca:re:ta:ke:r0:00"
        query = 'SELECT mean("diastolic") FROM caretaker4 WHERE ("location" = \''+unit+'\') and time >= ' +stampIni+' and time <= '+stampEnd
        result = client.query(query)
        points = list(result.get_points())
        values =  map(operator.itemgetter('mean'), points)
        times  =  map(operator.itemgetter('time'),  points)
        # import pdb; pdb.set_trace()
        diastolic = int(np.array(list(values)))
        # info[4] = diastolic
        print(diastolic)


        filtered_data_list.append(list(data)+[systolic]+[diastolic])
    except:
        continue

filtered_data = np.asarray(filtered_data_list)
np.save('../data/115_125',filtered_data)

import pdb; pdb.set_trace()

# pdf_not_good.close()


pdf = PdfPages(f'../{low_range}_{high_range}_raw_spectrum.pdf')

# plt.figure(figsize=(20, 4))
for jnd, each_filtered_data in enumerate(filtered_data):

    plt.figure(figsize=(20, 4))



    each = each_filtered_data[:1000]
    systolic = each_filtered_data[1000]
    diastolic = each_filtered_data[1001]
    # use seismic data
    x = each/np.linalg.norm(each)
    fs = 100
    fmin = 2
    fmax = 10
    npseg = 200
    slidingstep=20

    plt.subplot(2,1,1)
    plt.plot(x)
    plt.vlines(index_line, min(x), max(x),'r')
    plt.title(f'Raw {info[0]}, SBP {systolic}, DBP {diastolic}')

    plt.subplot(2,1,2)
    f, t, Zxx = signal.spectrogram(x, fs)
    plt.pcolormesh(t, f[int(fmin*npseg/fs):int(fmax*npseg/fs)+1], np.abs(Zxx[int(fmin*npseg/fs):int(fmax*npseg/fs)+1,:]), shading='gouraud')
    plt.title(f'STFT {info[0]}, SBP {systolic}, DBP {diastolic}')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')

    # plt.show()


    pdf.savefig()
    plt.close()



    # # plt.figure(figsize=(20, 4))
    # plt.subplot(filtered_data.shape[0],2,2*jnd+1)
    # plt.plot(x)
    # plt.title(f'Raw {info[0]}, SBP {systolic}, DBP {diastolic}')
    #
    #
    # f, t, Zxx = signal.spectrogram(x, fs)
    # # f, t, Zxx = signal.stft(x, fs, nperseg=npseg, noverlap = npseg-slidingstep)
    # plt.subplot(filtered_data.shape[0],2,2*jnd+2)
    # plt.pcolormesh(t, f[int(fmin*npseg/fs):int(fmax*npseg/fs)+1], np.abs(Zxx[int(fmin*npseg/fs):int(fmax*npseg/fs)+1,:]), shading='gouraud')
    # #plt.pcolormesh(t, f, np.abs(Zxx), shading='gouraud')
    #
    #
    # plt.title(f'STFT {info[0]}, SBP {systolic}, DBP {diastolic}')
    # plt.ylabel('Frequency [Hz]')
    # plt.xlabel('Time [sec]')




# plt.show()
pdf.close()
plt.close()



# # [2]
# # https://homedots.us:3000/grafana/d/OSjxFKvGk/caretaker-vital-signs?orgId=1&var-mac=b8:27:eb:6c:6e:22&from=1612294469323&to=1612294482742
# ip = "sensorweb.us"
# unit = "b8:27:eb:6c:6e:22" # BedJ
# stampIni = "1612294469323" # UTC time = EST time + 4
# stampEnd = "1612294482742"
# client = InfluxDBClient(ip, "8086", "test", "sensorweb", "shake_ctru", True)
# query = 'SELECT "value" FROM Z WHERE ("location" = \''+unit+'\')  and time >= '+stampIni+'000000 and time <= '+stampEnd+'000000'

# result = client.query(query)
# points = list(result.get_points())
# values =  map(operator.itemgetter('value'), points)
# times  =  map(operator.itemgetter('time'),  points)
# data = np.array(list(values))
# fs = 100 # for vibration data

# if(len(data) == 0):
#     print("No data in the chosen time range!")
#     quit()

# # use seismic data
# x = data/np.linalg.norm(data)
# fs = 100
# fmin = 2
# fmax = 10
# npseg = 200

# slidingstep=20

# plt.figure(figsize=(20, 4))
# plt.plot(x)
# plt.show()

# f, t, Zxx = signal.stft(x, fs, nperseg=npseg, noverlap = npseg-slidingstep)
# plt.figure(figsize=(16, 4))
# plt.pcolormesh(t, f[int(fmin*npseg/fs):int(fmax*npseg/fs)+1], np.abs(Zxx[int(fmin*npseg/fs):int(fmax*npseg/fs)+1,:]), shading='gouraud')
# #plt.pcolormesh(t, f, np.abs(Zxx), shading='gouraud')

# plt.title('STFT Magnitude')
# plt.ylabel('Frequency [Hz]')
# plt.xlabel('Time [sec]')
# plt.show()
