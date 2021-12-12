import os
import glob
import array
import base64
import xmltodict
import numpy as np

def decodeLeadData(leadData):
    leads = dict()
    for lead in leadData:
        lead_id = lead['LeadID'] # [I, II, V1, V2, V3, V4, V5, V6] OR # [I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6]
        lead_data = lead['WaveFormData']
        if type(lead_data) == str:
            lead_data = lead_data.replace("\n", "")
            lead_b64 = base64.b64decode(lead_data)
            if len(lead_b64) > 10000: lead_b64 = lead_b64[:10000]
            if len(lead_b64) % 2 == 1: lead_b64 = lead_b64[:len(lead_b64)-1]
            while len(lead_b64) < 10000:
                lead_b64 += b"\x00"
            lead_vals = np.array(array.array('h', lead_b64))

            n_samples = len(lead_vals)
            for i in range(5000 - n_samples):
                lead_vals = np.append(lead_vals, [0])     # Zero Padding

            leads[lead_id] = lead_vals
    return leads

def loadECG(path):
    with open(path, 'rb') as xml:
        ECG = xmltodict.parse(xml.read().decode('ISO-8859-1'))
    waveforms= ECG['RestingECG']['Waveform']
    if type(waveforms) == list:
        for wf in waveforms:
            if wf['WaveformType'] == 'Rhythm':  # 'Median', 'Rhythm'
                n_leads = int(wf['NumberofLeads'])
                leads = decodeLeadData(wf['LeadData'])
    else:
        wf = waveforms
        n_leads = int(wf['NumberofLeads'])
        leads = decodeLeadData(wf['LeadData'])
        
    # zero padding for missed lead
    if not 'I' in leads:
        leads['I'] = np.zeros(5000)
    if not 'II' in leads:
        leads['II'] = np.zeros(5000)
    if not 'III' in leads:
        leads['III'] = np.zeros(5000)
    if not 'aVR' in leads:
        leads['aVR'] = np.zeros(5000)
    if not 'aVL' in leads:
        leads['aVL'] = np.zeros(5000)
    if not 'aVF' in leads:
        leads['aVF'] = np.zeros(5000)
    if not 'V1' in leads:
        leads['V1'] = np.zeros(5000)
    if not 'V2' in leads:
        leads['V2'] = np.zeros(5000)
    if not 'V3' in leads:
        leads['V3'] = np.zeros(5000)
    if not 'V4' in leads:
        leads['V4'] = np.zeros(5000)
    if not 'V5' in leads:
        leads['V5'] = np.zeros(5000)
    if not 'V6' in leads:
        leads['V6'] = np.zeros(5000)
        
    return np.vstack((leads['I'],leads['II'],leads['III'],leads['aVR'],leads['aVL'],leads['aVF'],leads['V1'],leads['V2'],leads['V3'],leads['V4'],leads['V5'],leads['V6']))

def augmetation(X_orig):
    """ 8 leads to 12 leads """
    X_aug = np.copy(X_orig)

    for i, X in enumerate(X_aug):
        has_zeros = np.count_nonzero(X, axis=0) == 0
        if np.any(has_zeros):
            if has_zeros[0]:  # Lead I
                if not has_zeros[1] and not has_zeros[2]:
                    X[:,0] = X[:,1] - X[:,2] # I = II - III
                    has_zeros[0] = False
                elif not has_zeros[1] and not has_zeros[3]:
                    X[:,0] = -X[:,3]*2. - X[:,1] # I = - 2 * aVR - II
                    has_zeros[0] = False
                elif not has_zeros[1] and not has_zeros[4]:
                    X[:,0] = X[:,4] + X[:,1]*0.5 # I = aVL + II / 2
                    has_zeros[0] = False
                elif not has_zeros[1] and not has_zeros[5]:
                    X[:,0] = 2.*(X[:,1] - X[:,5]) # I = 2 * (II - aVF)
                    has_zeros[0] = False
                elif not has_zeros[2] and not has_zeros[3]:
                    X[:,0] = -0.5*X[:,2] - X[:,3]
                    has_zeros[0] = False
                elif not has_zeros[2] and not has_zeros[4]:
                    X[:,0] = X[:,2] + 2*X[:,4]
                    has_zeros[0] = False
                elif not has_zeros[2] and not has_zeros[5]:
                    X[:,0] = 2*(X[:,5] - X[:,2])
                    has_zeros[0] = False
                elif not has_zeros[3] and not has_zeros[4]:
                    X[:,0] = 2*(X[:,4] - X[:,3])/3
                    has_zeros[0] = False
                elif not has_zeros[3] and not has_zeros[5]:
                    X[:,0] = -2*(2*X[:,3] + X[:,5])/3
                    has_zeros[0] = False
                elif not has_zeros[4] and not has_zeros[5]:
                    X[:,0] = 2*(X[:,5] + 2*X[:,4])/3
                    has_zeros[0] = False
            if has_zeros[1]:  # Lead II
                if not has_zeros[0] and not has_zeros[2]:
                    X[:,1] = X[:,0] + X[:,2] # II = I + III
                    has_zeros[1] = False
                elif not has_zeros[0] and not has_zeros[3]:
                    X[:,1] = -X[:,3]*2. - X[:,0] # II = - 2 * aVR - I
                    has_zeros[1] = False
                elif not has_zeros[0] and not has_zeros[4]:
                    X[:,1] = 2.*(X[:,0] - X[:,4]) # II = 2 * (I - aVL)
                    has_zeros[1] = False
                elif not has_zeros[0] and not has_zeros[5]:
                    X[:,1] = X[:,5] + X[:,0]*0.5 # II = aVF + I / 2
                    has_zeros[1] = False
                elif not has_zeros[2] and not has_zeros[3]:
                    X[:,1] = 0.5*X[:,2] - X[:,3]
                    has_zeros[1] = False
                elif not has_zeros[2] and not has_zeros[4]:
                    X[:,1] = 2*(X[:,2] + X[:,4])
                    has_zeros[1] = False
                elif not has_zeros[2] and not has_zeros[5]:
                    X[:,1] = 2*X[:,5] - X[:,2]
                    has_zeros[1] = False
                elif not has_zeros[3] and not has_zeros[4]:
                    X[:,1] = -2*(X[:,4] + 2*X[:,3])/3
                    has_zeros[1] = False
                elif not has_zeros[3] and not has_zeros[5]:
                    X[:,1] = 2*(X[:,5] - X[:,3])/3
                    has_zeros[1] = False
                elif not has_zeros[4] and not has_zeros[5]:
                    X[:,1] = 2*(X[:,4] + 2*X[:,5])
                    has_zeros[1] = False
            if has_zeros[2]:  # Lead III
                if not has_zeros[0] and not has_zeros[1]:
                    X[:,2] = X[:,1] - X[:,0]  # III = II - I
                    has_zeros[2] = False
            if has_zeros[3]:
                if not has_zeros[0] and not has_zeros[1]:
                    X[:,3] = (X[:,0] + X[:,2])*(-0.5)
                    has_zeros[3] = False
            if has_zeros[4]:
                if not has_zeros[0] and not has_zeros[1]:
                    X[:,4] = X[:,0] - 0.5*X[:,1]
                    has_zeros[4] = False
            if has_zeros[5]:
                if not has_zeros[0] and not has_zeros[1]:
                    X[:,5] = X[:,1] - 0.5*X[:,0]
                    has_zeros[5] = False
    return X_aug 

def read_files(path):
    X = list()
    y = list()
    for file in glob.glob(os.path.join(*[path, 'normal', '*.xml'])):
        y.append(0)
        X.append(loadECG(file))
    for file in glob.glob(os.path.join(*[path, 'arrhythmia', '*.xml'])):
        y.append(1)
        X.append(loadECG(file))
    return np.array(X), np.array(y)

def read_data(path, n_leads=2):
    X, y = read_files(path)
    X = X.transpose(0,2,1)
    X = augmetation(X)
    if n_leads == 2:
        X = np.stack((X[:,:,0], # Lead I
                    X[:,:,1], # Lead II
                    ), axis=-1)
    elif n_leads == 8:
        X = np.stack((X[:,:,0], # Lead I
                    X[:,:,1], # Lead II
                    X[:,:,6], # Lead V1
                    X[:,:,7], # Lead V2
                    X[:,:,8], # Lead V3
                    X[:,:,9], # Lead V4
                    X[:,:,10],# Lead V5
                    X[:,:,11],# Lead V6
                    ), axis=-1)
    return X, y



if __name__ == '__main__':
    X_train, y_train = read_data('electrocardiogram/data/train')
    print(X_train)
    print(y_train)