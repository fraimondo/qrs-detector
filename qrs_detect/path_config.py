from uuid import getnode as get_mac

PATH_CONFIG = dict()

this_mac = ':'.join(("%012X" % get_mac())[i:i+2] for i in range(0, 12, 2))

if this_mac == '78:31:C1:D0:B0:90':  # Fede MBP
    PATH_CONFIG['MATLAB_DATABASE'] = '/Users/fraimondo/data/DATABASE_FINAL'
    # PATH_CONFIG['MATLAB_SCRIPTS'] = '/Users/fraimondo/Dropbox/Jaco/scripts'
    # PATH_CONFIG['DB_DETAILS'] = '/Users/fraimondo/data/extra/DB_details.mat'
    PATH_CONFIG['MNE_PATH'] = '/Users/fraimondo/dev/mne-python'
    PATH_CONFIG['PATIENTS_DATABASE'] = '/Volumes/ExtData/database_fiff'
    # PATH_CONFIG['DB_JR'] = '/Users/fraimondo/data/extra/ICM_database_jr.mat'
    PATH_CONFIG['HEART_DATABASE'] = '/Users/fraimondo/data/heart'
    PATH_CONFIG['EXTRA'] = '/Users/fraimondo/data/extra'
elif this_mac in ['2C:27:D7:EF:48:69', '2C:27:D7:EF:48:68']:  # Charles
    PATH_CONFIG['MATLAB_DATABASE'] = '/media/data/DATABASE_FINAL'
    # PATH_CONFIG['MATLAB_SCRIPTS'] = '/Users/fraimondo/Dropbox/Jaco/scripts'
    PATH_CONFIG['PATIENTS_DATABASE'] = '/media/data/database_fiff'
    PATH_CONFIG['ANESTESIA_DATABASE'] = '/media/data/anestesia'
    PATH_CONFIG['MNE_PATH'] = '/home/fraimondo/dev/mne-python'
    PATH_CONFIG['NEW_PATIENTS_DATABASE'] = '/media/data/new_database_fiff'
    PATH_CONFIG['INCOMING_PATIENTS'] = '/media/data/new_patients'
elif this_mac == '90:2B:34:3E:E7:22':
    PATH_CONFIG['HEART_DATABASE'] = '/Volumes/Big/heart'
else:
    raise ValueError('Host ' + this_mac + ' not configured')
