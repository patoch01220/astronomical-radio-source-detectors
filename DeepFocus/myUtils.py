import os
# import sys
import numpy as np
import math
# from tqdm import tqdm

from astropy.table import Table
# from astropy.coordinates import SkyCoord
# import astropy.units as u

def writingModel(folder_path, listSources, prefix,  indices, keys):
    '''
    Function that writes the model file
    Inputs:
        - path: path where the file will be written
        - listSources: list of sources
        - prefix: prefix of the txt files
        - indices: indices of the set
        - keys: keys of the set
    '''
    for i_file in range(len(indices)):
        ind = indices[i_file]
        key = keys[ind]
        
        file_path = os.path.join(folder_path, prefix + key + ".txt")
        
        with open(file_path, 'w+') as f:
            f.write("# format:name ra_d dec_d i q u v emaj_s emin_s pa_d\n")
            for i_source in range(len(listSources[i_file])):
                f.write("source" + str(i_source) + " " 
                    + str(listSources[i_file][i_source][0]) + " " 
                    + str(listSources[i_file][i_source][1]) + " " 
                    + str(listSources[i_file][i_source][2]) + " " 
                    + str(listSources[i_file][i_source][3]) + " " 
                    + str(listSources[i_file][i_source][4]) + " " 
                    + str(listSources[i_file][i_source][5]) + " " 
                    + str(listSources[i_file][i_source][6]) + " "
                    + str(listSources[i_file][i_source][7]) + " "
                    + str(listSources[i_file][i_source][8]) + "\n")

# def detectSourcesFromAllFiles(testSet_path):
#     listDetectedSources_test = []
#     listIndex_NoSources = []
#     for i in range(len(testSet_path)):
#         path = testSet_path[i]
        
#         image_tmp = image.Image.read_from_file(path)
#         try:
#             detection = result.SourceDetectionResult.detect_sources_in_image(image_tmp, quiet = True)

#         except IndexError:
#             listIndex_NoSources.append(i)
#             listDetectedSources_test.append(np.array([]))
#             continue

#         listDetectedSources_test.append(detection.get_pixel_position_of_sources())
#     return listDetectedSources_test


# def saveSourceDetection(listDetectedSources, saveSourceDetection_path, prefix,  indices, keys):
#     '''
#     Function that saves all the sources detected into a npy file
#     Inputs:
#         - listDetectedSources: list of all the sources detected
#         - saveSourceDetection_path: path where to save the npy files
#         - prefix: prefix of the npy files
#         - indices: indices of the set
#         - keys: keys of the set
#     '''

#     for i in range(len(indices)):
#         ind = indices[i]
#         key = keys[ind]
        
#         final_path = os.path.join(saveSourceDetection_path, prefix + key + ".npy")
#         np.save(final_path, listDetectedSources[i])

# def loadSourceDetection(loadSourceDetection_path, prefix, indices, keys):
#     '''
#     Function that loads all the sources detected from a npy file
#     Inputs:
#         - loadSourceDetection_path: path where the npy files are located
#         - prefix: prefix of the npy files
#         - indices: indices of the set
#         - keys: keys of the set
#     '''
#     listDetectedSources = []
#     for i in range(len(indices)):
#         ind = indices[i]
#         key = keys[ind]
        
#         final_path = os.path.join(loadSourceDetection_path, prefix + key + ".npy")
#         listDetectedSources.append(np.load(final_path))
    
#     return listDetectedSources

def loadAllCatFile(folder_path, prefix, indices, keys):
    '''
    Function that loads all the cat files from the folder
    Inputs:
        - folder_path: path where the cat files are located
        - indices: indices of the set
        - keys: keys of the set
    '''
    listCat = []
    for i in range(len(indices)):
        ind = indices[i]
        key = keys[ind]
        
        sourcesCoord = []
        table = Table.read(os.path.join(folder_path, prefix + key + ".cat"), format='ascii')
        for row in range(len(table)):
            majAxis = float(table[row]['majoraxis'][:-6])
            minAxis = float(table[row]['minoraxis'][:-6])
            PA = float(table[row]['PA'][:-3])
            sourcesCoord.append([table[row]['ra'].astype(float), table[row]['dec'].astype(float), table[row]['flux'].astype(float), 0.000, 0.000, 0.0000, majAxis, minAxis, PA])

        listCat.append(np.array(sourcesCoord))
    return listCat

def RaDec2pixels_help(x, y, headers):
    '''
    Function that converts the Ra & Dec coordinates to pixels coordinates
    Inputs:
        - x: x coordinate
        - y: y coordinate
        - headers: headers of the fits file
    '''

    pixel_x = headers['CRPIX1'] - np.round((headers['CRVAL1'] - x) / headers["CDELT1"]) - 1
    pixel_y = headers['CRPIX2'] - np.round((headers['CRVAL2'] - y) / headers["CDELT2"]) - 1

    return np.array([pixel_x, pixel_y])

def RaDec2pixels(sourcesCoord, headers):
    '''
    Function that converts the Ra & Dec coordinates to pixels coordinates
    Inputs:
        - sourcesCoord: list of the sources coordinates
        - headers: headers of the fits file
    '''

    sourcesCoord = np.array(sourcesCoord)
    sourcesCoord = sourcesCoord.T
    sourcesCoord = np.array([RaDec2pixels_help(sourcesCoord[0][i], sourcesCoord[1][i], headers) for i in range(len(sourcesCoord[0]))])
    sourcesCoord = sourcesCoord.T

    return sourcesCoord

def pixels2RaDec(x, y, headers):
    '''
    Function that converts the pixels coordinates to Ra & Dec coordinates
    Inputs:
        - x: x coordinate
        - y: y coordinate
        - headers: headers of the fits file
    '''
    ra = headers['CRVAL1'] - (headers['CRPIX1'] - x - 1) * headers["CDELT1"]
    dec = headers['CRVAL2'] - (headers['CRPIX2'] - y - 1) * headers["CDELT2"]

    return np.array([ra, dec])

def compareRealAndDetectedSources_pixels(listDetectedSources, listRealDetection, Rmax = 10):
    '''
    Function that compares the real sources with the detected sources and computes the TP, FP and FN
    Inputs:
        - listDetectedSources: list of the detected sources
        - listRealDetection: list of the real sources
        - Rmax: maximum distance between the real and the detected source
    Outputs:
        - nTP: number of true positives
        - nFP: number of false positives
        - nFN: number of false negatives
    '''
    nFP = 0
    nFN = 0
    nTP = 0

    if len(listDetectedSources) != len(listRealDetection):
        raise Exception("The number of files in the two list doesn't match")
 
    for ind in range(len(listDetectedSources)):    # Iterate through all the files

        if listDetectedSources[ind].size == 0:
            nFN += listRealDetection[ind].shape[1]
            continue

        detectedSources_tmp = np.round(np.array(listDetectedSources[ind].copy()))
        
        for ir in range(listRealDetection[ind].shape[1]):     # iterate through all the real sources
            listDist_tmp = np.array([])
            for id in range(detectedSources_tmp.shape[0]):     # iterate through all the detected sources
                x_real, y_real = listRealDetection[ind][0][ir], listRealDetection[ind][1][ir]
                x_detected, y_detected = detectedSources_tmp[id][0], detectedSources_tmp[id][1]

                R = math.sqrt((x_real - x_detected) ** 2 + (y_real - y_detected) ** 2)
                listDist_tmp = np.append(listDist_tmp, R)

            if True in (listDist_tmp <= Rmax):   # If there is a detected source close enough to the real source
                minIndex = np.argmin(listDist_tmp)
                detectedSources_tmp = np.delete(detectedSources_tmp, minIndex, axis=0)
                nTP += 1
            else:   # If there is no detected source close enough to the real source
                nFN += 1

        nFP += detectedSources_tmp.shape[0]     # The remaining detected sources are false positives
    
    return np.array([nTP, nFP, nFN])