import os
import lmdb # install lmdb by "pip install lmdb"
import cv2
import numpy as np
import codecs
import random


def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    imageBuf = np.fromstring(imageBin, dtype=np.uint8)
    if imageBuf is None:
        return False
    img = cv2.imdecode(imageBuf, cv2.IMREAD_COLOR)
    if img is None:
        return False
    imgH, imgW, imgC = img.shape[0], img.shape[1], img.shape[2]
    if imgH * imgW * imgC == 0:
        return False
    # img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
    # imgH, imgW = img.shape[0], img.shape[1]
    # if imgH * imgW == 0:
    #     return False
    return True


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.iteritems():
            txn.put(k, v)


def createDataset(outputPath, imagePathList, labelList, lexiconList=None, checkValid=True):
    """
    Create LMDB dataset for CRNN training.
    ARGS:
        outputPath    : LMDB output path
        imagePathList : list of image path
        labelList     : list of corresponding groundtruth texts
        lexiconList   : (optional) list of lexicon lists
        checkValid    : if true, check the validity of every image
    """
    assert(len(imagePathList) == len(labelList))
    nSamples = len(imagePathList)
    if os.path.exists(outputPath):
        print('%s exists, remove the old one.' % outputPath)
        import shutil
        shutil.rmtree(outputPath)
    env = lmdb.open(outputPath, map_size=1099511627776)
    cache = {}
    cnt = 1
    for i in xrange(nSamples):
        imagePath = imagePathList[i]
        label = labelList[i]
        if not os.path.exists(imagePath):
            print('%s does not exist' % imagePath)
            continue
        with open(imagePath, 'r') as f:
            imageBin = f.read()
        if checkValid:
            if not checkImageIsValid(imageBin):
                print('%s is not a valid image' % imagePath)
                continue

        imageKey = 'image-%09d' % cnt
        labelKey = 'label-%09d' % cnt
        cache[imageKey] = imageBin
        cache[labelKey] = label
        if lexiconList:
            lexiconKey = 'lexicon-%09d' % cnt
            cache[lexiconKey] = ' '.join(lexiconList[i])
        if cnt % 1000 == 0:
            print('Written %d / %d' % (cnt, nSamples))            
            cache['num-samples'] = str(cnt)
            writeCache(env, cache)
            cache = {}
        cnt += 1
    nSamples = cnt-1
    cache['num-samples'] = str(nSamples)
    writeCache(env, cache)
    print('Created dataset with %d samples' % nSamples)


if __name__ == '__main__':
    # load the train sample list.
    img_path_list = []
    label_list = []
    with codecs.open('/home/helios/Documents/zenghui/caffe-20160826/data/challengeIV/mnt/ramdisk/max/90kDICT32px/annotation_train.txt', \
        'r') as f:
        lines = f.readlines()
        random.shuffle(lines)
        for line in lines:
            tmp = line.split(' ')
            if len(tmp) != 2:
                print 'invalid line: ', line
                raise ValueError('invalid line!')
            path = os.path.join('/home/helios/Documents/zenghui/caffe-20160826/data/challengeIV/mnt/ramdisk/max/90kDICT32px', tmp[0])
            img_path_list.append(path)
            label = tmp[0].split('_')[1]
            def labelCheck(inputString):
                target_set = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
                return all(char in target_set for char in inputString)
            if not labelCheck(label):
                print ('label contains char outside the target set: ', label)
                continue
            label_list.append(label)
    
    assert(len(img_path_list) == len(label_list))

    print 'total training sample: ', len(img_path_list)

    createDataset('./datasets/syntext90K_train', img_path_list, label_list, checkValid=True)

    img_path_list = []
    label_list = []
    with codecs.open('/home/helios/Documents/zenghui/caffe-20160826/data/challengeIV/mnt/ramdisk/max/90kDICT32px/annotation_.valtxt', \
        'r') as f:
        lines = f.readlines()
        random.shuffle(lines)
        for line in lines:
            tmp = line.split(' ')
            if len(tmp) != 2:
                print 'invalid line: ', line
                raise ValueError('invalid line!')
            path = os.path.join('/home/helios/Documents/zenghui/caffe-20160826/data/challengeIV/mnt/ramdisk/max/90kDICT32px', tmp[0])
            img_path_list.append(path)
            label = tmp[0].split('_')[1]
            def labelCheck(inputString):
                target_set = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
                return all(char in target_set for char in inputString)
            if not labelCheck(label):
                print ('label contains char outside the target set: ', label)
                continue
            label_list.append(label)
    
    assert(len(img_path_list) == len(label_list))

    print 'total testing sample: ', len(img_path_list)

    createDataset('./datasets/syntext90k_val', img_path_list, label_list)