import numpy as np
import os
from statistics import mean
import helper_function
import sys
import miscFunctions as msc

baseDir = os.path.dirname(os.path.abspath(__file__))
extractFrames = baseDir+'/frameExtract/'
videoDir = baseDir+'/monkey.avi'

#HyperParamters
blockSize = 3
framenumber = 752
width = 6
jump = 12

for frameIndx in range(0,framenumber-1,2):
    initFrame = frameIndx
    localGroup = [[msc.readFrames(extractFrames,frameIndx)],[msc.readFrames(extractFrames,frameIndx+1)],[msc.readFrames(extractFrames,frameIndx+2)],[msc.readFrames(extractFrames,frameIndx+3)],[msc.readFrames(extractFrames,frameIndx+4)]]
    imageFrag = msc.segmentFrame(localGroup[0][0],blockSize)

    for groupIndex in range(0,4):
        nextImageFrag = msc.segmentFrame(localGroup[groupIndex+1][0],blockSize)
        vectorSpace = np.zeros((imageFrag.shape[0],imageFrag.shape[1],4))

        for x in range(0,imageFrag.shape[0]):
            for y in range(0,imageFrag.shape[1]):
                if (msc.percentageBlue(imageFrag[x][y]) < 0.49):
                    sourse = imageFrag[x][y]
                    neighbours = msc.sliceNeighbours(nextImageFrag,width,x,y)
                    destX,destY,score = msc.searchElement(sourse,neighbours,width)
                    
                    if (not (destX == destY == 0)):
                        vectorSpace[x][y] = np.array([x,y,(x+destX),(y+destY)])
        
        localGroup[groupIndex].append(vectorSpace)
        imageFrag = nextImageFrag
    
    searchAreas = []
    vectorSpace = localGroup[0][1]
    for x in range(0,vectorSpace.shape[0],jump): 
        for y in range(0,vectorSpace.shape[1],jump): 
            searchArea = msc.additionalSegmentation(vectorSpace,x,y,jump,blockSize)
            if (not searchArea == None):
                searchAreas.append(searchArea)

    for index in range(0,4):
        vectorSpace = localGroup[index][1]
        for searchAreaIndex in range(0,len(searchAreas)):
            searchArea = searchAreas[searchAreaIndex]
            localArrows = []
            for x in range(searchArea['start'][0],searchArea['end'][0]):
                for y in range(searchArea['start'][1],searchArea['end'][1]):
                    try :
                        if not (np.all(vectorSpace[x][y] == 0)):
                            localArrows.append(vectorSpace[x][y])
                    except:
                        continue

            if (len(localArrows) > 0):
                averageArrow = msc.average_arrows(localArrows)
                if (searchAreaIndex == 0):
                    searchArea['history'].append(np.array([averageArrow[0],averageArrow[1]]))

                diffX = averageArrow[2] - searchArea['center'][0]
                diffY = averageArrow[3] - searchArea['center'][1]
                
                searchArea['start'][0] += diffX
                searchArea['end'][0] += diffX
                searchArea['start'][1] += diffY
                searchArea['end'][1] += diffY
                searchArea['center'][0] += diffX
                searchArea['center'][1] += diffY
                searchArea['history'].append(np.array(searchArea['center']))

    for index in range(0,len(localGroup)-1):
        image = localGroup[index][0]
        for pathIndex in range(0,len(searchAreas)):
            paths = searchAreas[pathIndex]['history']
            for vertex in range(0,len(paths)-1):
                if (vertex+1 >= len(paths)-1):
                    image = helper_function.arrowdraw(image,paths[vertex][1]*blockSize,paths[vertex][0]*blockSize,paths[vertex+1][1]*blockSize,paths[vertex+1][0]*blockSize)
                else:
                    image = helper_function.arrowLine(image,paths[vertex][1]*blockSize,paths[vertex][0]*blockSize,paths[vertex+1][1]*blockSize,paths[vertex+1][0]*blockSize)
        msc.writeSingleFrame(baseDir,image,initFrame+index)
        print('produced Frame',initFrame+index)

msc.create_video_from_images(baseDir,framenumber,16)