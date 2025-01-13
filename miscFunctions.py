import numpy as np
import cv2
import math
from statistics import mean
import sys
import os
import helper_function

def writeFramesFromVideo(extractDir,videoDir):
    frameNum = 0
    video = cv2.VideoCapture(videoDir)
    while True:
        ret,frame = video.read()
        if not ret:
            break
        cv2.imwrite(extractDir+"/frame"+str(frameNum)+".jpg",frame)
        frameNum = frameNum + 1
    video.release()

def writeSingleFrame(dir,frame,num=0):
    try:
        cv2.imwrite(dir+"/singleFrame"+str(num)+".jpg",frame)
    except Exception as e:
        print(e)

def readFrames(baseDirectory,frameNum,filename = 'frame'):
    videoDirectory = baseDirectory + f'/{filename}{frameNum}.jpg'
    image = cv2.imread(videoDirectory)
    return np.array(image)

def calculate_ssd(arr1, arr2):
    return np.sum((arr1 - arr2) ** 2)

def segmentFrame(videoFrame,step=5):
    segmentedFrames = []
    numRows = videoFrame.shape[0]
    numCols = videoFrame.shape[1]

    for x in range(0,numRows,step):

        blockRow = []
        for y in range(0,numCols,step):
            row = []
            for xRow in range(x,x+step):
                column = []
                for yCol in range(y,y+step):
                    try:
                        column.append(videoFrame[xRow][yCol])
                    except:
                        column.append(np.array([0,0,0]))
                row.append(np.array(column))
            blockRow.append(np.array(row))

        segmentedFrames.append(np.array(blockRow))
    return np.array(segmentedFrames)

def unsegmentFrame(videoFrame,blockSize,shapeX,shapeY):
    unsegmentedFrame = []

    for x in range(0,shapeX):
        row = []
        for y in range(0,shapeY):
            try:
                blockX = int((x/blockSize)//1)
                blockY = int((y/blockSize)//1)

                innerBlockX = x-(blockSize*math.floor(x/blockSize))
                innerBlockY = y-(blockSize*math.floor(y/blockSize))

                pixel = videoFrame[blockX][blockY][innerBlockX][innerBlockY]

                row.append(pixel)
            except:
                continue
        row = np.array(row)
        if row.shape[0] > 0:
            unsegmentedFrame.append(np.array(row))
    
    return np.array(unsegmentedFrame)

def sliceNeighbours(frame,width,xIndx,yIndx):
    neighbours = []
    dimension = math.ceil(width/2)
    for x in range(xIndx-dimension,xIndx+dimension+1):
        row = []
        for y in range(yIndx-dimension,yIndx+dimension+1):

            if (x < 0 or y < 0 or x > frame.shape[0]-1 or y > frame.shape[1]-1):
                row.append(np.array([None]))
            else:
                row.append(frame[x][y])

        neighbours.append((row))
    return (neighbours)

def additionalSegmentation(vectors,x,y,jump,blocksize):
    for innerX in range(x,x+jump):
        for innerY in range(y,y+jump):
            try:
                if not (np.all(vectors[innerX][innerY]) == 0):
                    return {'start':[x,y],'end':[x+jump,y+jump],'center':[math.ceil((jump/2)+x),math.ceil((jump/2)+y)],'history':[]}
            except:
                continue
    return None

def searchElement(element,neighbours,width):
    bestX = 0 
    bestY = 0
    bestScore = sys.maxsize
    width = math.ceil(width/2)

    for x in range(0,len(neighbours)):
        for y in range(0,len(neighbours)):
            if not (neighbours[x][y].any() == None):
                score = calculate_ssd(element,neighbours[x][y])
                if score < bestScore:
                    bestScore = score
                    bestX = x-width
                    bestY = y-width

    return [bestX,bestY,bestScore]

def searchVectors(startX,startY,mapOne,mapTwo,mapThree,search):
    CombinedMapOne = []

    for x in range(startX-search,startX+search):
        for y in range(startY-search,startY+search):
            try:
                if (not (mapOne[x][y][0] == mapOne[x][y][1] == mapOne[x][y][2] == mapOne[x][y][3] == 0)):
                    CombinedMapOne.append(mapOne[x][y])
                if (not (mapTwo[x][y][0] == mapTwo[x][y][1] == mapTwo[x][y][2] == mapTwo[x][y][3] == 0)):
                    CombinedMapOne.append(mapTwo[x][y])
                if (not (mapThree[x][y][0] == mapThree[x][y][1] == mapThree[x][y][2] == mapThree[x][y][3] == 0)):
                    CombinedMapOne.append(mapThree[x][y])
            except:
                continue
    
    return average_arrows(CombinedMapOne)

def average_arrows(arrow_list):
    if not arrow_list:
        return None

    sum_x1 = 0.0
    sum_y1 = 0.0
    sum_x2 = 0.0
    sum_y2 = 0.0

    for arrow in arrow_list:
        if len(arrow) != 4:
            raise ValueError("Each arrow should have four components: [x1, y1, x2, y2]")

        x1, y1, x2, y2 = arrow
        sum_x1 += x1
        sum_y1 += y1
        sum_x2 += x2
        sum_y2 += y2

    avg_x1 = sum_x1 / len(arrow_list)
    avg_y1 = sum_y1 / len(arrow_list)
    avg_x2 = sum_x2 / len(arrow_list)
    avg_y2 = sum_y2 / len(arrow_list)

    avg_arrow = [round(avg_x1), round(avg_y1), round(avg_x2), round(avg_y2)]
    return avg_arrow

def percentageBlue(element):
    redGreen = 0
    blue = 0
    for x in range(0,element.shape[0]):
        for y in range(0, element.shape[1]):
            blue = blue+(int)(element[x][y][0])
            redGreen = redGreen + (int)(element[x][y][1]) + (int)(element[x][y][2])
    return blue/(redGreen+blue)

def medianMotion(motionVectors,windowSize):
    filteredVectors = []

    for i in range(0,len(motionVectors),2):
        startIdx = max(0,i-windowSize//2)
        endIdx = min(len(motionVectors),i + windowSize // 2 + 1)
        window = motionVectors[startIdx:endIdx]

        xValues = [vector[2]-vector[0] for vector in window]
        yValues = [vector[3]-vector[1] for vector in window]

        median_x = np.median(xValues)
        median_y = np.median(yValues)

        filteredVector = [
            int(motionVectors[i][0]),
            int(motionVectors[i][1]),
            int(motionVectors[i][0] + median_x),
            int(motionVectors[i][1] + median_y)
        ]

        filteredVectors.append(filteredVector)
    
    return filteredVectors

def drawArrows(image,vectors):
    for x in range(0,len(vectors)):
        image = helper_function.arrowdraw(image, int(vectors[x][0]), int(vectors[x][1]), int(vectors[x][2]), int(vectors[x][3]))
    return image

def euclidean_distance(coord1, coord2):
    x1, y1 = coord1
    x2, y2 = coord2
    distance = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    return distance

def find_nearest(start_x, start_y, radius, coordlist, acceptStr=False):
    init_coord = (start_x,start_y)
    
    startX = start_x - radius
    startY = start_y - radius

    endX = start_x + radius
    endY = start_y + radius

    best_dist = sys.maxsize
    key = (None,None)

    for x in range(startX,endX):
        for y in range(startY,endY):
            coordKey = f'{x}{y}'
            coord = (x,y)

            if (coordKey in coordlist and not coord == init_coord):
                if (isinstance(coordlist[coordKey], str) or acceptStr):
                    continue
                
                dist = euclidean_distance(init_coord,coord)
                if (dist < best_dist):
                    key = coordKey
                    best_dist = dist
    return key

def grabFrames(start, end, baseDir):
    images = []
    for x in range(start, end):
        image = readFrames(baseDir, x, 'singleFrame')
        images.append(image)
    return images

def create_video_from_images(baseDir, lastFrame, frame_rate):
    images = grabFrames(0, lastFrame, baseDir)

    if len(images) > 0:
        frame_height, frame_width, _ = images[0].shape
    else:
        raise Exception('Image read error')

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
    output_video_path = os.path.join(baseDir, 'video.mp4')  
    out = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (frame_width, frame_height))

    count = 0
    for image in images:
        try:
            out.write(image)
            print(f'Writing frame {count}')
            count += 1
        except:
            break

    out.release()
    print(f'Video saved to {output_video_path}')