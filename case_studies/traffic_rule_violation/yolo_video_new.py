# import the necessary packages
import numpy as np
import argparse
import imutils
import time
import cv2
import os
from . import trafficLightColor

# Additional Packages Needed (added implementation)
import sys
import torch
from .utils.general import (
    non_max_suppression, scale_boxes, xywh2xyxy, 
    xyxy2xywh
)
from ultralytics.utils.plotting import Annotator
from ultralytics.utils.plotting import colors as colors_extra
from .utils.augmentations import letterbox
import pickle
import warnings
import crypten.communicator as comm
from .crypten_detect import multiproc_gpu, _run_sec_model

# GLOBAL
# object trackers got to work by using 
# opencv-python 4.9.0.80
# opencv-contrib-python 4.9.0.80
# opencv-contrib-python-headless 4.9.0.80
#-------------------------------------
OPENCV_OBJECT_TRACKERS = {
	"csrt": cv2.TrackerCSRT_create,
	"kcf": cv2.TrackerKCF_create,
	"boosting": cv2.legacy_TrackerBoosting.create,
	"mil": cv2.TrackerMIL_create,
	"tld": cv2.legacy_TrackerTLD.create,
	"mosse": cv2.legacy_TrackerMOSSE.create
}

def read_im_frame(frame, img_size, stride=32):
    '''
    reads frames for detections that require the YOLOv5 object 
    detector. all other detections / bounding box operations with 
    algorithms from cv2 are performed over standard numpy arrays.
    '''
    im = letterbox(frame, img_size, stride=stride, auto=False, scaleFill=False)[0]
    im = im.transpose((2,0,1))[::-1]
    im = np.ascontiguousarray(im)
    im_tensor = torch.from_numpy(im)/255 # recolor the pixels for inference
    
    # no need to pass the original and new image sizes simultaneously
    return im_tensor

def up(
    trackersList
):
    deleted = []
    for n, pair in enumerate(trackersList):
        tracker, box = pair
        (x, y, w, h) = box
        for n2,pair2 in enumerate(trackersList):
            if(n == n2):
                continue
            tracker2, box2 = pair2
            (x2, y2, w2, h2) = box2
            val = bb_intersection_over_union([x, y, x + w, y + h], [x2, y2, x2 + w2, y2 + h2])
            if(val > 0.4):
                deleted.append(n)
                break
    print(deleted)
    for i in deleted:
        del trackersList[i]

def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

def setLightCoordinates(
    args,               ## args dictionary used for stuff ...
    net_type:str,            ## model to perform inference with
    colors,             ## random color distribution?
    secure=True,        ## bool - indicates whether to use secure inference or not 
    imgsize=(256,256),  ## tuple - the square image size to resize the inference image to
    debug=False         ## bool - debug mode indicator - helpful for dev.
):
    '''
    Args:
    - None

    Returns: 
    - tuple --- (x, y, w, h) where x and y are the (x,y) coordinate 
    pair indicating the location of the center of the bounding box
    and w and h represent the width and height of the bounding box. 
    This bounding box surrounds a traffic light in the input image
    - None --- if no traffic light is detected in the input, None is 
    returned to the caller
    '''
    # load the plaintext model
    net = torch.hub.load('ultralytics/yolov5', net_type, force_reload=True, trust_repo=True)
    
    if secure:
        com_costs = []
        round_nums = []
    run_times = []
    
    vss = cv2.VideoCapture(args["input"])

    num_frames_read = 0
    while True:
        W=None
        H=None
        # read the next frame from the file
        (grabbed, frame) = vss.read()
        if not grabbed:
            break
        else: #### TODO: CHANGE TO REFLECT THE OPTIMAL IMAGE SIZE FOR OUR MODEL
            frame = cv2.resize(frame, (1000, 750))
        # if the frame dimensions are empty, grab them
        if W is None or H is None:
            (H, W) = frame.shape[:2]

        # construct a blob from the input frame and then perform a forward
        # pass of the YOLO object detector, giving us our bounding boxes
        # and associated probabilities

        ###################### TODO TODO TODO: CHANGE THIS TO MAKE WORK WITH YOLOv5 ####################################
        # NOTE: only need to change the object detector used
        # TODO: adjust the output shape as needed for use with the corresponding
        # yolo model of choice. This method simply transforms the image as desired
        
        # add fourth dimension (singleton) for inference
        if debug: 
            print("[DEBUG]: frame = {}".format(frame))
            print("[DEBUG]: frame.shape = {}".format(frame.shape))
            
            
        im_tensor = read_im_frame(frame, imgsize, stride=net.stride).unsqueeze(0) 
    
        if debug: 
            print("[DEBUG-147]: im_tensor.shape = {}".format(im_tensor.shape))
            print("[DEBUG-147]: im_tensor values: \n\t{}".format(im_tensor))
        
        # blob = cv2.dnn.blobFromImage(image=frame, 
        #                              scalefactor=1 / 255.0, 
        #                              size=(416, 416),
        #                              swapRB=True, crop=False)
        
        # TODO: CHANGE net to yolov5 model
        # net.setInput(blob)

        # TODO: What are the dimensions of the output from the layerOutputs function
        # start = time.time()
        # layerOutputs = net.forward(ln)
        # end = time.time()
        
        if secure: 
            cryp_folder = "crypten_tmp"
            if cryp_folder not in os.listdir("experiments"):
                os.mkdir("{}".format(cryp_folder))
            torch.save(im_tensor, "experiments/{}/frame.pth".format(cryp_folder))
            
        # blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
        #     swapRB=True, crop=False)
        
        # # TODO: Change net input to yolo5 model - get it to work
        # net.setInput(blob)
        # start = time.time()
        
        if secure: # record data relevant to the inference process for the paper
            yolo_args = {
                "world_size":2,
                "img_size": imgsize, 
                "model":net,
                "data_path":"experiments/crypten_tmp/frame.pth", 
                "run_label":0, 
                "batch_size":im_tensor.shape[0],
                "folder":"crypten_tmp", 
                "device":'cpu',
                "debug":False                    
            }
            multiproc_gpu(_run_sec_model, 'cpu_val', args=yolo_args) 
            with open("experiments/crypten_tmp/run_0.pkl", "rb") as f:
                outputs, inf_start, inf_end = pickle.load(f)
            with open("experiments/crypten_tmp/comm_tmp_0.pkl", "rb") as com_0:
                alice_com = pickle.load(com_0)
            with open("experiments/crypten_tmp/comm_tmp_1.pkl", "rb") as com_1:
                bob_com = pickle.load(com_1)
            cost = (alice_com['bytes'] + bob_com['bytes'])/(2e6) # convert to MB
            round_vals = (alice_com['rounds'] + bob_com['rounds'])/2
            com_costs.append(cost)
            round_nums.append(round_vals)
            run_times.append(inf_end - inf_start)
        else:
            inf_start = time.time()
            outputs = net(im_tensor)  
            inf_end = time.time()
            run_times.append(inf_end - inf_start)

        # initialize our lists of detected bounding boxes, confidences,
        # and class IDs, respectively
        # boxes = []
        # confidences = []
        # classIDs = []

        # # loop over each of the layer outputs
        # for output in layerOutputs:
        #     # loop over each of the detections
        #     for detection in output:
        #         # extract the class ID and confidence (i.e., probability)
        #         # of the current object detection
        #         scores = detection[5:]
        #         classID = np.argmax(scores)
        #         confidence = scores[classID]

        #         # filter out weak predictions by ensuring the detected
        #         # probability is greater than the minimum probability
        #         if confidence > 0.1:
        #             # scale the bounding box coordinates back relative to
        #             # the size of the image, keeping in mind that YOLO
        #             # actually returns the center (x, y)-coordinates of
        #             # the bounding box followed by the boxes' width and
        #             # height
        #             box = detection[0:4] * np.array([W, H, W, H])
        #             (centerX, centerY, width, height) = box.astype("int")

        #             # use the center (x, y)-coordinates to derive the top
        #             # and and left corner of the bounding box
        #             x = int(centerX - (width / 2))
        #             y = int(centerY - (height / 2))

        #             # update our list of bounding box coordinates,
        #             # confidences, and class IDs
        #             boxes.append([x, y, int(width), int(height)])
        #             confidences.append(float(confidence))
        #             classIDs.append(classID)

        # # apply non-maxima suppression to suppress weak, overlapping
        # # bounding boxes
        # idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.1,
                                # args["threshold"])
        classIDs = []
        preds = non_max_suppression(
            prediction = outputs, 
            conf_thres=args['confidence'], #0.5
            iou_thres=args['iou_threshold'], #0.3
            classes=[2,3,7,9], 
            agnostic=None, 
            max_det=1000
        )[0]
        for i in range(len(preds)):
            if debug: 
                print("[DEBUG]: preds = {}".format(preds))
                print("[DEBUG-261]: preds[i] = {}".format(preds[i]))      
            classIDs.append(int(preds[i][5]))

        # ensure at least one detection exists
        if len(preds) > 0:
            for i in range(len(preds)):                
                # color = [int(c) for c in colors[classIDs[i]]]
                if (classIDs[i] == 9): 
                    frame0 = frame.copy() # get copy of the frame
                    im1 = np.ascontiguousarray(im_tensor)
                    if debug:
                        print("[INFO]: preds = {}".format(preds[i]))
                        print("[INFO]: preds[i][:4] = {}".format(preds[i][:4]))
                        print("[INFO]: im1.shape[2:] = {}, frame.shape = {}".format(im1.shape[2:], frame.shape))
                    
                    # rescale boxes to correct image size    
                    preds[i][:4] = scale_boxes(im1.shape[2:], preds[i][:4], frame0.shape)
                    label = f'{classIDs[i]} {preds[i][4]:.2f}'
                    
                    # annotate image for visual confirmation
                    annotator = Annotator(frame0, line_width=1, example=str(net.names))
                    annotator.box_label(preds[i][:4], label, color=colors_extra(classIDs[i], True))
                    
                    if debug: print("[INFO]: preds[i] = {}".format(preds[i]))
                    
                    preds[i][:4] = xyxy2xywh(preds[i][:4]) # convert the shape after performing scaling
                    
                    cv2.imwrite("experiments/plain_tmp/light_visual.jpg", frame0)
                    
                    del net # free memory associated with the network 
                    
                    x, y, w, h = preds[i][0], preds[i][1], preds[i][2], preds[i][3]
                    vss.release()
                    print("\t[INFO]: num frames read finding light = {}".format(num_frames_read))
                    return (int(x),int(y),int(w),int(h))
                
            # # loop over the indexes we are keeping
            # for i in idxs.flatten():
            #     # extract the bounding box coordinates
            #     (x, y) = (boxes[i][0], boxes[i][1])
            #     (w, h) = (boxes[i][2], boxes[i][3])

            #     # draw a bounding box rectangle and label on the frame
            #     color = [int(c) for c in colors[classIDs[i]]]
            #     if (classIDs[i] == 9):

            #         vss.release()
            #         # cv2.imshow('r',frame[y:y + h, x:x + w])
            #         # cv2.waitKey()
            #         return (x,y,w,h)
        num_frames_read += 1
                
    print("[INFO] if this executes, no light coordinates were found")
    print("[INFO] {} frames were read".format(num_frames_read))
    return None

#--------------------------------------#
# trackers = cv2.MultiTracker_create() #
#--------------------------------------#

def getLightThresh(
        ylight,
        xlight,
        wlight,
        args,
        showStages=True
):
    vss = cv2.VideoCapture(args["input"])
    (grabbed, frame) = vss.read()

    frame = cv2.resize(frame, (1000, 750))
    if(showStages==True):
        cv2.imshow('fr',frame)
        cv2.waitKey()
        cv2.destroyAllWindows()
    temp =frame.copy()
    temp2=frame.copy()

    # Convert the image to gray scale for binary level thresholding
    grayscaled = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # image threshold
    # NOTE: Arguments input, may want to remove these if the code stops working after
    # perform binary level thresholding based on experimentally determined threshold value
    th = cv2.adaptiveThreshold(src=grayscaled,
                               maxValue=250, 
                               adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                               thresholdType=cv2.THRESH_BINARY, 
                               blockSize=115, 
                               C=1)
    kernel = np.ones((3, 3), np.uint8)

    # perform erosion and dilation operations to improve components representing the 
    # crosswalk - limits noise present in image following binarization
    #TODO: figure out what this cv2.erode function does
    th = cv2.erode(th, kernel, iterations=1)
    th = cv2.dilate(th, kernel, iterations=2)
    
    # # kernel = np.ones((5, 5), np.uint8)
    # # th = cv2.erode(th, kernel, iterations=1)
    # # th = cv2.dilate(th, kernel, iterations=2)
    if(showStages==True):
        cv2.imshow('thresh',th)
        cv2.waitKey()
        cv2.destroyAllWindows()
        
    # contour method: RETR_TREE - retrieve all contours and full family hierarchy list
    # contour approximation method: CHAIN_APPROX_TC89_KCOS
        
    # find contours in the image 
    # NOTE: Removed 'image' result since cv2.findContours only returns the contours and hierarchy
    contours, hierarchy = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)

    # cv2.imshow('cont',ct)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    # im2, contours, hierarchy = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contIndex = 0
    allContours = []
    for contour in contours:
        M = cv2.moments(contour)
        if(cv2.contourArea(contour)>800):
            # print('length: ',len(contour))
            if(len(contour)<100):
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
                if(len(approx)==4):
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.drawContours(frame, contours, contIndex, (0, 255, 0), 3)
                    cv2.rectangle(temp,(x,y),(x+w,y+h),(255,0,0),2)
                    allContours.append((x,y,w,h))
        contIndex = contIndex + 1

    cv2.drawContours(temp2, contours, -1, (0, 255, 0), 3)
    if(showStages==True):
        cv2.imshow('frame',temp2)
        cv2.waitKey()
        cv2.destroyAllWindows()
        cv2.imshow('frame',frame)
        cv2.waitKey()
        cv2.destroyAllWindows()
        cv2.imshow('frame',temp)
        cv2.waitKey()
        cv2.destroyAllWindows()

    # cv2.imshow('x',frame)
    # cv2.waitKey()
    frame=temp
    minIndex = 0
    count=0
    minDistance = 10000000

    # find the detected cross walk to determine the one closest to the 
    # traffic light - this is treated as the correct detection line
    for rect in allContours:
        x,y,w,h = rect
        if(ylight+wlight<y):
         cv2.line(temp, (xlight,ylight), (x, y), (0, 0, 255), 2)
         if (((x-xlight)**2 + (y-ylight)**2)**0.5) < minDistance:
             minDistance = (((x-xlight)**2 + (y-ylight)**2)**0.5)
             minIndex=count
        count=count+1
    (x,y,w,h) = allContours[minIndex]
    if(showStages==True):
        cv2.imshow('with-dist',temp)
        cv2.waitKey()
        cv2.destroyAllWindows()
        cv2.line(temp, (0, y), (1300, y), (0, 0, 0), 4, cv2.LINE_AA)
        cv2.imshow('with-dist',temp)
        cv2.waitKey()
        cv2.destroyAllWindows()
    vss.release()

    # the y component of the detection line is returned as a representative
    # detection line for the image (if the center of a bounding box crosses this 
    # line, a violation should be inferred)
    return y

def updateTrackers(
        image, 
        trackersList:list, 
        thresholdRedLight, 
        iDsWithIoUList:list,
        redTrackers:list,
        displayCounter,
        recentlyViolated:list, 
        redLightViolatedCounter,
        redTrackingCounters:list, 
        ylight, 
        hlight, 
        xlight, 
        wlight, 
        frame, 
        ctr
    ):
    '''
    Takes as input the image, list of existing trackers, red light threshold, 
    and IDs with IOU (intersection over union between trackers and most recent
    bounding box detections), and updates the position of bounding boxes for 
    objects with existing image trackers in the current frame. Additionally, identifies
    if an existing tracker has become a violator, and adds them to the redTrackers 
    list if so.
    
    Args:
    - image --- 1000 x 750 RGB image which is passed to the tracker update method
    - trackersList --- list of image trackers for currently tracked objects. Will 
      be empty if no cars have yet been detected in the image
    - thresholdRedLight --- y (vertical) coordinate value in the image indicating the 
      horizontal detection line. I.e., if any vehicle pases this value, it is considered
      a Red Light Running violation
    - iDsWithIoUList --- a list of objects with following format 
      any item --- 
            [
                (NA), 
                Object ID,
                listWithFrame = [
                    Previous bounding box for object,
                    (NA)
                ],
                violationList
            ]
      If a violation is detected, this list is iterated through, performing an 
      IoU pass between the current frame and the list of frames in the tracker list.
      If there is a match (IoU val > 0.20), then the tracker is moved into the 
      RedLightTrackers list 
      
    Returns:
    - boxes --- list of bounding boxes for the new images
    - trackersList --- updated values
    - thresholdRedLight --- updated values
    - iDsWithIoUList --- updated values
    - redTrackers --- updated values
    - displayCounter --- updated values
    - recentlyViolated --- updated values
    - redLightViolatedCounter -- updated values
    - redTrackingCounters --- updated values
    '''
    print('\n[INFO]: --- BEGIN tracker updates ---')
    # global displayCounter
    # global redTrackers
    # global redLightViolatedCounter
    # global recentlyViolated
    # global redTrackingCounters
    boxes = []

    print(f"[INFO-513]: len of tracker list = {len(trackersList)}")
    for n, pair in enumerate(trackersList):
        tracker, box = pair

        success, bbox = tracker.update(image)

        if not success:
            del trackersList[n]
            continue

        boxes.append(bbox)  # Return updated box list


        xmin = int(bbox[0])
        ymin = int(bbox[1])
        xmax = int(bbox[0] + bbox[2])
        ymax = int(bbox[1] + bbox[3])
        xmid = int(round((xmin + xmax) / 2))
        ymid = int(round((ymin + ymax) / 2))
        light = frame[ylight:ylight + hlight, xlight:xlight + wlight]
        b, g, r = cv2.split(light)
        light = cv2.merge([r, g, b])

        # update tracker - car exceeds RLR violation threshold
        if(ymid < thresholdRedLight and trafficLightColor.estimate_label(light)=="red"):
            displayCounter = 10
            print(displayCounter)
            clone = image.copy()
            cv2.line(clone, (0, thresholdRedLight), (1300, thresholdRedLight), (0, 0, 0), 4, cv2.LINE_AA)

            print(trafficLightColor.estimate_label(light))
            cv2.rectangle(clone, (xmax, ymax), (xmin, ymin), (0, 255, 0), 2)
            print(redLightViolatedCounter)
            redLightViolatedCounter = redLightViolatedCounter + 1
            print(f"[INFO-547]: {trackersList[n]})")
            recentlyViolated.append((trackersList[n][1],10))
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print(redLightViolatedCounter)
            print(box)
            print("__")
            print(bbox)
            print("[INFO-563]: red light violation detected")
            #password
            # cv2.imshow('please', clone)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            print("___")
            # print(trackersList[n])

            tracker, box = trackersList[n]

            (xt, yt, wt, ht) = box # new bounding box 
            print(f"[INFO-574]: iDsWithIoUList(1) = {iDsWithIoUList}")
            print(f"[INFO-575]: len of this ^ list = {len(iDsWithIoUList)}")
            for n, item in enumerate(iDsWithIoUList):
                print(item)
                ___, id, listWithFrame, violationList = item
                print(listWithFrame)
                box2, __ = listWithFrame[len(listWithFrame) - 1]
                print(item)
                print(list)
                (x1, y1, w1, h1) = box2 # old bounding box for image n

                val = bb_intersection_over_union([xt, yt, xt + wt, yt + ht], [x1, y1, x1 + w1, y1 + h1])
                print("IoU -------")
                print(val)
                print("IoU_______")

                if (val > 0.20):
                    ___ = True
                    iDsWithIoUList[n] = (___, id, listWithFrame,[(box,ctr)])
                    break
                
            print(f"[INFO-593]: len of tracker list = {len(trackersList)}")
            print(f"[INFO-594]: iDsWithIoUList(2) = {iDsWithIoUList}")
            print(f"[INFO-595]: len of this ^ list = {len(iDsWithIoUList)}")
            tracker, box = trackersList[n]
            print(box)
            print("____")
            redTrackers.append(trackersList[n])
            redTrackingCounters.append(10)
            del trackersList[n]
            # print(box)
            # print("__")
            # print(bbox)
            # print("rab ah")

    # set of values to return to caller #MODIFIED 7/31/24 (og returns boxes only)
    return_list = [
        boxes, 
        trackersList, 
        thresholdRedLight, 
        iDsWithIoUList, 
        redTrackers, 
        displayCounter, 
        recentlyViolated, 
        redLightViolatedCounter, 
        redTrackingCounters
    ]
    print('[INFO]: --- END tracker updates ---\n')
    # here will check if it passes the red light
    return return_list

def updateRedTrackers(
    image, 
    redTrackers, 
    redTrackingCounters, 
    iDsWithIoUList,
    boxes, 
    ctr:int
):
    '''
    args: 
    - image ---
    - redTrackers ---
    - redTrackingCounters ---
    - iDsWithIoUList ---
    - boxes --- list of bounding boxes obtained
    - ctr ---
    '''
    clonedImage = image.copy()
    for n, pair in enumerate(redTrackers):
        tracker, box = pair
        success, bbox = tracker.update(image)

        # of the object is out of scope, delete it from the list of red-trackers
        if not success:
            del redTrackers[n]
            continue
        
        redTrackingCounters[n] = redTrackingCounters[n] - 1
        
        if(redTrackingCounters[n] > 0):
            (xt, yt, wt, ht) = bbox
            for n, item in enumerate(iDsWithIoUList):
                print(item)
                ___, id, listWithFrame, violationList = item

                if(___ == False):
                    continue
                print(listWithFrame)
                box2, __ = listWithFrame[len(listWithFrame) - 1]
                print(item)
                print(list)
                (x1, y1, w1, h1) = box2

                val = bb_intersection_over_union([xt, yt, xt + wt, yt + ht], [x1, y1, x1 + w1, y1 + h1])
                print("IoU -------")
                print(val)
                print("IoU_______")

                if (val > 0.20):
                    violationList.append(([bbox],ctr))
                    iDsWithIoUList[n] = (___, id, listWithFrame,violationList)
                    break

            boxes.append(bbox)  # Return updated box list

            xmin = int(bbox[0])
            ymin = int(bbox[1])
            xmax = int(bbox[0] + bbox[2])
            ymax = int(bbox[1] + bbox[3])
            xmid = int(round((xmin + xmax) / 2))
            ymid = int(round((ymin + ymax) / 2))

            cv2.rectangle(clonedImage, (xmax, ymax), (xmin, ymin), (0, 0, 255), 2)
    return clonedImage # we don't need this updated list it seems?

def add_object(image, box, trackersList):
    # NOTE: TrackerMedianFlow.create() is a legacy method in cv2 4.9.0.80
    tracker = cv2.legacy_TrackerMedianFlow.create()
    (x, y, w, h) = [int(v) for v in box]

    success = tracker.init(image, (x, y, w, h))

    if success:
        trackersList.append((tracker, (x, y, w, h)))

def rlr_detect(
    vs,                 ## the video reader to grab frames from for analysis
    args:dict,          ## dictionary of input arguments for the system
    fps,                ## number of frames per second to analyze?
    colors,             ## numpy array of color values 
    total,              ## total number of frames to analyze (perform inference on)
    big_net:str,        ## model to perform traffic light bbx inference with
    small_net:str,      ## model to perform car bbx inference with
    imgsize=(256,256),  ## size of image to rescale to
    secure:bool=False,  ## If true - perform secure inference, otherwise use plaintext  
    W=None,             ## Width specification for something?
    H=None,             ## height specification for something?
    writer=None,        ## video writer object ... (should be none)
    debug=False         ## debug bool for printing debug statements during run
):
    '''
    Args: 
    - None
    Returns:
    - None
    
    Loops through each of the videos (grabbable) from the video input stream. 
    If a video frame can't be grabbed, then the end of the stream has been reached
    and the process ends. 
    '''
    # initialize system parameters parameters
    listAll = []
    
    print("[INFO] Attempting to get light coordinates")
    xlight, ylight, wlight, hlight = setLightCoordinates(args=args, net_type=big_net, colors=colors, secure=secure, imgsize=(672,672), debug=debug)
    print("[INFO]: light coordinates\n\t x={} y={} w={} h={}".format(xlight, ylight, wlight, hlight))
    
    ctr = 0
    penaltyList = []
    redLightViolatedCounter = 0
    thresholdRedLight = getLightThresh(ylight=ylight, xlight=xlight, wlight=wlight, args=args, showStages=False)
    trackersList = []
    redTrackers = []
    recentlyViolated = []
    redTrackingCounters = []
    iDsWithIoUList = []
    idCounter = 0
    displayCounter = 0
    
    # load the primary inference network here ()
    net = torch.hub.load("ultralytics/yolov5", small_net, force_reload=True, trust_repo=True)
    labels = net.names.keys() ## load dictionary of names -> list of keys
    
    # set cost and other metric recording lists
    if secure: 
        com_costs = []
        round_nums = []
    run_times = []
    
    # start timer for program execution
    startTime = time.time()
    prevCurrentList = [] # TODO: Come back to this (not sure what it's for)
    
    while True:
        start = time.time()
        print(thresholdRedLight)
        (grabbed, frame) = vs.read()
        if (not grabbed):
            break
        else:
            frame = cv2.resize(frame, (1000, 750))
            
        # create copies of the current frame
        frameTemp2 = frame.copy() # used for image display purposes
        frameTemp3 = frame.copy() # used for red light tracker updates and writing to new frame
        
        # update red light violators and current trackers
        sys_params = updateTrackers(
                        frame, 
                        trackersList, 
                        thresholdRedLight,
                        iDsWithIoUList, 
                        redTrackers,
                        displayCounter, 
                        recentlyViolated, 
                        redLightViolatedCounter, 
                        redTrackingCounters, 
                        ylight=ylight, 
                        hlight=hlight, 
                        xlight=xlight, 
                        wlight=wlight, 
                        frame=frame, 
                        ctr=ctr
                    )
        
        # unpack results from updateTrackers()
        boxesTemp, trackersList, thresholdRedLight, iDsWithIoUList = sys_params[:4]
        redTrackers, displayCounter, recentlyViolated, redLightViolatedCounter, redTrackingCounters = sys_params[4:]
        
        print(f"[INFO]: (main loop) ctr = {ctr}")

        # every 5 frames, run the below detection code block
        if(ctr % 5== 0):
            # start timer for inference

            # frameTemp = imutils.resize(frame, width=600)
            frameTemp = frame.copy()

            cv2.line(frameTemp, 
                    (0, thresholdRedLight),
                    (1300, thresholdRedLight), 
                    (0, 0, 0), 
                    4, 
                    cv2.LINE_AA)
            
            # Draw the running total of cars in the image in the upper-left corner
            # boxesTemp = updateTrackers(frame)
            # (success, boxesTemp) = trackers.update(frame)
            
            print('tracker boxes : ')
            print(boxesTemp)
            print("___ tracked boxes done")
            # password

            # cv2.imshow("Temp Frame", frameTemp)
            #
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            for idx, box in enumerate(boxesTemp):
                (x, y, w, h) = [int(v) for v in box]
                cv2.rectangle(frameTemp, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # password
            # cv2.imshow("Temp Frame", frameTemp)
            #
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            # read the next frame from the file
            # print(ctr)
            # if the frame was not grabbed, then we have reached the end
            # of the stream
            if not grabbed:
                break

            # if the frame dimensions are empty, grab them
            if W is None or H is None:
                (H, W) = frame.shape[:2]

            # construct a blob from the input frame and then perform a forward
            # pass of the YOLO object detector, giving us our bounding boxes
            # and associated probabilities
            
            # get the stride from the loaded network object now
            im_tensor = read_im_frame(frame, img_size=imgsize, stride=net.stride).unsqueeze(0)
            
            # if running secure inference: save to a crypten .pth data file
            if secure: 
                cryp_folder = "experiments/crypten_tmp"
                if cryp_folder not in os.listdir("experiments"):
                    os.mkdir("experiments/{}".format(cryp_folder))
                torch.save(im_tensor, "experiments/{}/frame.pth".format(cryp_folder))
                
            # blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
            #     swapRB=True, crop=False)
            
            # # TODO: Change net input to yolo5 model - get it to work
            # net.setInput(blob)
            # start = time.time()
            
            if secure: # record data relevant to the inference process for the paper
                yolo_args = {
                    "world_size":2,
                    "img_size": imgsize, 
                    "model":net,
                    "data_path":"experiments/crypten_tmp/frame.pth", 
                    "run_label":0, 
                    "batch_size":im_tensor.shape[0],
                    "folder":"crypten_tmp", 
                    "device":'cpu',
                    "debug":False                    
                }
                multiproc_gpu(_run_sec_model, 'cpu_val', args=yolo_args) 
                with open("experiments/crypten_tmp/run_0.pkl", "rb") as f:
                    outputs, inf_start, inf_end = pickle.load(f)
                with open("experiments/crypten_tmp/comm_tmp_0.pkl", "rb") as com_0:
                    alice_com = pickle.load(com_0)
                with open("experiments/crypten_tmp/comm_tmp_1.pkl", "rb") as com_1:
                    bob_com = pickle.load(com_1)
                cost = (alice_com['bytes'] + bob_com['bytes'])/(2e6) # convert to MB
                round_vals = (alice_com['rounds'] + bob_com['rounds'])/2
                com_costs.append(cost)
                round_nums.append(round_vals)
                run_times.append(inf_end - inf_start)
            else:
                inf_start = time.time()
                outputs = net(im_tensor)  
                inf_end = time.time()
                run_times.append(inf_end - inf_start)

            # # TODO: Figure out what needs to be changed to get a forward pass of the 
            # # network to work as desired
            # layerOutputs = net.forward(ln)
            # end = time.time()

            # initialize our lists of detected bounding boxes, confidences,
            # and class IDs, respectively
            # boxes = []
            
            confidences = []
            classIDs = []

            # # NOTE: select valid bounding boxes by filtering raw yolo output
            # # relevant information appended to `boxes`, `confidences`, and `classIDs`
            # # loop over each of the layer outputs
            # for output in layerOutputs:
            #     # loop over each of the detections
            #     for detection in output:
            #         # extract the class ID and confidence (i.e., probability)
            #         # of the current object detection
            #         scores = detection[5:]
            #         classID = np.argmax(scores)
            #         confidence = scores[classID]

            #         # filter out weak predictions by ensuring the detected
            #         # probability is greater than the minimum probability
            #         if confidence > args["confidence"]:
            #             # scale the bounding box coordinates back relative to
            #             # the size of the image, keeping in mind that YOLO
            #             # actually returns the center (x, y)-coordinates of
            #             # the bounding box followed by the boxes' width and
            #             # height
            #             box = detection[0:4] * np.array([W, H, W, H])
            #             (centerX, centerY, width, height) = box.astype("int")

            #             # use the center (x, y)-coordinates to derive the top
            #             # and and left corner of the bounding box
            #             x = int(centerX - (width / 2))
            #             y = int(centerY - (height / 2))

            #             # update our list of bounding box coordinates,
            #             # confidences, and class IDs
            #             if(classID ==2 or classID==7 or classID ==3 or classID == 9):
            #                 boxes.append([x, y, int(width), int(height)])
            #                 confidences.append(float(confidence))
            #                 classIDs.append(classID)

            # # apply non-maxima suppression to suppress weak, overlapping
            # # bounding boxes
            # idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
            #     args["threshold"])

            currentBoxes = []
            preds = non_max_suppression(
                prediction = outputs, 
                conf_thres=args['confidence'], 
                iou_thres=args['iou_threshold'], 
                classes=[2,3,7,9], 
                agnostic=None, 
                max_det=1000
            )
            if len(preds) > 0:
                preds = preds[0] # index out the outermost layer
                for i in range(len(preds)):   
                     
                    # upscale b-boxes from the prediction size  
                    preds[i][:4] = scale_boxes(im_tensor.shape[2:], 
                                            preds[i][:4], 
                                            frameTemp.shape)
                    preds[i][:4] = xyxy2xywh(preds[i][:4])
                    if debug: print("[INFO]: pred = {}".format(preds[i]))
                    
                    # convert coordinates to integers for reading by opencv
                    (x,y) = (int(preds[i][0]), int(preds[i][1]))
                    (w,h) = (int(preds[i][2]), int(preds[i][3]))
                    x = int(x - (w/2)) # adjust coordinate positions properly
                    y = int(y - (h/2)) 
                    currentBoxes.append((x,y,w,h))
                    cv2.rectangle(frameTemp2, (x,y), (x+w, y+h), (255,0,0))
                
            # # ensure at least one detection exists
            # if len(idxs) > 0:
            #     # loop over the indexes we are keeping
            #     for i in idxs.flatten():
            #         # extract the bounding box coordinates
            #         (x, y) = (boxes[i][0], boxes[i][1])
            #         (w, h) = (boxes[i][2], boxes[i][3])

            #         # if ((y + y + h) / 2 > thresholdRedLight):
            #         currentBoxes.append((x,y,w,h))
            #         # draw a bounding box rectangle and label on the frame

            #         cv2.rectangle(frameTemp2, (x, y), (x + w, y + h), (255, 0, 0), 2)
            # if len(preds) > 0:
            #     for pred in preds:
            #         if debug: print("[INFO]: pred = {}".format(pred))
            #         (x,y) = (preds[0], preds[1])
            #         (w,h) = (preds[2], preds[3])
                    
            #         currentBoxes.append((x,y,w,h))
            #         cv2.rectangle(frameTemp2, (x,y), (x+w, y+h), (255,0,0), 2)

            #password
            # cv2.imshow("Current Frame", frame)
            # cv2.waitKey(0)
            # print('current boxes:')
            # print(len(currentBoxes))
            # cv2.destroyAllWindows()
            prevCurrentList = currentBoxes.copy()
            addedBoxes = []
            # print(addedBoxes)
            
            #---------------------
            # NOTE: loop through object tracker (updated) boxes (boxesTemp) and compare 
            # against the object detector boxes. If the IoU between any two boxes is greater 
            # than 0.25, we consider the detected box to be the correct, refined version of 
            # the object tracker box. If not, (and no matching detection box is found) the 
            # tracker which generated the tracker box in question is deleted from our list of
            # trackers and from the object tracker detected boxes. 
            # Additionally, a new object tracker is created (added_boxes maintains
            # a list of boxes which need to have a tracker created for them, and added to the 
            # tracker list - this replaces the old, less accurate tracker
            index = 0
            for idx, box in enumerate(boxesTemp):
                if (len(trackersList) == 0):
                    break
                i = 0
                (x, y, w, h) = [int(v) for v in box]
                print('iteration')
                print((x, y, w, h))
                flagg = False
                yt, ht = 0,0
                
                # look through all bbxs found by object detector and compare against tracker box
                # corresponding to idx
                for idx2, box2 in enumerate(currentBoxes):
                    (x2, y2, w2, h2) = [int(v2) for v2 in box2]
                    val = bb_intersection_over_union([x,y,x+w,y+h],[x2,y2,x2+w2,y2+h2])

                    print(val)

                    if (val > .25):
                        flagg = True
                        i = idx2
                        yt, ht = y2,h2
                        # print('yes')
                        # if(addedBoxes.__contains__((x2, y2, w2, h2))):
                        #     addedBoxes.remove((x2, y2, w2, h2))

                # if the object detector does not detect an object that matches a 
                # bounding box generated by the object tracker, then that tracker
                # is removed (no longer tracking that object)
                if (flagg == False):
                    del trackersList[index]
                    del boxesTemp[index]
                    index = index - 1 # ensures that index stays at 0 after loop finishes (increments every time)
                else:
                    print('INDEX DELETED',index)
                    print('Length',len(trackersList))
                    del trackersList[index]

                    # add the newly detected box with high overlap with 
                    # box at index idx
                    index = index - 1
                    # if ((yt + yt + ht) / 2 > thresholdRedLight - 1):
                    
                    # add the box for which there is an overlapping object 
                    # tracker box
                    addedBoxes.append(currentBoxes[i])

                    (xt,yt, wt, ht) = currentBoxes[i]
                    print(iDsWithIoUList)
                    
                    # iterate through bounding boxes with prev. frame IoU values and object IDs
                    # if the intersection between a previous object box and a detected box
                    # is high enough, that box will be added to the list of "previous frame detections"
                    # in the iDsWithIoUList 
                    for n,item in enumerate(iDsWithIoUList):
                        print(item)
                        ___,id,listWithFrame,violationList = item
                        print(listWithFrame)
                        box2, __ = listWithFrame[len(listWithFrame) - 1]
                        print(item)
                        print(list)
                        (x1, y1, w1, h1) = box2

                        val = bb_intersection_over_union([xt, yt, xt + wt, yt + ht], [x1, y1, x1 + w1, y1 + h1])
                        print("IoU -------")
                        print(val)
                        print("IoU_______")
                        if(val > 0.20):
                            listWithFrame.append((currentBoxes[i],ctr)) # (bounding box, frame number)
                            iDsWithIoUList[n] = (___, id, listWithFrame,violationList)
                            break # stop iteration once the matching box has been found (if it is)

                index = index + 1
            
            #----------------------
            # NOTE: loop through object detector found boxes and each of the object tracker boxes
            # after removing all of the boxes which have no detection box matches. If the intersection
            # between any two boxes is greater than 0.25, then continue (the tracker already exists). 
            # Otherwise, we assume that there is a new bounding box and add it to the list of bounding
            # boxes to track in the video feed
            for idx, box in enumerate(currentBoxes):
                (x, y, w, h) = [int(v) for v in box]
                flagg = False
                for box2 in boxesTemp:
                    (x2, y2, w2, h2) = [int(v2) for v2 in box2]
                    val = bb_intersection_over_union([x, y, x + w, y + h], [x2, y2, x2 + w2, y2 + h2])
                    print(val)

                    if (val > .25):
                        flagg = True
                fl = False
                if (flagg == False):
                    if ((y + y + h) / 2 > thresholdRedLight):
                        addedBoxes.append(box)
                        print(box)
                        # tuple=(__, ObjectId, [(bounding_box, frame_num)], violations_list)
                        iDsWithIoUList.append((False, idCounter, [(box,ctr)],[])) 
                        idCounter = idCounter + 1
            
            print("______________")
            print("iDs with IoU")
            for i in iDsWithIoUList:
                print(i)
            print("_______________")

            print(addedBoxes)
            print('enter input to continue')
            # varss = input()
            print(len(boxesTemp))
            print('TRACKER LIST LENGTH: ',len(trackersList))
            print(len(penaltyList))
            
            # append new bounding boxes to the trackersList for the next
            # detection pass
            for box in addedBoxes:
                add_object(frameTemp2,box,trackersList)
                penaltyList.append(0)
            print("_____________")
            print(len(trackersList))
            print(len(penaltyList))


            print('enter input to continue')
            # varss = input()
            # check if the video writer is None
        
        # ---------------------------------------
        # NOTE: End of object detection component
        # ---------------------------------------
            
        # update object trackers tracking red light violators
        frameTemp3 = updateRedTrackers(
                        frameTemp3, 
                        redTrackers, 
                        redTrackingCounters, 
                        iDsWithIoUList, 
                        currentBoxes, 
                        ctr
                     )

        # if the y-position of the traffic light is set ...
        # check the color of the light, and estimate whether the light
        # is green or not. 
        if(ylight!=None):
            light = frame[ylight:ylight + hlight, xlight:xlight + wlight]
            b, g, r = cv2.split(light)
            light=cv2.merge([r,g,b])
            greenCount = 0
            if(trafficLightColor.estimate_label(light)=="green"):
                print('GREEEENNNN')
            cv2.imwrite('green.png',light)
            # cv2.waitKey()
            cv2.putText(frameTemp3, 
                        trafficLightColor.estimate_label(light), 
                        (xlight, ylight),
                        cv2.FONT_HERSHEY_DUPLEX, 
                        1.5, 
                        (255, 255, 255), 
                        2, 
                        cv2.LINE_AA)

        cv2.line(frameTemp3,
                (0, thresholdRedLight), 
                (1300, thresholdRedLight), 
                (0, 0, 0), 
                4, 
                cv2.LINE_AA)
        cv2.putText(frameTemp3, 
                    'Violation Counter: ' + str(redLightViolatedCounter), 
                    (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1.5, 
                    (0, 255, 255), 
                    4, 
                    cv2.LINE_AA)

        # if a violation occurs, it will be updated by the update trackers function
        # performed at the beginning of the main while loop
        if (displayCounter != 0):
            cv2.putText(frameTemp3, 'Violation', (30, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4, cv2.LINE_AA)
            displayCounter = displayCounter - 1
            
        # iterate through object tracker boxes and add frames to image which 
        # correspond to those trackers in frameTemp3. At this point, all of the 
        # dead bounding boxes - removed by the YOLO pass will have been destroyed
        # so only valid boxes should remain. 
        for idx, box in enumerate(boxesTemp):
            (x, y, w, h) = [int(v) for v in box]
            cv2.rectangle(frameTemp3, (x, y), (x + w, y + h), (0, 255, 0), 2)

        end = time.time()
        if writer is None:
            # initialize our video writer

            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(
                        args["output"], 
                        fourcc, 
                        fps,
                        (frameTemp3.shape[1], frameTemp3.shape[0]), 
                        True
                    )

            # some information on processing single frame
            if total > 0:
                elap = (end - start)
                print("[INFO] single frame took {:.4f} seconds".format(elap))
                print("[INFO] estimated total time to finish: {:.4f}".format(
                    elap * total))
        #password
        # write the output frame to disk
        # cv2.imshow('OUTPUT',frameTemp2)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        writer.write(frameTemp3)

        # cv2.imshow('frame',frameTemp3)
        # cv2.waitKey()
        print(f"[INFO]: (main loop) finished loop {ctr}")
        ctr = ctr + 1
        print(ctr)
        
    print(f"[INFO]: finished exec (main loop)")

    # release the file pointers
    print("[INFO] cleaning up...")
    writer.release()
    vs.release()
    endTime = time.time()
    print('Total Time: ', endTime - startTime)

def run(
    model_type='yolov5s', 
    confidence=0.5, 
    iou_thres=0.3, 
    video_path='../../Fully-Automated-red-light-Violation-Detection/videos/aziz2.MP4', 
    n_labels=80
):    
    vid_name = video_path.split("/")[-1] # get file name
    vid_name = vid_name.split(".")[0]    # get file title no ext.
    
    args = {
        "input": video_path, ## the second input value
        "output": "experiments/video_output/{}.avi".format(vid_name), 
        "confidence": confidence, 
        "iou_threshold": iou_thres, 
    }
    
    # initialize a list of colors to represent each possible class label
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(n_labels, 3),
        dtype="uint8")

    # initialize the video stream, pointer to output video file, and
    # frame dimensions
    vs = cv2.VideoCapture(args["input"])
    fps = vs.get(cv2.CAP_PROP_FPS)
    writer = None
    (W, H) = (None, None)

    # try to determine the total number of frames in the video file
    try:
        prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
            else cv2.CAP_PROP_FRAME_COUNT
        total = int(vs.get(prop))
        print("[INFO] {} total frames in video".format(total))

    # an error occurred while trying to determine the total
    # number of frames in the video file
    except:
        print("[INFO] could not determine # of frames in video")
        print("[INFO] no approx. completion time can be provided")
        total = -1
        
    rlr_detect(
        vs, 
        args, 
        fps, 
        colors,
        total,
        big_net='yolov5s', 
        small_net='yolov5s',
        imgsize=(256,256), 
        secure=False, 
        W=W, 
        H=H, 
        writer=writer, 
        debug=True
    )

if __name__ == "__main__":
    run() ## run main function