import cv2
from cvzone.HandTrackingModule import HandDetector
import cvzone
import numpy as np


cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
detector = HandDetector(detectionCon=0.8)


class DragRect():
    lockR = False

    def __init__(self, posCenter, size=[200,200], colorR = (21, 0, 232)):
        self.posCenter = posCenter
        self.size = size
        self.colorR = colorR

    def update(self, cursor):
        cx, cy = self.posCenter
        w, h = self.size

        if (cx - w // 2 < cursor[0] < cx + w // 2 and
            cy - h // 2 < cursor[1] < cy + h // 2):
            self.colorR = 246, 181, 0
            self.posCenter = cursor
            self.lockR = True

    def updateColor(self, colorR = (21, 0, 232)):
        self.colorR = colorR


rectList = []
for x in range(5):
    rectList.append(DragRect([x*250 + 150, 150]))

while True:
    success, img = cap.read()
    img = cv2.flip(img,1)
    img = detector.findHands(img)
    lmList, _ = detector.findPosition(img)

    if lmList:
        l, _, _ = detector.findDistance(8, 12, img, draw = False)
        if l < 40:
            cursor = lmList[8] # index finger tip landmark
            for rectI in rectList:  
                rectI.update(cursor)
        else:
            for rectI in rectList:
                rectI.updateColor((21, 0, 232))


    # Draw solid
    # for rectI in rectList:
    #     cx, cy = rectI.posCenter
    #     w, h = rectI.size
    #     cv2.rectangle(img, (cx - w//2, cy - h//2), (cx + w//2, cy + h//2), colorR, cv2.FILLED)
    #     cvzone.cornerRect(img, (cx - w//2, cy - h//2, w, h),20, rt=0)


    # Draw Transparent
    imgNew = np.zeros_like(img, np.uint8)
    for rectI in rectList:
        cx, cy = rectI.posCenter
        w, h = rectI.size
        colorR = rectI.colorR
        cv2.rectangle(imgNew, (cx - w//2, cy - h//2), (cx + w//2, cy + h//2), colorR, cv2.FILLED)
        cvzone.cornerRect(imgNew, (cx - w//2, cy - h//2, w, h),20, rt=0)

    out = img.copy()
    alpha = 0.01
    mask = imgNew.astype(bool)
    out[mask] = cv2.addWeighted(img, alpha, imgNew, 1 - alpha, 0)[mask]

    cv2.imshow("image", out)
    cv2.waitKey(1)