import cv2
from pathlib import Path
import os
import numpy as np

_INPUT_DIR = Path(__file__).parents[2]/"data"/"input"

def detect_document_edges(image_path):
    image = cv2.imread(image_path)
    original = image.copy()

    """grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.namedWindow('grayscale', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('grayscale', grayscale.shape[1], grayscale.shape[0])
    cv2.imshow('Grayscale', grayscale)"""

    """blurred = cv2.GaussianBlur(image, (5, 5), 0)
    cv2.resizeWindow('Blurred', blurred.shape[1], blurred.shape[0])
    cv2.namedWindow('Blurred', cv2.WINDOW_NORMAL)
    cv2.imshow('Blurred', blurred)"""

    """_, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.resizeWindow('Binary', binary.shape[1], binary.shape[0])
    cv2.namedWindow('Binary', cv2.WINDOW_NORMAL)
    cv2.imshow('Binary', binary)"""

    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.namedWindow('Grayscale', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Grayscale', 800, 600)

    cv2.imshow('Grayscale', grayscale)


    blurred = cv2.GaussianBlur(grayscale, (5, 5), 0)
    cv2.namedWindow('Blurred', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Blurred', 800, 600)

    cv2.imshow('Blurred', blurred)

    scan = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    canny = cv2.Canny(scan, 50, 150)
    cv2.namedWindow('Canny edge detection', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Canny edge detection', 800, 600)
    cv2.imshow('Canny edge detection', canny)

    contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        if len(approx) == 4:
            doc_contour = approx
            break
    else:
        print("No document detected.")
        return None

    cv2.drawContours(original, [doc_contour], -1, (0, 255, 0), 10, cv2.LINE_AA)

    cv2.namedWindow("Detected document contour", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Detected document contour", 800, 600)
    cv2.imshow("Detected document contour", original)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

detect_document_edges(_INPUT_DIR/"IMG_20260617_192347.jpg")