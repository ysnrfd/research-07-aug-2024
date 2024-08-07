#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 19:32:54 2024

@author: ysnrfd
"""

import cv2
import numpy as np
from skimage import exposure

def adjust_brightness_contrast(image, alpha=2.0, beta=50):
    """افزایش روشنایی و کنتراست تصویر"""
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

def gamma_correction(image, gamma=2.0):
    """تصحیح گاما برای تنظیم روشنایی و کنتراست کلی"""
    image = image / 255.0
    image = np.power(image, gamma)
    return (image * 255).astype(np.uint8)

def apply_clahe(image):
    """اعمال CLAHE برای بهبود کنتراست محلی"""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)

def enhance_low_light(image):
    """بهبود تصویر در شرایط کم‌نور"""
    # مرحله 1: تبدیل به خاکستری
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # مرحله 2: افزایش روشنایی و کنتراست
    bright_contrast_image = adjust_brightness_contrast(gray_image, alpha=1.5, beta=70)
    
    # مرحله 3: تصحیح گاما
    gamma_corrected_image = gamma_correction(bright_contrast_image, gamma=5)
    
    # مرحله 4: افزایش کنتراست محلی با CLAHE
    clahe_image = apply_clahe(gamma_corrected_image)
    
    return clahe_image

def main():
    """اتصال به دوربین و نمایش تصویر بهبود یافته"""
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("خطا: دوربین باز نشد!")
        return

    while True:
        # خواندن فریم از دوربین
        ret, frame = cap.read()

        if not ret:
            print("خطا: نمی‌توان فریم را خواند!")
            break

        # بهبود تصویر در شرایط کم‌نور
        improved_frame = enhance_low_light(frame)

        # نمایش تصویر اصلی و تصویر بهبود یافته
        cv2.imshow('Original Frame', cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        cv2.imshow('Improved Frame', improved_frame)

        # خروج از برنامه با فشار دادن کلید 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # آزادسازی منابع و بستن پنجره‌ها
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
