#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 22:20:23 2024

@author: ysnrfd
"""

import cv2
import numpy as np
from skimage import exposure, img_as_float
from scipy.ndimage import gaussian_filter

def enhance_image(image):
    # تبدیل تصویر به فضای رنگی خاکستری
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # نرمال‌سازی و تبدیل به float
    gray_image_float = img_as_float(gray_image)
    
    # تبدیل لگاریتمی برای افزایش کنتراست
    log_transformed = np.log1p(gray_image_float * 255)
    
    # نرمال‌سازی شدت تصویر
    log_transformed = exposure.rescale_intensity(log_transformed, in_range='image')
    
    # استفاده از فیلتر گوسی برای کاهش نویز
    denoised_image = gaussian_filter(log_transformed, sigma=1)
    
    # بازگشت به دامنه اصلی تصویر
    enhanced_image = (denoised_image * 255).astype(np.uint8)
    
    # استفاده از افزایش کنتراست هیستوگرام
    enhanced_image = exposure.equalize_hist(enhanced_image) * 255
    enhanced_image = enhanced_image.astype(np.uint8)
    
    # تبدیل تصویر به فضای رنگی BGR
    enhanced_image = cv2.cvtColor(enhanced_image, cv2.COLOR_GRAY2BGR)
    
    return enhanced_image

# باز کردن دوربین (عدد 0 برای دوربین پیش‌فرض)
cap = cv2.VideoCapture(0)

while True:
    # خواندن یک فریم از دوربین
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # بهبود تصویر
    enhanced_frame = enhance_image(frame)
    
    # نمایش تصویر بهبود یافته
    cv2.imshow('Night Vision', enhanced_frame)
    
    # خروج از حلقه با زدن کلید 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# آزادسازی منابع
cap.release()
cv2.destroyAllWindows()
