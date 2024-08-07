import cv2
import numpy as np
from skimage import exposure, filters

def adjust_brightness_contrast_gray(image, alpha=2.0, beta=50):
    """افزایش روشنایی و کنتراست تصویر خاکستری"""
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

def gamma_correction_gray(image, gamma=2.0):
    """تصحیح گاما برای تنظیم روشنایی و کنتراست کلی در تصاویر خاکستری"""
    image = image / 255.0
    image = np.power(image, gamma)
    return (image * 255).astype(np.uint8)

def apply_clahe_gray(image):
    """اعمال CLAHE برای بهبود کنتراست محلی در تصاویر خاکستری"""
    clahe = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(1, 1))
    clahe_image = clahe.apply(image)
    return clahe_image

def enhance_details_gray(image):
    """افزایش جزئیات با استفاده از فیلتر شارپنینگ در تصاویر خاکستری"""
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpened_image = cv2.filter2D(image, -1, kernel)
    return sharpened_image

def reduce_noise_gray(image):
    """کاهش نویز با استفاده از فیلتر نهایی"""
    return cv2.fastNlMeansDenoising(image, None, h=10, templateWindowSize=7, searchWindowSize=21)

def improve_low_light_gray(image):
    """بهبود تصویر خاکستری در شرایط نور کم"""
    # مرحله 1: افزایش روشنایی و کنتراست اولیه
    bright_image = adjust_brightness_contrast_gray(image, alpha=5.0, beta=10)
    
    # مرحله 2: تصحیح گاما
    gamma_corrected_image = gamma_correction_gray(bright_image, gamma=1.0)
    
    # مرحله 3: افزایش کنتراست با CLAHE
    clahe_image = apply_clahe_gray(gamma_corrected_image)
    
    # مرحله 4: افزایش جزئیات با فیلتر شارپنینگ
    detailed_image = enhance_details_gray(clahe_image)
    
    # مرحله 5: کاهش نویز
    denoised_image = reduce_noise_gray(detailed_image)
    
    return denoised_image

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

        # تبدیل فریم به خاکستری
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # بهبود تصویر خاکستری در شرایط نور کم
        improved_frame = improve_low_light_gray(gray_frame)

        # نمایش تصویر اصلی و تصویر بهبود یافته
        cv2.imshow('Original Frame', gray_frame)
        cv2.imshow('Improved Frame', improved_frame)

        # خروج از برنامه با فشار دادن کلید 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # آزادسازی منابع و بستن پنجره‌ها
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
