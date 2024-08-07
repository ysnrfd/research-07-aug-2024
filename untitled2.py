import cv2
import numpy as np
from skimage import exposure, filters
from scipy.ndimage import gaussian_filter

def adjust_brightness_contrast(image, beta=50, alpha=1.5):
    """افزایش روشنایی و کنتراست تصویر"""
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

def gamma_correction(image, gamma=2.0):
    """تصحیح گاما برای تنظیم روشنایی و کنتراست کلی"""
    yuv_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    yuv_image[..., 0] = np.clip(np.power(yuv_image[..., 0] / 255.0, gamma) * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR)

def apply_clahe(image):
    """اعمال CLAHE برای بهبود کنتراست محلی"""
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    clahe_image = clahe.apply(gray_image)
    return cv2.cvtColor(clahe_image, cv2.COLOR_GRAY2BGR)

def convert_to_hsv(image):
    """تبدیل تصویر به فضای رنگی HSV و افزایش روشنایی"""
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_image[..., 2] = cv2.convertScaleAbs(hsv_image[..., 2], alpha=1.5, beta=30)
    return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

def enhance_details(image):
    """افزایش جزئیات با استفاده از فیلتر شارپنینگ"""
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpened_image = cv2.filter2D(image, -1, kernel)
    return sharpened_image

def noise_reduction(image):
    """کاهش نویز با استفاده از فیلتر وینزر"""
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised_image = cv2.fastNlMeansDenoising(gray_image, None, h=10, templateWindowSize=7, searchWindowSize=21)
    return cv2.cvtColor(denoised_image, cv2.COLOR_GRAY2BGR)

def enhance_contrast(image):
    """افزایش کنتراست با استفاده از تکنیک‌های پیشرفته"""
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equalized_image = exposure.equalize_hist(gray_image)
    return cv2.cvtColor((exposure.rescale_intensity(equalized_image, out_range=(0, 255))).astype(np.uint8), cv2.COLOR_GRAY2BGR)

def improve_low_light(image):
    """بهبود تصویر در شرایط نور کم"""
    # مرحله 1: افزایش روشنایی اولیه
    bright_image = adjust_brightness_contrast(image, beta=50, alpha=1.5)
    
    # مرحله 2: تصحیح گاما
    gamma_corrected_image = gamma_correction(bright_image, gamma=2.0)
    
    # مرحله 3: افزایش کنتراست با CLAHE
    clahe_image = apply_clahe(gamma_corrected_image)
    
    # مرحله 4: تبدیل به فضای رنگی HSV و افزایش روشنایی
    hsv_image = convert_to_hsv(clahe_image)
    
    # مرحله 5: افزایش جزئیات با فیلتر شارپنینگ
    detailed_image = enhance_details(hsv_image)
    
    # مرحله 6: کاهش نویز (اگر نیاز باشد)
    # denoised_image = noise_reduction(detailed_image)
    
    # مرحله 7: افزایش کنتراست نهایی
    final_image = enhance_contrast(detailed_image)
    
    return final_image

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

        # بهبود تصویر در شرایط نور کم
        improved_frame = improve_low_light(frame)

        # نمایش تصویر اصلی و تصویر بهبود یافته
        cv2.imshow('Original Frame', frame)
        cv2.imshow('Improved Frame', improved_frame)

        # خروج از برنامه با فشار دادن کلید 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # آزادسازی منابع و بستن پنجره‌ها
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
