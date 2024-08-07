import cv2
import numpy as np

def adjust_brightness_contrast(image, beta=50, alpha=1.5):
    # افزایش روشنایی و کنتراست تصویر
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

def gamma_correction(image, gamma=2.0):
    # تبدیل تصویر به فضای رنگی YUV
    yuv_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    
    # اعمال تصحیح گاما بر روی کانال Y
    yuv_image[..., 0] = np.clip(np.power(yuv_image[..., 0] / 255.0, gamma) * 255.0, 0, 255).astype(np.uint8)
    
    # تبدیل مجدد به فضای رنگی BGR
    return cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR)

def apply_clahe(image):
    # تبدیل تصویر به فضای رنگی خاکستری
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # ایجاد شیء CLAHE و تنظیم پارامترها
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    
    # اعمال CLAHE بر روی تصویر خاکستری
    clahe_image = clahe.apply(gray_image)
    
    # تبدیل تصویر خاکستری به BGR
    return cv2.cvtColor(clahe_image, cv2.COLOR_GRAY2BGR)

def enhance_details(image):
    # استفاده از sharpening filter برای افزایش جزئیات
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpened_image = cv2.filter2D(image, -1, kernel)
    
    return sharpened_image

def convert_to_hsv(image):
    # تبدیل تصویر به فضای رنگی HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # افزایش روشنایی در کانال V
    hsv_image[..., 2] = cv2.convertScaleAbs(hsv_image[..., 2], alpha=1.5, beta=30)
    
    # تبدیل مجدد به فضای رنگی BGR
    return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

def improve_low_light(image):
    # تبدیل به فضای رنگی HSV برای افزایش روشنایی
    hsv_image = convert_to_hsv(image)
    
    # تنظیم روشنایی و کنتراست
    bright_image = adjust_brightness_contrast(hsv_image, beta=50, alpha=1.5)
    
    # تصحیح گاما
    gamma_corrected_image = gamma_correction(bright_image, gamma=2.0)
    
    # اعمال CLAHE برای افزایش کنتراست محلی
    clahe_image = apply_clahe(gamma_corrected_image)
    
    # افزایش جزئیات
    detailed_image = enhance_details(clahe_image)
    
    return detailed_image

def main():
    # اتصال به دوربین (0 برای دوربین پیش‌فرض)
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
