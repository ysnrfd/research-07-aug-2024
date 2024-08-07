import cv2
import numpy as np

# تابع برای شبیه‌سازی تصویر مادون قرمز
def simulate_ir(image):
    # تبدیل تصویر به فضای رنگی خاکستری
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # معکوس کردن تصویر خاکستری برای شبیه‌سازی تصویر مادون قرمز
    ir_image = cv2.bitwise_not(gray_image)
    
    return ir_image

# باز کردن دوربین (عدد 0 نشان‌دهنده دوربین پیش‌فرض است)
cap = cv2.VideoCapture(0)

while True:
    # خواندن فریم از دوربین
    ret, frame = cap.read()
    
    if not ret:
        print("خطا در خواندن فریم")
        break
    
    # شبیه‌سازی تصویر مادون قرمز
    ir_frame = simulate_ir(frame)
    
    # نمایش تصویر اصلی و شبیه‌سازی شده
    cv2.imshow('Original Frame', frame)
    cv2.imshow('Simulated IR Frame', ir_frame)
    
    # خروج از حلقه با فشار دادن کلید 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# آزاد کردن دوربین و بستن تمام پنجره‌ها
cap.release()
cv2.destroyAllWindows()
