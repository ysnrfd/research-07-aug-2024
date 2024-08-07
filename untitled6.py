import cv2
import numpy as np

def main():
    # ایجاد شیء VideoCapture برای دسترسی به دوربین
    cap = cv2.VideoCapture(0)

    # ایجاد مدل پس‌زمینه با استفاده از MOG2
    backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)

    if not cap.isOpened():
        print("خطا در باز کردن دوربین")
        return

    while True:
        # خواندن فریم از دوربین
        ret, frame = cap.read()

        if not ret:
            print("خطا در خواندن فریم")
            break

        # پردازش تصویر با مدل پس‌زمینه
        fgMask = backSub.apply(frame)

        # استفاده از فیلتر مورفولوژیکی برای بهبود نتایج
        kernel = np.ones((5, 5), np.uint8)
        fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_CLOSE, kernel)
        fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_OPEN, kernel)

        # پیدا کردن کانتورهای موجود در تصویر باینری
        contours, _ = cv2.findContours(fgMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # کشیدن مربع دور هر کانتور
        for contour in contours:
            if cv2.contourArea(contour) > 500:  # برای فیلتر کردن کوچکترین کانتورها
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # نمایش فریم اصلی و ماسک پس‌زمینه
        cv2.imshow('Frame', frame)
        cv2.imshow('Foreground Mask', fgMask)

        # خروج از حلقه با فشار دادن کلید 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # آزادسازی منابع
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
