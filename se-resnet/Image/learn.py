import albumentations as a
import cv2 as cv

img = cv.imread('7b4bc5d7bc1c7a6a9f4ca449d9b3a4a2.jpg')
# mask = cv.imread('7b4bc5d7bc1c7a6a9f4ca449d9b3a4a2.jpg', cv.IMREAD_GRAYSCALE)

light = a.Compose(
    [a.RandomBrightnessContrast(p=1),
     a.RandomGamma(p=1),
     a.CLAHE(p=1)], p=1
)

medium = a.Compose(
    [a.CLAHE(p=1), a.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=50, val_shift_limit=50, p=1)], p=1
)

strong = a.Compose([a.ChannelShuffle(p=1)], p=1)

test = a.Compose(
    [a.RandomCrop(224, 224),
     a.ShiftScaleRotate(),
     a.RGBShift(),
     a.Blur()]
)

underwater = a.Compose(
    [a.MotionBlur(blur_limit=6), a.GaussianBlur(blur_limit=6), a.RandomRotate90(), a.RandomCrop(224, 224),
     a.RandomBrightnessContrast(p=1), a.RandomGamma(p=1)
     ], p=1)

augmented_light = light(image=img)
augmented_medium = medium(image=img)
augmented_strong = strong(image=img)
my_augmente = underwater(image=img)
# img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

cv.imshow('origin', img)
print(augmented_medium)
cv.imshow('augmented', augmented_light['image'])

cv.waitKey(0)
# cv.imshow('result', result)
