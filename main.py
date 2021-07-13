from scanner import scanner
from object_detection import yolov5
from ocr import crop_line_and_ocr

pathImage = 'data'

images = scanner.scanner(pathImage)
obj = yolov5.object_detection(images)
dataframe = crop_line_and_ocr.craft_and_ocr(obj)

dataframe.to_csv('data.csv', index = True, header=True, encoding = 'utf-8')

print(dataframe)