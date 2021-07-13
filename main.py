import scanner
import yolov5
import crop_line_and_ocr
import sys, getopt

pathImage = ''

def main(argv):
  global pathImage
  try:
    opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
  except getopt.GetoptError:
    print('main.py -i <inputfile>')
    sys.exit(2)
  for opt, arg in opts:
    if opt == '-h':
        print('main.py -i <inputfile>')
        sys.exit()
    elif opt in ("-i", "--ifile"):
        pathImage = arg

if __name__ == "__main__":
  main(sys.argv[1:])

  images = scanner.scanner(pathImage)
  obj = yolov5.object_detection(images)
  dataframe = crop_line_and_ocr.craft_and_ocr(obj)

  dataframe.to_csv('data.csv', index = True, header=True, encoding = 'utf-8')

  dataframe.head()
