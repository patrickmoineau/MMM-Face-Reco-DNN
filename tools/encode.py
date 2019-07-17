# USAGE
## Create dataset using camera
# python3 encode.py --dataset ../dataset --add --name "user1" --photocount 20
## Encode dataset
# python3 encode.py --dataset ../dataset --encodings encodings.pickle --execute
## Create Dataset and encode
# python3 encode.py -a -n "user1" -x

# import the necessary packages
from imutils import paths
import face_recognition
import argparse
import pickle
import cv2
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", required=False, default="../dataset/",
	help="path to input directory of faces + images")
ap.add_argument("-e", "--encodings", required=False, default="encodings.pickle",
	help="path to serialized db of facial encodings")
ap.add_argument("-d", "--detection-method", type=str, default="hog",
	help="face detection model to use: either `hog` or `cnn`")
ap.add_argument("-n", "--name", required=False, default="no_name",
    help="name of person to add")
ap.add_argument("-p", "--usePiCamera", type=int, required=False, default=0,
	help="Is using picamera or builtin/usb cam")
ap.add_argument("-c", "--photocount", required=False, type=int, default=10,
    help="number of photos of face")
ap.add_argument("-a", "--add", required=False, action='store_true',
    help="add new user")
ap.add_argument("-s","--show", required=False, action='store_true',
    help="show cv2 window")
ap.add_argument("-x","--execute", required=False, action='store_true',
    help="execute encoding")
args = vars(ap.parse_args())

def encodeDataset():
    # grab the paths to the input images in our dataset
    print("[INFO] quantifying faces...")
    imagePaths = list(paths.list_images(args["dataset"]))
    # initialize the list of known encodings and known names
    knownEncodings = []
    knownNames = []
    # loop over the image paths
    for (i, imagePath) in enumerate(imagePaths):
        # extract the person name from the image path
        print("[INFO] processing image {}/{}".format(i + 1,
            len(imagePaths)))
        name = imagePath.split(os.path.sep)[-2]
        # load the input image and convert it from RGB (OpenCV ordering)
        # to dlib ordering (RGB)
        image = cv2.imread(imagePath)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # detect the (x, y)-coordinates of the bounding boxes
        # corresponding to each face in the input image
        boxes = face_recognition.face_locations(rgb,
            model=args["detection_method"])
        # compute the facial embedding for the face
        encodings = face_recognition.face_encodings(rgb, boxes)
        # loop over the encodings
        for encoding in encodings:
            # add each encoding + name to our set of known names and
            # encodings
            knownEncodings.append(encoding)
            knownNames.append(name)
    # dump the facial encodings + names to disk
    print("[INFO] serializing encodings...")
    data = {"encodings": knownEncodings, "names": knownNames}
    f = open(args["encodings"], "wb")
    f.write(pickle.dumps(data))
    f.close()

def cam_to_image():
    print("[INFO] Starting Cam to Image...")
	#load xml for face detection 
    faceCascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
	#set save dir for dataset
    save_dir_root = os.path.abspath(args["dataset"])
	#set name, used to create folder structure 
    name = args["name"]
	#check and create folder structure
    if not os.path.isdir(save_dir_root):
        os.mkdir(save_dir_root)
    save_dir = os.path.join(save_dir_root, name)
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
	#set number of photos to take once a face is detected
    photocount = args["photocount"]
    count = 0
	#set video capture to use camera, default is 0
    cap = cv2.VideoCapture(args["usePiCamera"])
	#video loop
    while True:
        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		#detects if faces are in frame
        faces = faceCascade.detectMultiScale(
            gray,     
            scaleFactor=1.2,
            minNeighbors=5,     
            minSize=(200, 200),
            maxSize=(300, 300)
        )
		#loop when faces are detected
        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
			#crops face
            roi_gray = gray[y:y+h, x:x+w]
			#saves png in [dataset]\[name]\
            filename = os.path.join(save_dir, 'frame{}.png'.format(count)) #set filename
            print("[INFO] Saving {}".format(filename))
            cv2.imwrite(filename, roi_gray) # saving
            count += 1
			#if show set to try will display cv2 window with image
            if args["show"]:
                cv2.imshow('face', roi_gray)
		#if show set to try will display cv2 window with image
        if args["show"]:
            cv2.imshow('video',img)
            k = cv2.waitKey(30) & 0xff
            if k == 27: # press 'ESC' to quit
                break
		#once count is reached, end loop
        if count >= photocount:
            break
	#cleanup capture
    print("[INFO] Cleaning up Cam to Image...")
    cap.release()
	#clean up cv2 window if show is true
    if args["show"]:
        cv2.destroyAllWindows()
    print("[INFO] Done with Cam to Image.")
    

def main():
	#if -a or --add then run cam_to_image
    if args["add"]:
        cam_to_image()
	#if -x or --execute then encode dataset
    if args["execute"]:
        encodeDataset()

if __name__ == "__main__":
    main()