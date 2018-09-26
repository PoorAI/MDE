import cv2
import glob
import random
import numpy as np
import os

class ColorDistribution:

    # Contains definitions and methods to calculate the ColorDistribution of given images
    # This is just a simple test
    # This is another test
    def global_cd(self, image_path):
        # Calculate one color-histogram for the entire image
        # image parameter is the path to an actual image
        image = cv2.imread(image_path)
        # Maybe tweak some of these parameters
        hist = cv2.calcHist([image], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
        return hist

    def local_cd(self, image_path):
        print("Calculate Histogram Using QuadTree Decomposition")

    def closest_hist(self, image_path, path_container, method):
        # Find and display the picture with the closest color distribution to a given picture
        # Returns a list of paths to images with decreasing "similarity"
        ref_hist = self.global_cd(image_path)
        hist_container = {}
        if method == 0:
            for item in path_container:
                curr_hist = self.global_cd(item)
                # 3 means Bacc. Distance of two seperate pictures
                hist_container[item] = cv2.compareHist(ref_hist, curr_hist, 3)
            hist_list = sorted(hist_container, key=hist_container.get)
            new_img = cv2.imread(hist_list[0])
            cv2.imshow("Picture", new_img)
        elif method == 1:
            print("Method with QuadTree Decomp")
        return hist_list

    def create_Hists(self, path_container, method):
        # Generate a list of Histograms for a given dataset of images
        # TODO: Store them in a database
        # Ggf. als dictionary implementieren Path: Histogram (uebesetzt also Bild -> Histogram)
        hist_list = []
        if method == 0:
            for item in path_container:
                hist_list.append(self.global_cd(item))
            return hist_list
        elif method == 1:
            print("Method with QuadTree Decomp")
        return hist_list


class Emotions:

    # Contains definitions and method to recognize the emotions of entities on a given picture

    def rec_face(self, image_path):
        print("Recoginze the face on a given image to use for classification")
        # Classifier
        faceDet = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        faceDet_two = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
        faceDet_three = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
        faceDet_four = cv2.CascadeClassifier("haarcascade_frontalface_alt_tree.xml")
        # Start of actual method
        image = cv2.imread(image_path) # Read the picture
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # Image to grayscale
        face = faceDet.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
        face_two = faceDet_two.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
        face_three = faceDet_three.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
        face_four = faceDet_four.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
        # Go over all classifiers, empty if no faces found

        if len(face) >= 1:
            facefeatures = face
        elif len(face_two) >= 1:
            facefeatures = face_two
        elif len(face_three) >= 1:
            facefeatures = face_three
        elif len(face_four) >= 1:
            facefeatures = face_four
        else:
            facefeatures = ""
        for (x, y, w, h) in facefeatures:
            # Method to extract faces to classify for emotions later on
            gray_f = gray[y:y + h, x:x + w] # Only get the face
            cv2.rectangle(gray_f, (x, y), (x + w, y + h), (255, 0, 0), 2)
            out = cv2.resize(gray, (260,380))
            cv2.imshow("Face", out)
        print(len(facefeatures))
        return out

    # Read data from the specific emotion folders
    def gather_files(self, emotion):
        #dir_name = Path("C:/Users/Marcel/Documents/MDE/venv/Training")
        files = glob.glob("C:\\Users\\Marcel\\Documents\\MDE\\venv\\Training\\%s\\*" %emotion)
        #random.shuffle(files)
        training = files # Kann man später erhöhen
        prediction = []
        if emotion == "Neutral":
            prediction.append("C:\\Users\\Marcel\\Desktop\\201a.jpg")
        elif emotion == "Smile":
            prediction.append("C:\\Users\\Marcel\\Desktop\\201b.jpg")
        print(len(training))
        print(len(prediction))
        print(prediction)
        return training, prediction

    def make_sets(self):
        emotions = ["Neutral", "Smile"]
        training_data = []
        training_labels = []
        prediction_data = []
        prediction_labels = []
        for emotion in emotions:
            training, prediction = self.gather_files(emotion)
            for item in training:
                image = cv2.imread(item)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                training_data.append(gray)
                training_labels.append(emotions.index(emotion)) #0 oder 1
            for item in prediction:
                #image = cv2.imread(item)
                gray = self.rec_face(item)
                prediction_data.append(gray)
                cv2.imshow("img", gray)
                prediction_labels.append(emotions.index(emotion)) #0 oder 1
        return training_data, training_labels, prediction_data, prediction_labels

    def run_recognizer(self):
        fishface = cv2.face.FisherFaceRecognizer_create()
        training_data, training_labels, prediction_data, prediction_labels = self.make_sets()
        print("training fisher face classifier")
        print("size of training set is:"+str(len(training_labels))+"images")
        fishface.train(training_data, np.asarray(training_labels)) #Classifier wird trainiert
        print("predicting classification set")
        cnt = 0
        correct = 0
        incorrect = 0
        for image in prediction_data:
            pred, conf = fishface.predict(image)
            if pred == prediction_labels[cnt]:
                correct += 1
                cnt += 1
            else:
                incorrect += 1
                cnt += 1
            if pred == 0:
                print("Person" + str(cnt) +" is neutral")
            elif pred == 1:
                print("Person" + str(cnt) + " is smiling")
        return ((100*correct)/(correct+incorrect))

    metascore = []
    def run_classifier(self):
        correct = self.run_recognizer()
        print("got "+ correct + " percent correct!")
        metascore.append(correct)
        print("\n\nend score: "+np.mean(metascore)+" percent correct!")

    def rec_emotions(self):
        print("Do something")