from tkinter import *
from tkinter.filedialog import askopenfilename, askdirectory
from os import getcwd, fsencode, fsdecode, listdir
from PIL import Image, ImageTk
from meta_data_generation import *
import cv2
from pathlib import Path

class GUI:
    def __init__(self, root):
        root.geometry("1024x768")
        root.resizable(0, 0)
        root.title("IMDE")
        self.var = StringVar(root)
        self.var.set("Color Distribution")  # default value

        self.left_frame = Frame(root, background="gray")
        self.right_frame = Frame(root)
        self.right_top_frame = Frame(self.right_frame, background="white")
        self.right_bottom_frame = Frame(self.right_frame, background="brown")
        self.left_frame.pack(fill="both", expand="1", side=LEFT)
        self.right_frame.pack(fill="both", expand="1", side=RIGHT)
        self.right_frame.pack_propagate(0)
        self.right_top_frame.pack(fill="both", expand="1")
        self.right_top_frame.pack_propagate(0)
        self.right_bottom_frame.pack(fill="both", side=BOTTOM)


        self.select_button = Button(self.right_bottom_frame, text="Browse files", command=self.select_File)
        self.directory_button = Button(self.right_bottom_frame, text="Load directory", command=self.select_Directory)
        self.meta_data_menu = OptionMenu(self.right_bottom_frame, self.var, "Color Distribution", "Face Detection", "Emotion Classifier")
        self.start_button = Button(self.right_bottom_frame, text="Start", command=self.start_Calculation)
        self.exit_button = Button(self.right_bottom_frame, text="Exit", command=root.destroy)
        self.forward_button = Button(self.right_top_frame, text="->", command=self.browse_Forwards)
        self.backward_button = Button(self.right_top_frame, text="<-", command=self.browse_Backwards)
        self.exit_button.pack(fill="both", side=BOTTOM)
        self.start_button.pack(fill="both", side=BOTTOM)
        self.meta_data_menu.pack(fill="both", side=BOTTOM)
        self.directory_button.pack(fill="both", side=BOTTOM)
        self.select_button.pack(fill="both", side=BOTTOM)
        self.forward_button.pack(side=RIGHT)
        self.backward_button.pack(side=LEFT)
        root.mainloop()

    def select_File(self):
        self.currdir = getcwd()
        self.image_path = askopenfilename(initialdir=self.currdir,
                                          filetypes=(("Image File", "*.jpg"), ("All Files", "*.*")),
                                          title="Select an image."
                                          )

        if len(self.right_top_frame.winfo_children()) > 2:
            self.image_label.destroy()

        self.image = Image.open(self.image_path)
        self.size = (int(self.right_top_frame.winfo_height()), self.right_top_frame.winfo_width())
        self.resized = self.image.resize(self.size, Image.ANTIALIAS)
        self.photo = ImageTk.PhotoImage(self.resized)

        self.image_label = Label(self.right_top_frame, image=self.photo)
        self.image_label.image = self.photo
        self.image_label.pack(fill="both", expand="1")

    def select_Directory(self):
        #Lade ein Ordner mit mehreren Bildern
        #Speichere Bilder in einer Liste
        #Label zeigt das erste Bild an
        #Pfeiltasten zum browsen
        #image_path haelt Link zum derzeit angezeigten Bild

        self.currdir = getcwd()
        self.counter = 0
        self.image_container = [] # Create empty list for images
        self.dir_path = askdirectory(title="Select a directory.")
        self.directory = fsencode(self.dir_path)
        self.dir_path_container = []
        self.size = (int(self.right_top_frame.winfo_height()), self.right_top_frame.winfo_width())

        for file in listdir(self.directory):
            self.filename = fsdecode(file)
            if self.filename.endswith(".jpg"):
                self.image = Image.open(self.dir_path+"/"+self.filename)
                self.resized = self.image.resize(self.size, Image.ANTIALIAS)
                self.photo = ImageTk.PhotoImage(self.resized)
                self.image_container.append(self.photo)
                self.dir_path_container.append(self.dir_path+"/"+self.filename)
            else:
                print("Picture not jpg and therefore not loaded")
        print(len(self.image_container))#DEBUG

        if len(self.right_top_frame.winfo_children()) > 2:
            self.image_label.destroy()

        self.image_label = Label(self.right_top_frame, image=self.image_container[self.counter])
        self.image_label.image = self.image_container[self.counter]
        self.image_label.pack(fill="both", expand="1")

    def browse_Forwards(self):
        if len(self.image_container)>0:
            if(self.counter<len(self.image_container)-1):
                self.counter += 1

                if len(self.right_top_frame.winfo_children()) > 2:
                    self.image_label.destroy()

                self.image_label = Label(self.right_top_frame, image=self.image_container[self.counter])
                self.image_label.image = self.image_container[self.counter]
                self.image_label.pack(fill="both", expand="1")
                print(self.dir_path_container[self.counter])

    def browse_Backwards(self):
        if len(self.image_container)>0:
            if(self.counter>0):
                self.counter -= 1

                if len(self.right_top_frame.winfo_children()) > 2:
                    self.image_label.destroy()

                self.image_label = Label(self.right_top_frame, image=self.image_container[self.counter])
                self.image_label.image = self.image_container[self.counter]
                self.image_label.pack(fill="both", expand="1")
                print(self.dir_path_container[self.counter])

    def start_Calculation(self):
        if self.var.get() == "Color Distribution":
            print("Code for CD")
            self.currdir = getcwd()
            self.image_path = askopenfilename(initialdir=self.currdir,
                                          filetypes=(("Image File", "*.jpg"), ("All Files", "*.*")),
                                          title="Select an image.")
            self.dir_path = askdirectory(title="Select a directory")
            print("yo")
            self.directory = fsencode(self.dir_path)
            self.dir_path_container = []
            img = cv2.imread(self.image_path)
            cv2.imshow('Image', img)
            for file in listdir(self.directory):
                self.filename = fsdecode(file)
                if self.filename.endswith(".jpg"):
                    self.dir_path_container.append(self.dir_path + "/" + self.filename)
                else:
                    print("Picture not jpg and therefore not loaded")

            CD = ColorDistribution()
            CD.closest_hist(self.image_path, self.dir_path_container, 0)

        elif self.var.get() == "Face Detection":
            print("Code for object")
            self.currdir = getcwd()
            self.image_path = askopenfilename(initialdir=self.currdir,
                                          filetypes=(("Image File", "*.jpg"), ("All Files", "*.*")),
                                          title="Select an image.")
            em = Emotions()
            em.rec_face(self.image_path)

        elif self.var.get() == "Emotion Classifier":
            em = Emotions()
            #em.gather_files("Neutral")
            correct = em.run_recognizer()





# So kann ich auf root zugreifen
root = Tk()
x = GUI(root)