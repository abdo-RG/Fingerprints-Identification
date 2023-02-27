from tkinter import filedialog
from click import command, progressbar
import numpy as np
from scipy.spatial import distance
from tkinter import *
from tkinter import ttk
from sympy import im, maximum
import os
import cv2
import time
from PIL import Image, ImageTk


class App(Tk):
    def __init__(self):
        Tk.__init__(self)
        self.geometry('1000x500+100+50')
        self.title("Identification des empreintes")
        self.config(bg='#80ffff')
        self.chemin_entry = StringVar()
        self.img_req_entry = Entry(self,bg='#ffff00',width=60,textvariable=self.chemin_entry).place(x=180,y=10)
        self.img_req_btn = Button(self,text='Download',command=self.choisir_fichier).place(x=560,y=10)
        self.chercher_btn = Button(self,text='Search',width=20,command=self.fing_mtch).place(x=220,y=90)
        self.cnvs = Canvas(self,width=1000,height=320,bg='#808080')
        self.cnvs.place(x=0,y=170)
        self.prg = ttk.Progressbar(self,orient=HORIZONTAL,length=300)

    # #--- Empreinte à identifier------------------------------------
    def choisir_fichier(self):
        ask_file = filedialog.askopenfile()
        pth = ask_file.name
        llst = pth.split('/')
        self.chemin_entry.set(llst[-4]+'/'+llst[-3]+'/'+llst[-2]+'/'+llst[-1])
        self.prg.place(x=320,y=130)
        self.prg.start()
    

    # #----- Identification -----------------------------------#
    def fing_mtch(self):
        #-- Declaration des variables---------------------#
        chemin = str(self.chemin_entry.get())
        self.source_image = cv2.imread(chemin)
        score=0
        file_name=None
        image=None
        kp1,kp2,mp=None,None,None
        
        #---- Parcourir les empreintes de notre dataset ---------------#
        for file in [file for file in os.listdir("SOCOFing/Real/")]:
            self.target_image = cv2.imread("SOCOFing/Real/" + file)
            #-On utilise la Classe SIFT pour détecter les clés nécessaires ,Et extraire les caractéristiques
            sift = cv2.SIFT.create()
            kp1, des1 = sift.detectAndCompute(self.source_image, None)
            kp2, des2 = sift.detectAndCompute(self.target_image, None)
           
            #---On utilise l' interface cv::FlannBasedMatcher afin d'effectuer une mise en correspondance rapide et efficace ----#
            matches = cv2.FlannBasedMatcher(dict(algorithm=1, trees=10),dict()).knnMatch(des1, des2, k=2)
            #--- Comparaison ----------------#
            mp = []
            for p, q in matches:
                if p.distance < 0.1 * q.distance:
                    mp.append(p)
                    keypoints = 0
                    if len(kp1) <= len(kp2):
                        keypoints = len(kp1)
                    else:
                        keypoints = len(kp2)
                    #--- Si les caractéristiques d'empreinte requête existe dans notre base de données avec un sueille, on va l'afficher
                    if len(mp) / keypoints * 100 > score:
                        score=len(mp) / keypoints * 100
                        self.result = cv2.drawMatches(self.source_image,kp1,self.target_image,kp2,mp,None)
                        print('The best match :'+ "SOCOFing/Real/" + file)
                        fl = file
                        # self.result = cv2.resize(self.result, None, fx=2.5, fy=2.5)
                        # cv2.imshow("result", self.result)
                        # cv2.waitKey(0)
                        # cv2.destroyAllWindows()
                        break;

        self.cnvs.create_rectangle(500,40,900,300,fill='white')
        self.cnvs.create_text(628,160,text=f'First Name :\tAxxxx\n\nLastName :\tByyyy\n\nAge :\t\t34\n\nState of Residence :\tCA\n\nChareges :\t20\n\nConvictions :\t14\n\nReal/{fl}',font=50)
        unknown = Image.open('unknown.png')
        self.unknown = unknown.resize((100,100),Image.ANTIALIAS)
        self.unknown = ImageTk.PhotoImage(self.unknown)
        self.cnvs.create_image(740,60,anchor=NW,image=self.unknown)

        img_1 = Image.open(chemin)
        self.img_1 = img_1.resize((200,200),Image.ANTIALIAS)
        self.img_1 = ImageTk.PhotoImage(self.img_1)
        self.cnvs.create_image(50,50,anchor=NW,image=self.img_1)
        self.cnvs.create_rectangle(50,250,250,270,fill='white')
        self.cnvs.create_text(110,260,text='Altered image',font=50,fill='brown')

        img_2 = Image.open("SOCOFing/Real/" + file)
        self.img_2 = img_2.resize((200,200),Image.ANTIALIAS)
        self.img_2 = ImageTk.PhotoImage(self.img_2)  
        self.cnvs.create_image(250,50,anchor=NW,image=self.img_2)
        self.cnvs.create_rectangle(250,250,450,270,fill='white')
        self.cnvs.create_text(305,260,text='Original image',font=50,fill='brown')

a = App()
a.mainloop()