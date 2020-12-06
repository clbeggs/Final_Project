"""
GUI "Paint" program to accompany team "A Cool Name"'s final project for CSCI 4622 Machine Learning.
Provides a GUI to draw images and submit them to the machine learning model.
"""


from itertools import cycle
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from io import BytesIO

import cv2 as cv
import numpy as np
from cycle_gan import CycleGan
from pix2pix import Pix2Pix

class Gui:
    def __init__(self):
        self.mode = 0  # 0 = paint, 1 = erase
        self.x = 0
        self.y = 0

        self.root = tk.Tk()

        # Declare Variables Used in GUI
        self.size = tk.IntVar()  # Paint Brush Size
        self.size.set(5)
        self.selectedModel = tk.StringVar()  # What to generate in our GAN
        self.selectedModel.set('trees')
        self.selectedGAN = tk.StringVar()  # What GAN to use for real time display
        self.selectedGAN.set('cyclegan')
        self.canvasTitle = tk.StringVar()  # Title for drawing area
        self.canvasTitle.set('Drawing (Trees)')

        self.root.title("Pix2Pix GAN GUI")
        self.root.resizable(width=False, height=False)
        self.root.geometry("1075x645+436+218")  # 1050x590 window centered on 1920x1080 displays

        # Next 3 lines to bring window to front on launch, but allow other apps in front afterwards
        self.root.lift()
        self.root.attributes('-topmost', True)
        self.root.after_idle(self.root.attributes, '-topmost', False)

        # Create all needed buttons
        self.drawButton = ttk.Button(self.root, text="Draw", command=self.setDraw).grid(row=0, column=0)
        self.eraseButton = ttk.Button(self.root, text="Erase", command=self.setErase).grid(row=0, column=1)
        self.clearAllButton = ttk.Button(self.root, text="Clear All", command=self.clearAll).grid(row=1, column=0)
        self.saveButton = ttk.Button(self.root, text="Save", command=self.save).grid(row=1, column=1)

        # Create Model Selector above image viewer
        self.modelSelectorLabel = ttk.Label(self.root, text='Select What to Generate:').grid(row=0, column=3)
        self.treeButton = ttk.Radiobutton(self.root, text="Trees", variable=self.selectedModel, value='trees', command=self.selectModel)
        self.treeButton.grid(row=0, column=4)
        self.quickDraw = ttk.Radiobutton(self.root, text="QuickDraw", variable=self.selectedModel, value='quickdraw_trees', command=self.selectModel)
        self.quickDraw.grid(row=0, column=5)
        self.pizzaButton = ttk.Radiobutton(self.root, text="Pizza", variable=self.selectedModel, value='pizza', command=self.selectModel)
        self.pizzaButton.grid(row=0, column=6)
        self.appleButton = ttk.Radiobutton(self.root, text="Apples", variable=self.selectedModel, value='apples', command=self.selectModel)
        self.appleButton.grid(row=0, column=7)

        # Create Paint Size slider
        self.scaleLabel = tk.Label(self.root, text="Size").grid(row=1, column=2)
        self.scale = tk.Scale(self.root, variable=self.size, from_=1, to=30, cursor='target', orient=tk.HORIZONTAL).grid(row=0, column=2)

        # Create canvas to draw on, but don't pack it until adding event bindings
        self.canvasLabel = tk.Label(self.root, textvariable=self.canvasTitle, font=("Arial", 16)).grid(row=2, column=0, columnspan=3, pady=(25, 0))
        self.canvas = tk.Canvas(self.root, cursor='dot', height=512, width=512, bg='white', highlightthickness=3, highlightbackground='grey')

        self.canvas.bind('<ButtonPress-1>', self.setStartPoint)  # Gets initial left-click
        self.canvas.bind('<B1-Motion>', self.draw)  # Gets motion from mouse while left-click is held
        self.canvas.bind('<ButtonRelease-1>', self.updateImageViewer)  # Update image in real time on left-click release

        self.canvas.grid(row=3, column=0, columnspan=3, padx=5, pady=3)

        # Create Canvas to view images produced by GAN
        self.imageViewerLabel = tk.Label(self.root, text="Generated Image (Pix2Pix)", font=("Arial", 16)).grid(row=2, column=3, columnspan=2, pady=(25, 0))
        self.imageViewer = tk.Canvas(self.root, cursor='arrow', height=256, width=256, bg='white', highlightthickness=3, highlightbackground='grey')
        self.imageViewer.grid(row=3, column=3, columnspan=2, sticky=tk.NW, pady=3, padx=5)

        self.imageViewerLabel2 = tk.Label(self.root, text="Generated Image (CycleGAN)", font=("Arial", 16)).grid(row=2, column=5, columnspan=3, pady=(25, 0))
        self.imageViewer2 = tk.Canvas(self.root, cursor='arrow', height=256, width=256, bg='white', highlightthickness=3, highlightbackground='grey')
        self.imageViewer2.grid(row=3, column=5, columnspan=3, sticky=tk.NW, pady=3)


        # Initialize models
        self.cycle_gan_models = {}
        for name in ['trees', 'pizza', 'apples', 'quickdraw_trees']:
            self.cycle_gan_models[name] = CycleGan('cyclegan_%s'%name, epoch='latest')

        self.pix2pix_models = {}
        for name, epoch in zip(['trees', 'pizza', 'apples'], ['latest', 'latest', 'latest']):
            self.pix2pix_models[name] = Pix2Pix('pix2pix_%s'%name,epoch=epoch)
        self.pix2pix_models['quickdraw_trees'] = self.pix2pix_models['trees']

        self.root.mainloop()

    def setDraw(self):
        self.mode = 0

    def setErase(self):
        self.mode = 1

    def clearAll(self):
        self.canvas.delete('all')
        self.updateImageViewer()

    # Gets the right model based on the user input
    def getModel(self):
        model = {}
        if self.selectedModel.get() in self.pix2pix_models:
            model['pix2pix'] = self.pix2pix_models[self.selectedModel.get()]
        if self.selectedModel.get() in self.cycle_gan_models:
            model['cyclegan'] = self.cycle_gan_models[self.selectedModel.get()]
        if len(model) == 0:
            raise ValueError("Unknown model!")
        
        return model

    # Called on left-click release to update image in real time
    def updateImageViewer(self, event=None):
        canvasImage = self.getImageFromCanvas()

        model = self.getModel()
        
        if 'pix2pix' in model:
            generated_img_pix2pix = Image.fromarray(model['pix2pix'](canvasImage))
            self.pix2pixImg = ImageTk.PhotoImage(generated_img_pix2pix)
            self.imageViewer.create_image(0, 0, image=self.pix2pixImg, anchor=tk.NW)
        if 'cyclegan' in model:
            generated_img_cyclegan = Image.fromarray(model['cyclegan'](canvasImage))
            self.cycleGanImg = ImageTk.PhotoImage(generated_img_cyclegan)
            self.imageViewer2.create_image(0, 0, image=self.cycleGanImg, anchor=tk.NW)

        if event:
            self.x = event.x
            self.y = event.y
        self.root.update()

    # Generate images for saving from all GAN outputs possible for selected model
    def save(self):
        # Get 256x256 image drawn on canvas
        canvasImage = self.getImageFromCanvas()

        pix_model = self.pix2pix_models[self.selectedModel.get()]
        cycle_model = self.cycle_gan_models[self.selectedModel.get()]

        pix_img = pix_model(canvasImage)
        cycle_img = cycle_model(canvasImage)

        cv.imshow("Pix2Pix", cv.cvtColor(pix_img, cv.COLOR_RGB2BGR))
        cv.imshow("CycleGAN", cv.cvtColor(cycle_img, cv.COLOR_RGB2BGR))


    # On initial left mouse click, set where the click occurred
    def setStartPoint(self, event):
        self.x = event.x
        self.y = event.y

    # When moving the mouse with left button down, draw on canvas
    def draw(self, event):
        color = 'black'
        if self.mode:
            color = 'white'
        self.canvas.create_line(self.x, self.y, event.x, event.y, width=self.size.get(), smooth=True, fill=color, capstyle=tk.ROUND)
        self.x = event.x
        self.y = event.y

    def selectModel(self):
        modelSelected = self.selectedModel.get()
        self.canvasTitle.set('Drawing (%s)'%modelSelected)
        self.updateImageViewer()
        self.root.update()

    def getImageFromCanvas(self):
        canvasImage = self.canvas.postscript(colormode='color')  # Get Canvas image as Unicode string
        im = Image.open(BytesIO(canvasImage.encode('utf-8')))  # Convert to image
        im = im.resize((256, 256))  # Resize 512x512 canvas to 256x256 to match ML model
        return im


if __name__ == "__main__":
    g = Gui()
