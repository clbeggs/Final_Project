"""
GUI "Paint" program to accompany team "A Cool Name"'s final project for CSCI 4622 Machine Learning.
Provides a GUI to draw images and submit them to the machine learning model.
"""


import tkinter as tk
from tkinter import ttk
from PIL import Image
from io import BytesIO


class Gui:
    def __init__(self):
        self.mode = 0  # 0 = paint, 1 = erase
        self.x = 0
        self.y = 0

        root = tk.Tk()
        self.size = tk.IntVar()
        self.size.set(5)

        root.title("Pix2Pix GAN GUI")
        root.resizable(width=False, height=False)
        root.geometry("530x590+695+245")  # 530x590 window centered on 1920x1080 displays

        # Next 3 lines to bring window to front on launch, but allow other apps in front afterwards
        root.lift()
        root.attributes('-topmost', True)
        root.after_idle(root.attributes, '-topmost', False)

        # Create all needed buttons
        self.drawButton = ttk.Button(root, text="Draw", command=self.setDraw).grid(row=0, column=0)
        self.eraseButton = ttk.Button(root, text="Erase", command=self.setErase).grid(row=0, column=1)
        self.clearAllButton = ttk.Button(root, text="Clear All", command=self.clearAll).grid(row=1, column=0)
        self.submitButton = ttk.Button(root, text="Submit", command=self.submit).grid(row=1, column=1)

        # Create Paint Size slider
        self.scaleLabel = tk.Label(root, text="Size").grid(row=1, column=2)
        self.scale = tk.Scale(root, variable=self.size, from_=1, to=10, cursor='target', orient=tk.HORIZONTAL).grid(row=0, column=2)

        # Create canvas to draw on, but don't pack it until adding event bindings
        self.canvas = tk.Canvas(root, cursor='dot', height=512, width=512, bg='white', highlightthickness=3, highlightbackground='grey')

        self.canvas.bind('<ButtonPress-1>', self.setStartPoint)  # Gets initial left-click
        self.canvas.bind('<B1-Motion>', self.draw)  # Gets motion from mouse while left-click is held

        self.canvas.grid(row=2, column=0, columnspan=3, padx=5, pady=3)

        root.mainloop()

    def setDraw(self):
        self.mode = 0

    def setErase(self):
        self.mode = 1

    def clearAll(self):
        self.canvas.delete('all')

    # Submit image from Canvas to GAN
    def submit(self):
        canvasImage = self.canvas.postscript(colormode='color')  # Get Canvas image as Unicode string
        im = Image.open(BytesIO(canvasImage.encode('utf-8')))  # Convert to image
        im = im.resize((256, 256))  # Resize 512x512 canvas to 256x256 to match ML model
        im.show()

        # TODO: Feed image to GAN Model

    # On initial left mouse click, set where the click occurred
    def setStartPoint(self, event):
        self.x = event.x
        self.y = event.y

    # When moving the mouse with left button down, draw on canvas
    def draw(self, event):
        color = 'black'
        if self.mode:
            color = 'white'
        self.canvas.create_line(self.x, self.y, event.x, event.y, width=self.size.get(), smooth=True, fill=color)
        self.x = event.x
        self.y = event.y


if __name__ == "__main__":
    g = Gui()