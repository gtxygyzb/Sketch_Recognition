from tkinter import *
from PIL import ImageGrab
import torch

win_width = 512
win_height = 512 + 52

class Application(Frame):
    def __init__(self, master=None, bgcolor="#ffffff"):

        super().__init__(master)
        self.master = master
        self.bgcolor = bgcolor
        self.x = 0
        self.y = 0
        self.fgcolor = "#000000"
        self.lastDraw = 0  # 表示最后绘制的图形的id
        self.startDrawFlag = False
        self.pack()
        self.createWidget()
        # 创建画板


    def createWidget(self):
        self.drawCad = Canvas(self, width=win_width, height=win_height * 0.9, bg=self.bgcolor)
        self.drawCad.pack()
        # 创建按钮
        btn_pen = Button(self, text="画笔", name="pen")
        btn_pen.pack(side="left", padx=10)
        btn_clear = Button(self, text="清屏", name="clear")
        btn_clear.pack(side="left", padx=10)
        btn_earsor = Button(self, text="橡皮擦", name="earsor")
        btn_earsor.pack(side="left", padx=10)
        btn_save = Button(self, text="保存", name="save")
        btn_save.pack(side="left", padx=10)

        # 为按钮绑定事件
        btn_pen.bind_class("Button", "<1>", self.eventManger)
        self.drawCad.bind("<ButtonRelease-1>", self.stopDraw)

    def eventManger(self, event):
        name = event.widget.winfo_name()
        print(name)
        if name == "pen":
            self.drawCad.bind("<B1-Motion>", self.myPen)
        elif name == "earsor":
            self.drawCad.bind("<B1-Motion>", self.myEarsor)
        elif name == "clear":
            self.drawCad.delete("all")
        elif name == "save":
            x0 = self.winfo_rootx() + 1  # 画布的左上角的X轴坐标
            y0 = self.winfo_rooty() + 1  # 画布的左上角的Y轴坐标
            x1 = x0 + 510  # 画布的右下角的X轴坐标
            y1 = y0 + 510  # 画布的右下角的Y轴坐标　　
            ImageGrab.grab().crop((x0 * 1.25, y0 * 1.25, x1 * 1.25, y1 * 1.25)).save("./tmp/tmp.png")
            self.quit()
        # elif name == "color":
            # c = askcolor(color=self.fgcolor, title="画笔选择颜色")
            # self.fgcolor = c[1]

    def myline(self, event):
        self.startDraw(event)
        self.lastDraw = self.drawCad.create_line(self.x, self.y,
                                                 event.x, event.y, fill=self.fgcolor)

    def myPen(self, event):
        self.startDraw(event)
        self.drawCad.create_rectangle(self.x, self.y, event.x, event.y, outline=self.fgcolor)
        self.x = event.x
        self.y = event.y

    def myEarsor(self, event):
        self.startDraw(event)
        self.drawCad.create_rectangle(event.x-5, event.y-5, event.x+5, event.y+5,
                                      fill=self.bgcolor, outline=self.bgcolor)
        self.x = event.x
        self.y = event.y

    def stopDraw(self, event):
        self.startDrawFlag = False
        self.lastDraw = 0

    def startDraw(self, event):
        self.drawCad.delete(self.lastDraw)
        if not self.startDrawFlag:
            self.startDrawFlag = True
            self.x = event.x
            self.y = event.y

def work():
    root = Tk()
    app = Application(root)
    root.title("简易的画图工具")
    root.geometry(str(win_width)+"x"+str(win_height)+"+100+100")
    root.mainloop()


if __name__ == "__main__":
    work()

