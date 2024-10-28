import serial as sr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import keyboard
import os
import pyautogui
import subprocess
import time
from xlutils.copy import copy
from xlrd import open_workbook

file_path = r'C:\Users\MS\OneDrive\Desktop\Capstone\xlwt example.xls.xlsx'
if not os.path.exists(file_path):
    print(f"File not found: {file_path}")
else:
    print("File exists!")

# Open Arduino File
file = [r"C:\Users\MS\OneDrive\.netbeans\Desktop\Capstone\sketch_jul14a\sketch_jul14a.ino"]
subprocess.Popen(file, shell=True)
x = "taskkill /f /im javaw.exe"
time.sleep(10)
pyautogui.click(x=46, y=65)
time.sleep(5)

while True:
    s = sr.Serial('COM11', 9600)
    data = np.array([])
    time.sleep(5)
    length = 0
    while True:
        if length == 1500:
            break
        a = s.readline()
        a.decode()
        b = float(a[0:4])
        data = np.append(data, b)
        length += 1
        print(data[length - 1])
    s.close()
    val = np.array([])
    flag = 0
    for i in range(1000, len(data)):
        if data[i] < 400 and data[i + 1] > data[i] and flag == 0 and (data[i - 1] >= data[i]):
            val = np.append(val, data[i])
            flag = 1
        elif data[i] < 400 and data[i + 1] > data[i] and flag == 1 and (data[i - 1] >= data[i]):
            val = np.append(val, data[i])
            break
        elif flag == 1:
            val = np.append(val, data[i])
    cmin = 0
    cmax = 0
    for i in range(0, len(val)):
        if val[i] > 400 and (val[i] <= val[i + 1] and val[i] < val[i - 1]):
            cmin += 1
        if val[i] > 400 and (val[i] > val[i + 1] and val[i] >= val[i - 1]):
            cmax += 1
    if cmin == 1 and cmax == 2:
        print('Success')
        print(val)
        plt.plot(val)
        plt.show()
        systolic = max(val)
        print(systolic)
        break
    print(val)
    plt.plot(val)
    plt.show()
    systolic = max(val)
    print(systolic)
    print('Retrying')
    print(str(cmax) + ' ' + str(cmin))

    # Excel Sheet Interaction
    w = copy(open_workbook(r'C:\Users\MS\OneDrive\Desktop\Capstone\xlwt example.xls.xlsx'))
    pos = 0

    # Write data to Excel sheet
    for i in val:
        w.get_sheet(0).write(pos, 0, i)
        pos += 1

    # Save changes to the Excel file 
    w.save('xlwt example.xls')
