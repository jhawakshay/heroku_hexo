---
title: Solving Underwriting & Risk through OCR and NLP
date: 2020-01-19 21:58:47
tags: [OCR, Pytesseract,NLP,Text Parsing ]
---

As the recent pandemic has hit, Credit Underwriting managers are on their toes to contain the risk and subsequently increase the business of a Bank or a Fintech. As per the scenarios, Risk Managers have become more conservative in their appraoch to disburse loans. A part of it is basically reading through the payslips of the person who has applied for a loan.

So, How is OCR and Pytesseract be helpful?

A payslip is basically a pdf document and everything is available as a unstructured data. You can't copy from a pdf, conversion to an excel from pdf is also not very successful and then a Risk Manager is only left to see a payslip, view all the necessary details and then manually fill it ina  document or an excel. Imagine the amount of effort required for scanning through 1000's of payslips. This is where OCR and pytesseract will help to automatically get all the required details in a strucutred format.

I am going to introduce some lines on what is OCR and Pytesseract?
**OCR** is Optical Character Recognition is the electronic or mechanical conversion of images of typed, handwritten or printed text into machine-encoded text, whether from a scanned document, a photo of a document or from subtitle text superimposed on an image (for example: from a television broadcast) [Source: Wikipedia]

It is widely used in database management in Banking, Passports, Medical field. For banking, it is used in getting the Bank Statements, Payslips, Mini Statements, Transactions from passbook etc.

Below image will help you to understand what OCR does?
![OCR]](/images/OCR/OCR.jpg)

OCR as a process generally consists of several sub-processes to perform as accurately as possible. It involves:
* Preprocessing of the image
* Text Localization
* Character segmentation
* Character Recognition
* Post Processing

Now, how are we going to do it on Python? Pytesseract?
Pytesseract is an optical character recognition (OCR) tool for python. That is, it will recognize and “read” the text embedded in images.

Pytesseract is a wrapper for Google’s Tesseract-OCR Engine. It is also useful as a stand-alone invocation script to tesseract, as it can read all image types supported by the Pillow and Leptonica imaging libraries, including jpeg, png, gif, bmp, tiff, and others. 

Without wasting much of the time, Let's start coding directly.

### Install libraries in pip

* **Pytesseract** is an optical character recognition (OCR) tool for python. That is, it will recognize and “read” the text embedded in images. Python-tesseract is a wrapper for Google’s Tesseract-OCR Engine. It is also useful as a stand-alone invocation script to tesseract, as it can read all image types supported by the Pillow and Leptonica imaging libraries, including jpeg, png, gif, bmp, tiff, and others. Additionally, if used as a script, Python-tesseract will print the recognized text instead of writing it to a fil

* **pdf2image** is a python (3.5+) module that wraps pdftoppm and pdftocairo to convert PDF to a PIL Image object 

```python
pip install pytesseract
pip install Pillow
pip install tesseract
pip install pdf2image
```

### Import functions from the libraries which were installed above
```python
import re
import pytesseract
import argparse
import os
import pdf2image
import time
import numpy as np
import pandas as pd

from PIL import Image
from pdf2image import convert_from_path
```

### Import the pdf/payslip
```python
PDF_file = "D:/External Data/Basic-Salary-Slip-Example/PaySlip_and_Employee_account/payslip.pdf"
```
The same can be found in this [link](https://github.com/jhawakshay/heroku_hexo/tree/master/source/images/OCR)

### Convert the pdf to images

Here we are converting all the paged of the pdf into a jpeg format. The idea is first the pdf is converted to an images of respective pages in the pdf. Now, these images are PIL images which needs to be converted back to Jpeg or png.

Once, we have converted them to images, those are then converted to text.

```python
## Intialize Parameters
DPI = 200
OUTPUT_FOLDER = None
FIRST_PAGE = None
LAST_PAGE = None
FORMAT = 'jpg'
THREAD_COUNT = 1
USERPWD = None
USE_CROPBOX = False
STRICT = False
```
### Functions to get the pdf converted to image and then raw text

```python
def pdf_2_pil():
    start_time = time.time()
    pil_images = pdf2image.convert_from_path(PDF_file, dpi=DPI, output_folder=OUTPUT_FOLDER, 
                                             first_page=FIRST_PAGE, last_page=LAST_PAGE, fmt=FORMAT, thread_count=THREAD_COUNT, 
                                             userpw=USERPWD, use_cropbox=USE_CROPBOX, strict=STRICT, poppler_path= r'D:\External Data\Basic-Salary-Slip-Example\poppler-0.68.0\bin')
    return pil_images
    
def save_images(pil_images):
    index = 1
    for image in pil_images:
        image.save("page_" + str(index) + ".jpg")
        index += 1
```

```python
pil_images = pdf_2_pil()
save_images(pil_images)
```

### Convert the images to string or text format
```python
text = pytesseract.image_to_string(Image.open('page_1.jpg'))
```

```python
text 
```

### Text using NLP and parsing
We will be using text parsing and NLP to extract the unstructured text into meaning full arrays and tables.
The objective is to get all the details of a person on his payslip

### Remove the spaces which are there because of \n
```python
text_no_space = text.replace('\n', '')
```

We will now exract the required details from the payslip.
```python
## TYPE
type_    = text_no_space[0:8]
print(type_)
```
```python
pattern  = re.compile(r'Pay.Slip', re.IGNORECASE)
matches  = pattern.finditer(text_no_space)
for match in matches:
    print(match)
```

### Organization & City Name

Generally, after the first occurence of the Pay Slip it contains the Organization Name. So, we have got the first occurence of the Pay Slip from above, we can extend it further to get the company Name
```python
Org_name   = text_no_space[9:100]
print(Org_name)
```

```python
pattern  = re.compile(r',', re.IGNORECASE)
matches  = pattern.finditer(Org_name)
for match in matches:
    print(match)
```

```python
Org_  = Org_name[6:34]
print(Org_)
City  = Org_name[36:45]
print(City)
```

```python
Org_name[6:match.start()]
```

### Salary Month & Year

```python
pattern  = re.compile(r'[2]\d\d\d', re.IGNORECASE)
matches  = pattern.finditer(text_no_space)
```

```python
for match in matches:
    print(match)
```

### Salary of the Year is the first thing on the top of the payslip so it should be 2008
```python
Year_    = text_no_space[85:89]
print(Year_)
```

```python
pattern  = re.compile(r'(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)', re.IGNORECASE)
matches  = pattern.finditer(text_no_space)
for match in matches:
    print(match)
    
```

```python
### Here also we would find that the month of the salary is in the first occurence.
### The other things asked could be just a cumulative salary description in the Payslip

month_   = text_no_space[80:83]
print(month_)
```

### Finding the details of employer; Name, Date of Joining, Employee Number, Date of Birth, Parents Name
```python
pattern  = re.compile(r'\d\d[/]\d\d[/]\d\d\d\d', re.IGNORECASE)
matches  = pattern.finditer(text_no_space)
for match in matches:
    print(match)
```
### Logically, the DOB could not be 2005 as a person less than 18 years old can't work
```python
DoB_      = text_no_space[207:217]
print(DoB_)
DoJ_      = text_no_space[174:184]
print(DoJ_)
```


