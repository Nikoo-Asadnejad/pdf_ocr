# 📄 PDF OCR Text Extractor
High-quality text extraction from scanned PDFs with Persian/Arabic (RTL) support.

This project provides a powerful Python script that extracts text from PDFs — especially scanned or image-based documents — using Tesseract OCR, advanced image preprocessing, and robust text-cleaning tailored for Persian/Arabic languages.  
It outputs both **clean UTF-8 text** and **HTML with proper RTL support**.

---

## ✨ Features

### 🔍 OCR Extraction
- Works with scanned PDFs  
- Supports Farsi, Arabic, and English  
- Runs multiple OCR configurations and selects the best output  

### 🖼 Advanced Preprocessing
- Uses OpenCV (if available):
  - Adaptive thresholding  
  - Noise reduction  
  - Morphological cleanup  
- Uses Pillow fallback when OpenCV is not installed  

### 🇮🇷 RTL Persian/Arabic Handling
- Auto-detects Persian text  
- Normalizes common OCR mistakes  
- Fixes diacritics, punctuation, and spacing  
- Converts digits contextually  
- Uses arabic-reshaper + bidi for correct display  

### 📝 Output Formats
- `extracted_text_clean.txt`  
- `extracted_text_clean.html`  
- Page-separated output  

---

## 📦 Requirements

### Install Python Dependencies
```bash
pip install pytesseract PyMuPDF Pillow opencv-python arabic-reshaper python-bidi numpy
```

### Install Tesseract OCR
Windows:
<a href ="https://github.com/UB-Mannheim/tesseract/wiki"> https://github.com/UB-Mannheim/tesseract/wiki </a>

Ubuntu :
```bash 
sudo apt-get install tesseract-ocr tesseract-ocr-fas
```

MacOs:
```bash
brew install tesseract
brew install tesseract-lang
```

### Usage
Place the PDF into the same directory and name it: 
```Sample.pdf```

Run:
```bash
python pdf_ocr.py
```




