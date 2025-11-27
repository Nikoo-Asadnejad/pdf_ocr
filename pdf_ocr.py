import os
import sys
import platform
import numpy as np
import re
import unicodedata

try:
    import pytesseract
except ImportError:
    print("Error: pytesseract is not installed.")
    print("Install it using: pip install pytesseract")
    sys.exit(1)

try:
    import fitz  # PyMuPDF
except ImportError:
    print("Error: PyMuPDF is not installed.")
    print("Install it using: pip install PyMuPDF")
    sys.exit(1)

try:
    from PIL import Image, ImageEnhance, ImageFilter, ImageOps
    import io
except ImportError:
    print("Error: Pillow is not installed.")
    print("Install it using: pip install Pillow")
    sys.exit(1)

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    print("Warning: OpenCV not installed. Advanced preprocessing will be limited.")
    print("For better results, install it using: pip install opencv-python")
    OPENCV_AVAILABLE = False

try:
    from arabic_reshaper import reshape
    from bidi.algorithm import get_display
    BIDI_AVAILABLE = True
except ImportError:
    print("Warning: Arabic reshaper and python-bidi not installed.")
    print("Install using: pip install arabic-reshaper python-bidi")
    BIDI_AVAILABLE = False

# Configure Tesseract path for Windows
if platform.system() == "Windows":
    tesseract_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if not os.path.exists(tesseract_path):
        print("\nError: Tesseract-OCR not found at expected location.")
        print("Please install Tesseract-OCR from: https://github.com/UB-Mannheim/tesseract/wiki")
        print(f"Expected path: {tesseract_path}")
        sys.exit(1)
    pytesseract.pytesseract.tesseract_cmd = tesseract_path

# Get script directory and file paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_PATH = os.path.join(SCRIPT_DIR, "WebserviceDoc_1403-02-11-15.pdf")
OUTPUT_TEXT_PATH = os.path.join(SCRIPT_DIR, "extracted_text_clean.txt")
OUTPUT_HTML_PATH = os.path.join(SCRIPT_DIR, "extracted_text_clean.html")


def is_persian_text(text):
    """Check if text contains Persian/Arabic characters."""
    persian_chars = 0
    total_chars = 0
    
    for char in text:
        if char.isalpha():
            total_chars += 1
            if '\u0600' <= char <= '\u06FF' or '\u0750' <= char <= '\u077F':
                persian_chars += 1
    
    if total_chars == 0:
        return False
    
    return persian_chars / total_chars > 0.3  # More than 30% Persian chars


def clean_text(text):
    """Clean and normalize extracted text."""
    if not text:
        return ""
    
    # Remove multiple consecutive spaces
    text = re.sub(r' +', ' ', text)
    
    # Remove multiple consecutive newlines (keep max 2)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Fix Persian/Arabic numbers (convert to Persian digits)
    persian_digits = '۰۱۲۳۴۵۶۷۸۹'
    english_digits = '0123456789'
    trans_table = str.maketrans(english_digits, persian_digits)
    
    # Only convert digits in Persian context
    if is_persian_text(text):
        # Keep English numbers in specific patterns (like dates, codes)
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Skip lines that look like codes or IDs
            if re.match(r'^[A-Z0-9_\-\.]+$', line.strip()):
                cleaned_lines.append(line)
            else:
                # Convert numbers in Persian text
                cleaned_lines.append(line.translate(trans_table))
        
        text = '\n'.join(cleaned_lines)
    
    # Fix common OCR errors in Persian text
    if is_persian_text(text):
        # Fix separated Persian characters
        text = re.sub(r'(\S)\s+(?=[ًٌٍَُِّْ])', r'\1', text)  # Fix diacritics
        
        # Fix common OCR mistakes
        replacements = {
            'ي': 'ی',  # Arabic Yeh to Persian Yeh
            'ك': 'ک',  # Arabic Kaf to Persian Kaf
            '  ': ' ',  # Double space to single
            ' ، ': '، ',  # Fix comma spacing
            ' . ': '. ',  # Fix period spacing
            ' : ': ': ',  # Fix colon spacing
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
    
    # Remove extra whitespace at start and end of lines
    lines = text.split('\n')
    text = '\n'.join(line.strip() for line in lines)
    
    return text


def format_for_rtl(text):
    """Format text for proper RTL display."""
    if not BIDI_AVAILABLE:
        return text
    
    lines = text.split('\n')
    formatted_lines = []
    
    for line in lines:
        if is_persian_text(line):
            # Reshape Arabic/Persian text for proper display
            reshaped_text = reshape(line)
            # Apply bidirectional algorithm for proper RTL display
            bidi_text = get_display(reshaped_text)
            formatted_lines.append(bidi_text)
        else:
            formatted_lines.append(line)
    
    return '\n'.join(formatted_lines)


def advanced_preprocess_image(image):
    """Advanced image preprocessing for better OCR."""
    try:
        if OPENCV_AVAILABLE:
            # Convert PIL to OpenCV format
            img_array = np.array(image)
            
            # Convert to grayscale if needed
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            
            # Apply adaptive thresholding
            thresh = cv2.adaptiveThreshold(
                gray, 255, 
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 
                11, 2
            )
            
            # Denoise
            denoised = cv2.fastNlMeansDenoising(thresh, h=10)
            
            # Morphological operations
            kernel = np.ones((1, 1), np.uint8)
            morph = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel)
            
            # Convert back to PIL Image
            image = Image.fromarray(morph)
        
        else:
            # PIL-only processing
            if image.mode != 'L':
                image = image.convert('L')
            
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(3.0)
            
            threshold = 128
            image = image.point(lambda p: 255 if p > threshold else 0)
            
            image = image.filter(ImageFilter.SHARPEN)
            image = image.filter(ImageFilter.SHARPEN)
            image = image.filter(ImageFilter.EDGE_ENHANCE)
        
        return image
        
    except Exception as e:
        print(f"    Preprocessing error: {e}")
        return image


def extract_with_best_config(img):
    """Extract text using the best OCR configuration."""
    best_text = ""
    best_length = 0
    
    # Try different configurations
    configs = [
        ("fas", "--psm 6 --oem 3 -c preserve_interword_spaces=1"),
        ("fas+eng", "--psm 6 --oem 3"),
        ("fas", "--psm 3 --oem 3"),
        ("eng", "--psm 6 --oem 3"),
    ]
    
    for lang, config in configs:
        try:
            text = pytesseract.image_to_string(img, lang=lang, config=config)
            if len(text.strip()) > best_length:
                best_text = text
                best_length = len(text.strip())
                if best_length > 500:  # Good enough
                    break
        except:
            continue
    
    return best_text


def create_html_output(all_text, output_path):
    """Create an HTML file with proper RTL formatting for Persian text."""
    html_content = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Extracted PDF Text</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Arial, sans-serif;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
            line-height: 1.8;
            background-color: #f5f5f5;
        }
        .page {
            background: white;
            padding: 30px;
            margin: 20px 0;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            border-radius: 5px;
        }
        .page-header {
            background: #2c3e50;
            color: white;
            padding: 10px 20px;
            margin: -30px -30px 20px -30px;
            border-radius: 5px 5px 0 0;
            font-weight: bold;
        }
        .rtl {
            direction: rtl;
            text-align: right;
            font-family: 'Vazir', 'B Nazanin', 'Tahoma', sans-serif;
            font-size: 16px;
        }
        .ltr {
            direction: ltr;
            text-align: left;
            font-family: 'Courier New', monospace;
        }
        .mixed {
            direction: rtl;
            text-align: right;
        }
        pre {
            white-space: pre-wrap;
            word-wrap: break-word;
        }
        .line {
            margin: 10px 0;
            padding: 5px 0;
            border-bottom: 1px solid #f0f0f0;
        }
    </style>
</head>
<body>
"""
    
    for i, page_text in enumerate(all_text, 1):
        # Determine if page is primarily RTL
        is_rtl = is_persian_text(page_text)
        
        html_content += f"""
    <div class="page">
        <div class="page-header">صفحه {i} / Page {i}</div>
        <div class="{'rtl' if is_rtl else 'ltr'}">
"""
        
        # Process each line
        lines = page_text.split('\n')
        for line in lines:
            if line.strip():
                line_class = 'rtl' if is_persian_text(line) else 'ltr'
                escaped_line = line.replace('<', '&lt;').replace('>', '&gt;')
                html_content += f'            <div class="line {line_class}">{escaped_line}</div>\n'
            else:
                html_content += '            <br>\n'
        
        html_content += """        </div>
    </div>
"""
    
    html_content += """
</body>
</html>"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"HTML output saved to: {output_path}")


def extract_all_text_clean(pdf_path, output_text_path, output_html_path):
    """Extract and clean text from PDF."""
    if not os.path.exists(pdf_path):
        print(f"\nError: PDF file not found: {pdf_path}")
        return False
    
    print("="*60)
    print("PDF OCR Text Extractor - Clean Output")
    print("="*60)
    print(f"Processing: {os.path.basename(pdf_path)}")
    
    try:
        # Open PDF
        pdf_document = fitz.open(pdf_path)
        total_pages = pdf_document.page_count
        print(f"Total pages: {total_pages}")
        print("-"*60)
        
        all_text = []
        
        for page_num in range(total_pages):
            print(f"Processing page {page_num + 1}/{total_pages}...", end=" ")
            
            try:
                page = pdf_document[page_num]
                
                # Check for embedded text
                embedded_text = page.get_text().strip()
                
                # Render page as high-quality image
                mat = fitz.Matrix(600/72.0, 600/72.0)
                pix = page.get_pixmap(matrix=mat, alpha=False)
                
                # Convert to PIL Image
                img_data = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_data))
                
                # Preprocess image
                processed_img = advanced_preprocess_image(img)
                
                # Run OCR
                ocr_text = extract_with_best_config(processed_img)
                
                # Choose best text
                if len(embedded_text) > 100 and len(embedded_text) > len(ocr_text):
                    final_text = embedded_text
                    print(f"Extracted {len(final_text)} chars (embedded)")
                else:
                    final_text = ocr_text if ocr_text else embedded_text
                    print(f"Extracted {len(final_text)} chars (OCR)")
                
                # Clean the text
                final_text = clean_text(final_text)
                
                # Format for RTL if needed
                if BIDI_AVAILABLE:
                    final_text = format_for_rtl(final_text)
                
                all_text.append(final_text)
                
            except Exception as e:
                print(f"Error: {e}")
                all_text.append(f"[Error extracting page {page_num + 1}]")
        
        pdf_document.close()
        
        # Save text output
        with open(output_text_path, 'w', encoding='utf-8-sig') as f:  # UTF-8 with BOM for better Windows support
            for i, page_text in enumerate(all_text, 1):
                f.write(f"\n{'='*50}\n")
                f.write(f"صفحه {i} / Page {i}\n")
                f.write(f"{'='*50}\n\n")
                f.write(page_text)
                f.write("\n\n")
        
        # Create HTML output for better RTL viewing
        create_html_output(all_text, output_html_path)
        
        print("\n" + "="*60)
        print(f"SUCCESS!")
        print(f"Text output: {output_text_path}")
        print(f"HTML output: {output_html_path}")
        print(f"Total characters extracted: {sum(len(text) for text in all_text):,}")
        print("="*60)
        print("\nTip: Open the HTML file in a browser for better RTL text display!")
        
        return True
        
    except Exception as e:
        print(f"\nError: {e}")
        return False


if __name__ == "__main__":
    # Check OCR languages
    try:
        langs = pytesseract.get_languages()
        print(f"Available OCR languages: {', '.join(langs)}\n")
        
        if 'fas' not in langs:
            print("WARNING: Persian (fas) language not installed!")
            print("For better Persian text recognition, install it:")
            print("  Ubuntu/Debian: sudo apt-get install tesseract-ocr-fas")
            print("  Windows: Download fas.traineddata from Tesseract GitHub\n")
            
    except:
        pass
    
    # Install required packages if missing
    if not BIDI_AVAILABLE:
        print("\nInstalling RTL text formatting libraries...")
        os.system("pip install arabic-reshaper python-bidi")
        print("Please restart the script after installation.\n")
    
    # Process the PDF
    success = extract_all_text_clean(PDF_PATH, OUTPUT_TEXT_PATH, OUTPUT_HTML_PATH)
    
    if not success:
        sys.exit(1)
