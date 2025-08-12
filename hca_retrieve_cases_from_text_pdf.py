import os
import argparse
import re
import time
from datetime import datetime
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from io import BytesIO
import concurrent.futures
import threading

# Add a lock for print operations to avoid garbled output
print_lock = threading.Lock()

def safe_print(*args, **kwargs):
    """Thread-safe print function."""
    with print_lock:
        print(*args, **kwargs)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Convert text files to PDF recursively.')
    parser.add_argument('--source', '-s', required=True, help='Source directory containing text files')
    parser.add_argument('--destination', '-d', required=True, help='Destination directory for PDF files')
    parser.add_argument('--threads', '-t', type=int, default=os.cpu_count(), 
                        help=f'Number of threads to use (default: {os.cpu_count()})')
    return parser.parse_args()

def find_text_files(source_dir):
    """Recursively find all text files in the source directory."""
    text_files = []
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.lower().endswith('.txt'):
                text_files.append(os.path.join(root, file))
    return text_files

def extract_year_from_content(text_content):
    """Try to extract a year from the text content."""
    # Look for years between 1900 and current year
    current_year = datetime.now().year
    year_pattern = re.compile(r'\b(19\d{2}|20[0-2]\d)\b')
    
    matches = year_pattern.findall(text_content)
    if matches:
        # Return the first year found in the content
        return matches[0]
    
    return None

def convert_text_to_pdf(text_path, output_dir):
    """Convert a text file to PDF using reportlab."""
    try:
        # Get the filename without extension
        filename = os.path.basename(text_path)
        name_without_ext = os.path.splitext(filename)[0]
        
        # Read text file
        with open(text_path, 'r', encoding='utf-8', errors='ignore') as file:
            text_content = file.read()
        
        # Try to extract year from content
        year = extract_year_from_content(text_content)
        
        # If year not found in content, use file modification time
        if not year:
            mod_time = os.path.getmtime(text_path)
            year = datetime.fromtimestamp(mod_time).year
        
        # Format the output filename with year
        pdf_filename = f"{year}_{name_without_ext}.pdf"
        pdf_path = os.path.join(output_dir, pdf_filename)
        
        # Create a BytesIO buffer for the PDF
        buffer = BytesIO()
        
        # Create a PDF with reportlab
        c = canvas.Canvas(buffer, pagesize=letter)
        width, height = letter
        
        # Add filename as title
        c.setFont("Helvetica-Bold", 14)
        c.drawString(72, height - 72, f"{name_without_ext} ({year})")
        
        # Add content
        y_position = height - 100
        c.setFont("Helvetica", 10)
        
        # Process text line by line
        for line in text_content.split('\n'):
            if y_position < 72:  # Start a new page if we're near the bottom
                c.showPage()
                y_position = height - 72
                c.setFont("Helvetica", 10)
                
            # Add the line to the PDF
            if line.strip():  # Skip empty lines
                # Limit line length to avoid exceeding page width
                if len(line) > 90:
                    wrapped_text = [line[i:i+90] for i in range(0, len(line), 90)]
                    for wrapped_line in wrapped_text:
                        c.drawString(72, y_position, wrapped_line)
                        y_position -= 12
                else:
                    c.drawString(72, y_position, line)
                    y_position -= 12
        
        # Save and close the PDF document
        c.save()
        
        # Write directly from the buffer to the file
        buffer.seek(0)
        with open(pdf_path, 'wb') as output_file:
            output_file.write(buffer.getvalue())
        
        buffer.close()
        
        safe_print(f"Converted {text_path} to {pdf_path}")
        return True
    except Exception as e:
        safe_print(f"Error converting {text_path} to PDF: {e}")
        return False

def main():
    """Main function to orchestrate the text to PDF conversion."""
    start_time = time.time()
    args = parse_arguments()
    
    # Convert to absolute paths
    source_dir = os.path.abspath(args.source)
    dest_dir = os.path.abspath(args.destination)
    num_threads = args.threads
    
    # Check if source directory exists
    if not os.path.isdir(source_dir):
        print(f"Error: Source directory '{source_dir}' does not exist.")
        return
    
    # Create destination directory if it doesn't exist
    os.makedirs(dest_dir, exist_ok=True)
    
    # Find all text files
    text_files = find_text_files(source_dir)
    total_files = len(text_files)
    print(f"Found {total_files} text files to convert using {num_threads} threads")
    
    # Use ThreadPoolExecutor to process files in parallel
    successful_conversions = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Create a dictionary to track futures
        future_to_file = {
            executor.submit(convert_text_to_pdf, text_path, dest_dir): text_path
            for text_path in text_files
        }
        
        # Process completed futures as they finish
        for i, future in enumerate(concurrent.futures.as_completed(future_to_file), 1):
            text_path = future_to_file[future]
            try:
                if future.result():
                    successful_conversions += 1
            except Exception as e:
                safe_print(f"Exception processing {text_path}: {e}")
            
            # Print progress
            if i % 10 == 0 or i == total_files:
                safe_print(f"Progress: {i}/{total_files} files processed ({i/total_files*100:.1f}%)")
    
    elapsed_time = time.time() - start_time
    print(f"\nConversion completed in {elapsed_time:.2f} seconds:")
    print(f"- {successful_conversions} of {total_files} files successfully converted")
    print(f"- Average processing time: {elapsed_time/total_files:.4f} seconds per file")
    print(f"- Throughput: {total_files/elapsed_time:.2f} files per second")

if __name__ == "__main__":
    main()