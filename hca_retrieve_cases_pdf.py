import os
import argparse
from pathlib import Path
from bs4 import BeautifulSoup
import html2text
from PyPDF2 import PdfWriter
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from io import BytesIO

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Convert HTML files to PDF recursively.')
    parser.add_argument('--source', '-s', required=True, help='Source directory containing HTML files')
    parser.add_argument('--destination', '-d', required=True, help='Destination directory for PDF files')
    return parser.parse_args()

def find_html_files(source_dir):
    """Recursively find all HTML files in the source directory."""
    html_files = []
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.lower().endswith(('.html', '.htm')):
                html_files.append(os.path.join(root, file))
    return html_files

def create_output_directory(source_path, source_dir, dest_dir):
    """Create the corresponding output directory structure."""
    rel_path = os.path.dirname(os.path.relpath(source_path, source_dir))
    output_dir = os.path.join(dest_dir, rel_path)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def convert_html_to_pdf(html_path, output_dir):
    """Convert an HTML file to PDF using PyPDF2 and reportlab."""
    try:
        # Get the filename without extension
        filename = os.path.basename(html_path)
        name_without_ext = os.path.splitext(filename)[0]
        pdf_path = os.path.join(output_dir, f"{name_without_ext}.pdf")
        
        # Read HTML file
        with open(html_path, 'r', encoding='utf-8', errors='ignore') as file:
            html_content = file.read()
        
        # Parse HTML content
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Extract title or use filename if title not found
        title = soup.title.string if soup.title else name_without_ext
        
        # Convert HTML to plain text
        h = html2text.HTML2Text()
        h.ignore_links = False
        h.ignore_images = True
        text_content = h.handle(html_content)
        
        # Create a BytesIO buffer for the PDF
        buffer = BytesIO()
        
        # Create a PDF with reportlab
        c = canvas.Canvas(buffer, pagesize=letter)
        width, height = letter
        
        # Add title
        c.setFont("Helvetica-Bold", 14)
        c.drawString(72, height - 72, title[:60])
        
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
        
        print(f"Converted {html_path} to {pdf_path}")
        return True
    except Exception as e:
        print(f"Error converting {html_path} to PDF: {e}")
        return False

def main():
    """Main function to orchestrate the HTML to PDF conversion."""
    args = parse_arguments()
    
    # Convert to absolute paths
    source_dir = os.path.abspath(args.source)
    dest_dir = os.path.abspath(args.destination)
    
    # Check if source directory exists
    if not os.path.isdir(source_dir):
        print(f"Error: Source directory '{source_dir}' does not exist.")
        return
    
    # Create destination directory if it doesn't exist
    os.makedirs(dest_dir, exist_ok=True)
    
    # Find all HTML files
    html_files = find_html_files(source_dir)
    print(f"Found {len(html_files)} HTML files to convert")
    
    # Convert each HTML file to PDF
    successful_conversions = 0
    for html_path in html_files:
        output_dir = create_output_directory(html_path, source_dir, dest_dir)
        if convert_html_to_pdf(html_path, output_dir):
            successful_conversions += 1
    
    print(f"Conversion completed: {successful_conversions} of {len(html_files)} files successfully converted")

if __name__ == "__main__":
    main()