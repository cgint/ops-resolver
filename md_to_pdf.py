#!/usr/bin/env python3
# /// script
# dependencies = [
#     "markdown2>=2.4.0",
#     "weasyprint>=62.0",
#     "Pygments>=2.17.0"
# ]
# requires-python = ">=3.11"
# ///

import argparse
import os
import markdown2  # type: ignore[import-not-found]
from weasyprint import HTML, CSS  # type: ignore[import-not-found]
from pygments.formatters import HtmlFormatter

def create_pdf(markdown_file: str, pdf_file: str) -> None:
    """Converts a Markdown file to a PDF file with syntax highlighting."""

    if not os.path.exists(markdown_file):
        print(f"Error: Input file '{markdown_file}' not found.")
        return

    # Read markdown content
    with open(markdown_file, 'r', encoding='utf-8') as f:
        md_content = f.read()

    # Get Pygments CSS for syntax highlighting
    formatter = HtmlFormatter(style='default')
    pygments_css = formatter.get_style_defs('.codehilite')  # type: ignore[no-untyped-call]

    # CSS for overall styling
    main_css = f"""
    @page {{
        size: A4;
        margin: 2cm;
    }}
    body {{
        font-family: sans-serif;
        line-height: 1.6;
        font-size: 11px;
    }}
    h1, h2, h3, h4, h5, h6 {{
        font-weight: bold;
        margin-top: 1.5em;
        margin-bottom: 0.5em;
    }}
    h1 {{ font-size: 2em; }}
    h2 {{ font-size: 1.75em; }}
    h3 {{ font-size: 1.5em; }}
    pre {{
        border: 1px solid #ddd;
        border-radius: 4px;
        padding: 10px;
        overflow-x: auto;
        font-size: 0.9em;
    }}
    code {{
        font-family: monospace;
    }}
    .codehilite {{
        background: #f8f8f8;
    }}
    table {{
        border-collapse: collapse;
        width: 100%;
    }}
    th, td {{
        border: 1px solid #dddddd;
        text-align: left;
        padding: 8px;
    }}
    th {{
        background-color: #f2f2f2;
    }}
    {pygments_css}
    """

    # Convert markdown to HTML with extras
    markdowner = markdown2.Markdown(
        extras={
            "fenced-code-blocks": {"cssclass": "codehilite"},
            "code-friendly": None,
            "cuddled-lists": None,
            "tables": None,
        }
    )
    html_content = markdowner.convert(md_content)

    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>{os.path.basename(pdf_file)}</title>
    </head>
    <body>
        {html_content}
    </body>
    </html>
    """

    # Generate PDF
    html = HTML(string=full_html, base_url=os.path.dirname(os.path.abspath(markdown_file)))
    css = CSS(string=main_css)
    
    html.write_pdf(
        pdf_file,
        stylesheets=[css]
    )
    print(f"Successfully converted '{markdown_file}' to '{pdf_file}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert a Markdown file to a PDF with syntax highlighting.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "markdown_file",
        help="Input Markdown file path."
    )
    parser.add_argument(
        "-o", "--output",
        dest="pdf_file",
        help="Output PDF file path. If not provided, it will be the input filename with a .pdf extension."
    )

    args = parser.parse_args()

    if not args.pdf_file:
        args.pdf_file = os.path.splitext(args.markdown_file)[0] + ".pdf"

    create_pdf(args.markdown_file, args.pdf_file)
