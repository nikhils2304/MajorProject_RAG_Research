"""
Dataset Quality Audit Script
Compares PyPDF2 vs pdfplumber extraction for Tamil and Bengali PDFs.
Assesses whether PDFs have enough content for 6 QA pairs.
"""
import os, sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
import config

import PyPDF2
import pdfplumber

def extract_pypdf2(pdf_path):
    """Extract text using PyPDF2."""
    try:
        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            text = ""
            for page in reader.pages:
                text += (page.extract_text() or "") + "\n"
        return text.strip()
    except Exception as e:
        return f"ERROR: {e}"

def extract_pdfplumber(pdf_path):
    """Extract text using pdfplumber."""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = ""
            for page in pdf.pages:
                text += (page.extract_text() or "") + "\n"
        return text.strip()
    except Exception as e:
        return f"ERROR: {e}"

def audit_language(language):
    folder = config.DATASET_DIR / config.LANG_TO_FOLDER[language]
    pdf_files = sorted([f for f in folder.iterdir() if f.suffix.lower() == '.pdf'])
    
    print(f"\n{'='*60}")
    print(f"  DATASET AUDIT: {language}")
    print(f"{'='*60}")
    print(f"  Folder: {folder}")
    print(f"  Total PDFs: {len(pdf_files)}")
    
    # File sizes
    sizes = [f.stat().st_size for f in pdf_files]
    print(f"  Avg file size: {sum(sizes)/len(sizes)/1024:.1f} KB")
    print(f"  Min file size: {min(sizes)/1024:.1f} KB")
    print(f"  Max file size: {max(sizes)/1024:.1f} KB")
    
    # Sample 5 PDFs (first, last, and 3 random middle ones)
    sample_indices = [0, len(pdf_files)//4, len(pdf_files)//2, 3*len(pdf_files)//4, len(pdf_files)-1]
    sample_pdfs = [pdf_files[i] for i in sample_indices if i < len(pdf_files)]
    
    print(f"\n  Sampling {len(sample_pdfs)} PDFs for quality check...")
    
    quality_scores = []
    
    for pdf_path in sample_pdfs:
        print(f"\n  --- {pdf_path.name} ({pdf_path.stat().st_size/1024:.1f} KB) ---")
        
        # PyPDF2 extraction
        text_pypdf2 = extract_pypdf2(pdf_path)
        # pdfplumber extraction  
        text_plumber = extract_pdfplumber(pdf_path)
        
        print(f"  PyPDF2:     {len(text_pypdf2)} chars | Words: ~{len(text_pypdf2.split())}")
        print(f"  pdfplumber: {len(text_plumber)} chars | Words: ~{len(text_plumber.split())}")
        
        # Show first 300 chars of each
        print(f"\n  [PyPDF2 Preview]:")
        preview = text_pypdf2[:300].replace('\n', ' ')
        print(f"  {preview}")
        
        print(f"\n  [pdfplumber Preview]:")
        preview = text_plumber[:300].replace('\n', ' ')
        print(f"  {preview}")
        
        # Quality Assessment
        # Check for character fragmentation (spaces between chars)
        # A sign of fragmentation: high ratio of single-char "words"
        words_pypdf2 = text_pypdf2.split()
        single_char_ratio_pypdf2 = sum(1 for w in words_pypdf2 if len(w) <= 2) / max(len(words_pypdf2), 1)
        
        words_plumber = text_plumber.split()
        single_char_ratio_plumber = sum(1 for w in words_plumber if len(w) <= 2) / max(len(words_plumber), 1)
        
        print(f"\n  [Quality Metrics]:")
        print(f"  PyPDF2 single-char word ratio:     {single_char_ratio_pypdf2:.2%}")
        print(f"  pdfplumber single-char word ratio:  {single_char_ratio_plumber:.2%}")
        
        # Can 6 QA pairs be generated?
        word_count = max(len(words_pypdf2), len(words_plumber))
        qa_feasible = "YES" if word_count > 100 else ("MAYBE" if word_count > 50 else "NO")
        print(f"  Enough content for 6 QA pairs?:     {qa_feasible} ({word_count} words)")
        
        quality_scores.append({
            'file': pdf_path.name,
            'pypdf2_chars': len(text_pypdf2),
            'plumber_chars': len(text_plumber),
            'pypdf2_frag': single_char_ratio_pypdf2,
            'plumber_frag': single_char_ratio_plumber,
            'qa_feasible': qa_feasible
        })
    
    # Summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY for {language}")
    print(f"{'='*60}")
    avg_frag_pypdf2 = sum(s['pypdf2_frag'] for s in quality_scores) / len(quality_scores)
    avg_frag_plumber = sum(s['plumber_frag'] for s in quality_scores) / len(quality_scores)
    print(f"  Avg PyPDF2 fragmentation:     {avg_frag_pypdf2:.2%}")
    print(f"  Avg pdfplumber fragmentation: {avg_frag_plumber:.2%}")
    feasible = sum(1 for s in quality_scores if s['qa_feasible'] == 'YES')
    print(f"  PDFs adequate for 6 QA pairs: {feasible}/{len(quality_scores)} sampled")
    
    if avg_frag_plumber < avg_frag_pypdf2:
        improvement = (avg_frag_pypdf2 - avg_frag_plumber) / max(avg_frag_pypdf2, 0.01)
        print(f"  pdfplumber improvement:       {improvement:.0%} less fragmentation")
    else:
        print(f"  pdfplumber: No improvement over PyPDF2")

if __name__ == "__main__":
    audit_language("Tamil")
    audit_language("Bengali")
