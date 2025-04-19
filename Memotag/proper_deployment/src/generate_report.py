from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from datetime import datetime

def generate_analysis_report(output_file='analysis_report.pdf'):
    # Create the PDF document
    doc = SimpleDocTemplate(output_file, pagesize=letter)
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        alignment=1  # Center alignment
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        spaceAfter=12,
        textColor=colors.HexColor('#4e79a7')
    )
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['Normal'],
        fontSize=12,
        spaceAfter=12
    )
    
    # Content
    story = []
    
    # Title
    story.append(Paragraph("Cognitive Decline Detection Analysis Report", title_style))
    story.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Most Insightful Features
    story.append(Paragraph("Most Insightful Features", heading_style))
    story.append(Paragraph("""
    Our analysis identified several key speech features that proved most effective in detecting cognitive decline:
    """, body_style))
    
    features_data = [
        ['Feature', 'Description', 'Clinical Relevance'],
        ['Speech Rate', 'Words per second', 'Indicates processing speed and fluency'],
        ['Pause Duration', 'Average length of pauses', 'Reflects word-finding difficulties'],
        ['Pitch Variation', 'Standard deviation of pitch', 'Shows emotional expression and speech control'],
        ['Energy Level', 'Voice intensity variation', 'Indicates speech effort and control']
    ]
    
    t = Table(features_data, colWidths=[1.5*inch, 2.5*inch, 2.5*inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4e79a7')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f8f9fa')),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(t)
    story.append(Spacer(1, 20))
    
    # ML Methods
    story.append(Paragraph("Machine Learning Methods", heading_style))
    story.append(Paragraph("""
    The analysis pipeline employs several machine learning techniques:
    """, body_style))
    
    ml_methods = [
        ['Method', 'Purpose', 'Advantages'],
        ['Principal Component Analysis (PCA)', 'Dimensionality reduction and feature extraction', 'Identifies most significant speech patterns'],
        ['K-means Clustering', 'Pattern grouping and classification', 'Detects distinct speech patterns'],
        ['Feature Normalization', 'Data preprocessing', 'Ensures consistent feature scales'],
        ['Statistical Analysis', 'Feature significance testing', 'Validates clinical relevance']
    ]
    
    t = Table(ml_methods, colWidths=[2*inch, 2.5*inch, 2*inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4e79a7')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f8f9fa')),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(t)
    story.append(Spacer(1, 20))
    
    # Next Steps
    story.append(Paragraph("Next Steps for Clinical Robustness", heading_style))
    story.append(Paragraph("""
    To enhance the clinical utility of this tool, we recommend the following improvements:
    """, body_style))
    
    next_steps = [
        ['Area', 'Improvement', 'Expected Impact'],
        ['Data Collection', 'Larger, diverse clinical dataset', 'Better generalization'],
        ['Feature Engineering', 'Additional speech biomarkers', 'More comprehensive analysis'],
        ['Validation', 'Clinical trial with control groups', 'Statistical significance'],
        ['Integration', 'Electronic health record systems', 'Clinical workflow integration'],
        ['User Interface', 'Clinician-friendly dashboard', 'Better usability'],
        ['Regulatory', 'HIPAA compliance and FDA approval', 'Clinical deployment']
    ]
    
    t = Table(next_steps, colWidths=[1.5*inch, 2*inch, 2*inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4e79a7')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f8f9fa')),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(t)
    
    # Build the PDF
    doc.build(story)

if __name__ == '__main__':
    generate_analysis_report() 