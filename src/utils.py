import re
from datetime import datetime

def clean_text(text):
    if not text:
        return ""
    # Remove excessive newlines and whitespaces
    text = re.sub(r'\r\n', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def parse_vietnamese_date(date_str):
    if not date_str:
        return None
    
    date_str = date_str.strip()
    
    # Remove potential newline garbage like \n\n1,493\n3,794
    if '\n' in date_str:
        date_str = date_str.split('\n')[0].strip()
        
    # Remove prefixes like "Đã phát hiện ngày "
    date_str = re.sub(r'^[Đđ]ã phát hiện ngày\s+', '', date_str)
    
    # 1. Format: DD/MM/YYYY or D/M/YYYY (e.g. 20/11/2024, 29/5/2026)
    match_slash = re.search(r'(\d{1,2})/(\d{1,2})/(\d{4})', date_str)
    if match_slash:
        day = int(match_slash.group(1))
        month = int(match_slash.group(2))
        year = int(match_slash.group(3))
        try:
            return f"{year:04d}-{month:02d}-{day:02d}"
        except ValueError:
            pass
            
    # 2. Format: "DD Tháng MM, YYYY" (e.g. 29 Tháng 5, 2026)
    match_thang = re.search(r'(\d{1,2})\s+[Tt]háng\s+(\d{1,2}),\s+(\d{4})', date_str)
    if match_thang:
        day = int(match_thang.group(1))
        month = int(match_thang.group(2))
        year = int(match_thang.group(3))
        try:
            return f"{year:04d}-{month:02d}-{day:02d}"
        except ValueError:
            pass
            
    return None
