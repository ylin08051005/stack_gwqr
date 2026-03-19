import pandas as pd
import re

file_path = '/Users/ylin/Documents/stack_gwqr/data/raw/realprice_combined_all.csv'
df = pd.read_csv(file_path)

cn_nums = {
    '一': 1, '二': 2, '三': 3, '四': 4, '五': 5, '六': 6, '七': 7, '八': 8, '九': 9, '十': 10,
    '十一': 11, '十二': 12, '十三': 13, '十四': 14, '十五': 15, '十六': 16, '十七': 17, '十八': 18, '十九': 19, '二十': 20,
    '二十一': 21, '二十二': 22, '二十三': 23, '二十四': 24, '二十五': 25, '二十六': 26, '二十七': 27, '二十八': 28, '二十九': 29, '三十': 30,
    '三十一': 31, '三十二': 32, '三十三': 33, '三十四': 34, '三十五': 35, '三十六': 36, '三十七': 37, '三十八': 38, '三十九': 39, '四十': 40
}

def convert_floor(row):
    text = str(row['移轉層次'])
    total_floor = row['總樓層數']

    if text == '全':
        if isinstance(total_floor, str):
            match = re.search('[一二三四五六七八九十]+', str(total_floor))
            return cn_nums.get(match.group(), 0) if match else 0
        return total_floor
    
    normal_floors = re.findall(r'(?<!地下)([一二三四五六七八九十]+)層', text)
    num_list = [cn_nums.get(f, 0) for f in normal_floors]
    
    underground_floors = re.findall(r'地下([一二三四五六七八九十]+)層', text)
    num_list += [-cn_nums.get(f, 0) for f in underground_floors]
    
    if num_list:
        return max(num_list)
    
    if '地下' in text:
        return -1
        
    return 0

df['count_floor'] = df.apply(convert_floor, axis=1)

cols = df.columns.tolist()
idx = cols.index('移轉層次')
new_cols = cols[:idx+1] + ['count_floor'] + [c for c in cols[idx+1:] if c != 'count_floor']
df = df[new_cols]

output_path = '/Users/ylin/Documents/stack_gwqr/data/raw/realprice_cleaned_floors.csv'
df.to_csv(output_path, index=False, encoding='utf-8-sig')