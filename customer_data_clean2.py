import os
import numpy as np
import pandas as pd

# Work relative to the folder this script lives in (not VS Code's current folder)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------
# Load the messy CSV dataset
#  - first 3 lines are a report title / metadata / blank line -> skip
#  - one row has an extra unexpected column -> read 19 columns, drop the extra
# CHANGE THIS to your CSV's file name (must be in the same folder as this script)
# ---------------------------------------------------------------
INPUT_FILE = None   # leave None to auto-detect, or set e.g. 'my_file.csv'
if INPUT_FILE is None:
    for name in ['messy_ecommerce_transactions.csv', 'messy_ecommerce_transactions__1_.csv']:
        if os.path.exists(os.path.join(BASE_DIR, name)):
            INPUT_FILE = name
            break
if INPUT_FILE is None or not os.path.exists(os.path.join(BASE_DIR, INPUT_FILE)):
    raise FileNotFoundError(
        "Put the messy CSV in this folder: " + BASE_DIR + "\n"
        "Files found here: " + str(os.listdir(BASE_DIR)))
INPUT_FILE = os.path.join(BASE_DIR, INPUT_FILE)
COLS = ['Order_ID', 'Customer_ID', 'Date', 'Age', 'Gender', 'City',
        'Product_Category', 'Unit_Price', 'Quantity', 'Discount_Amount',
        'Total_Amount', 'Payment_Method', 'Device_Type',
        'Session_Duration_Minutes', 'Pages_Viewed', 'Is_Returning_Customer',
        'Delivery_Time_Days', 'Customer_Rating']
df = pd.read_csv(INPUT_FILE, skiprows=3, header=None,
                 names=COLS + ['Extra'], dtype=str)
df = df.drop(columns='Extra')
print(df.head(10))
print("Raw shape:", df.shape)

# 0. Remove junk rows: repeated header row and the GRAND TOTAL footer
df = df[~df['Order_ID'].str.strip().str.upper().isin(['ORDER_ID', 'GRAND TOTAL'])].copy()

# Trim whitespace in every text column and turn placeholders into NaN
df = df.apply(lambda s: s.str.strip())
df = df.replace({'': np.nan, 'null': np.nan, 'NULL': np.nan, 'None': np.nan,
                 '#N/A': np.nan, 'N/A': np.nan, 'nan': np.nan, 'ERROR': np.nan})


# Helper: turn "$54.28", "54,28 TRY", "1784,75", "244,9" etc. into floats
def to_number(series):
    s = (series.astype(str)
         .str.replace('TRY', '', regex=False)
         .str.replace('$', '', regex=False)
         .str.strip())
    # comma used as decimal separator (no dot present) -> dot
    comma_decimal = s.str.contains(',', regex=False) & ~s.str.contains('.', regex=False)
    s = s.where(~comma_decimal, s.str.replace(',', '.', regex=False))
    s = s.str.replace(',', '', regex=False)          # leftover thousand separators
    return pd.to_numeric(s, errors='coerce')


# 1. Remove duplicate orders (standardise the ID first)
df['Order_ID'] = df['Order_ID'].str.upper()
df = df.drop_duplicates(subset=['Order_ID'], keep='first')

# 2. Standardise Customer_ID  (cust-01337 -> CUST_01337)
df['Customer_ID'] = df['Customer_ID'].str.upper().str.replace('-', '_', regex=False)

# 3. Parse and standardise Date into datetime (YYYY-MM-DD)
#    Formats found:  2023-01-01 | 01/01/2023 (DD/MM) | 01.01.2023 (DD.MM)
#                    01-01-2023 (MM-DD)  | Jan 1, 2023
d = df['Date']
parsed = pd.Series(pd.NaT, index=df.index, dtype='datetime64[ns]')
patterns = [
    (r'^\d{4}-\d{2}-\d{2}$',  '%Y-%m-%d'),
    (r'^\d{2}/\d{2}/\d{4}$',  '%d/%m/%Y'),
    (r'^\d{2}\.\d{2}\.\d{4}$', '%d.%m.%Y'),
    (r'^\d{2}-\d{2}-\d{4}$',  '%m-%d-%Y'),
    (r'^[A-Za-z]{3} \d{1,2}, \d{4}$', '%b %d, %Y'),
]
for regex, fmt in patterns:
    mask = d.str.match(regex, na=False)
    parsed[mask] = pd.to_datetime(d[mask], format=fmt, errors='coerce')
df['Date'] = parsed

# 4. Clean Age (extract digits, keep between 18 and 100, impute median)
df['Age'] = pd.to_numeric(df['Age'].str.extract(r'(-?\d+)')[0], errors='coerce')
df.loc[(df['Age'] < 18) | (df['Age'] > 100), 'Age'] = np.nan
df['Age'] = df['Age'].fillna(df['Age'].median()).astype(int)

# 5. Clean & Standardise Gender
df['Gender'] = df['Gender'].astype(str).str.strip().str.upper()
gender_map = {'MALE': 'Male', 'M': 'Male',
              'FEMALE': 'Female', 'F': 'Female',
              'OTHER': 'Other'}
df['Gender'] = df['Gender'].map(gender_map).fillna('Unknown')

# 6. Clean City (strip, title case, fix spelling variants)
df['City'] = df['City'].str.strip().str.title()
df['City'] = df['City'].replace({'Istambul': 'Istanbul', 'G.Antep': 'Gaziantep'})
df['City'] = df['City'].fillna('Unknown')

# 7. Clean Product_Category
df['Product_Category'] = df['Product_Category'].str.strip().str.lower()
df['Product_Category'] = df['Product_Category'].replace(
    {'home and garden': 'home & garden', 'home/garden': 'home & garden'})
df['Product_Category'] = df['Product_Category'].str.title().fillna('Unknown')

# 8. Clean money columns (strip $, TRY, fix decimal commas)
df['Unit_Price'] = to_number(df['Unit_Price'])
df['Discount_Amount'] = to_number(df['Discount_Amount'])
df['Total_Amount'] = to_number(df['Total_Amount'])

df.loc[df['Unit_Price'] <= 0, 'Unit_Price'] = np.nan
df.loc[df['Discount_Amount'] < 0, 'Discount_Amount'] = np.nan
df['Discount_Amount'] = df['Discount_Amount'].fillna(0.0)          # no discount recorded
df.loc[df['Total_Amount'] == 99999.99, 'Total_Amount'] = np.nan    # placeholder value

# 9. Clean Quantity (words -> numbers, no zero/negative quantities)
df['Quantity'] = (df['Quantity'].str.lower()
                  .replace({'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5'}))
df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')
df.loc[df['Quantity'] <= 0, 'Quantity'] = np.nan

# 9b. Repair using the business rule:  Total = Unit_Price * Quantity - Discount
#     bad Quantity -> derive from Total ; bad Total -> recompute from Quantity
qty_from_total = ((df['Total_Amount'] + df['Discount_Amount']) / df['Unit_Price']).round()
df['Quantity'] = df['Quantity'].fillna(qty_from_total)
df['Total_Amount'] = df['Total_Amount'].fillna(
    (df['Unit_Price'] * df['Quantity'] - df['Discount_Amount']).round(2))
df['Quantity'] = df['Quantity'].astype(int)

# Flag (don't silently change) rows where the three money columns disagree
expected = df['Unit_Price'] * df['Quantity'] - df['Discount_Amount']
df['Needs_Review'] = (expected - df['Total_Amount']).abs() > 0.05

# 10. Standardise Payment_Method
df['Payment_Method'] = df['Payment_Method'].str.strip().replace(
    {'CC': 'Credit Card', 'COD': 'Cash on Delivery'}).fillna('Unknown')

# 11. Standardise Device_Type
df['Device_Type'] = df['Device_Type'].str.strip().str.title().fillna('Unknown')

# 12. Numeric behaviour columns ("4 mins", "#N/A" ...) -> numbers, median impute
for col in ['Session_Duration_Minutes', 'Pages_Viewed']:
    df[col] = pd.to_numeric(df[col].str.extract(r'(\d+\.?\d*)')[0], errors='coerce')
    df[col] = df[col].fillna(df[col].median()).astype(int)

# 13. Is_Returning_Customer -> True / False
bool_map = {'YES': True, 'TRUE': True, '1': True,
            'NO': False, 'FALSE': False, '0': False}
df['Is_Returning_Customer'] = (df['Is_Returning_Customer'].astype(str).str.upper()
                               .map(bool_map).astype('boolean'))

# 14. Delivery_Time_Days ("5 days" -> 5, negatives invalid -> median)
df['Delivery_Time_Days'] = pd.to_numeric(
    df['Delivery_Time_Days'].str.extract(r'(-?\d+)')[0], errors='coerce')
df.loc[df['Delivery_Time_Days'] < 0, 'Delivery_Time_Days'] = np.nan
df['Delivery_Time_Days'] = df['Delivery_Time_Days'].fillna(
    df['Delivery_Time_Days'].median()).astype(int)

# 15. Customer_Rating ("5/5" -> 5, valid range 1-5, leave unknown ratings blank)
df['Customer_Rating'] = pd.to_numeric(
    df['Customer_Rating'].str.extract(r'^(\d+)')[0], errors='coerce')
df.loc[~df['Customer_Rating'].between(1, 5), 'Customer_Rating'] = np.nan
df['Customer_Rating'] = df['Customer_Rating'].astype('Int64')

# Sort and export cleaned dataset
df = df.sort_values(['Date', 'Order_ID']).reset_index(drop=True)
df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
df.to_csv(os.path.join(BASE_DIR, 'cleaned_ecommerce_transactions.csv'), index=False)

print("Clean shape:", df.shape)
print(df.isna().sum())


# ===============================================================
# 16. Ask for statistics from the terminal
# ===============================================================
def average_rating():
    return df['Customer_Rating'].astype('float').mean()          # blank ratings are ignored


def average_shopping_per_person():
    # total spent by each customer, then the average across customers
    spend_per_customer = df.groupby('Customer_ID')['Total_Amount'].sum()
    return spend_per_customer.mean()


def average_age():
    return df['Age'].mean()


def average_delivery_time():
    return df['Delivery_Time_Days'].mean()


stats = {
    '1': ('Average rating',              lambda: f"{average_rating():.2f} / 5"),
    '2': ('Average shopping per person', lambda: f"{average_shopping_per_person():,.2f} TRY"),
    '3': ('Average age',                 lambda: f"{average_age():.1f} years"),
    '4': ('Average delivery time',       lambda: f"{average_delivery_time():.1f} days"),
}
# words you can type instead of the number
aliases = {'rating': '1', 'shopping': '2', 'spend': '2', 'spending': '2',
           'age': '3', 'delivery': '4'}


def show_menu():
    print("\nWhat do you want to know?")
    for key, (label, _) in stats.items():
        print(f"  {key}. {label}")
    print("  all. Show everything")
    print("  quit. Exit")


print(f"\nData ready: {len(df)} orders from {df['Customer_ID'].nunique()} customers.")
show_menu()

while True:
    try:
        choice = input("\nYour choice: ").strip().lower()
    except EOFError:
        break
    choice = aliases.get(choice, choice)

    if choice in ('q', 'quit', 'exit'):
        break
    elif choice == 'all':
        for label, func in stats.values():
            print(f"{label}: {func()}")
    elif choice in stats:
        label, func = stats[choice]
        print(f"{label}: {func()}")
    else:
        print("Please type 1, 2, 3, 4, a word like 'age', 'all' or 'quit'.")
        show_menu()
