import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
import warnings
warnings.filterwarnings('ignore')

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR   = os.path.join(BASE_DIR, 'data', 'raw')
PLOTS_DIR = os.path.join(BASE_DIR, 'results', 'eda_plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
def extract_enron_body(raw_message: str) -> str:
    """Strip email headers, keep only the body text."""
    if not isinstance(raw_message, str):
        return ''
    # Split on the first blank line — everything after is the body
    parts = re.split(r'\n\n', raw_message, maxsplit=1)
    return parts[1].strip() if len(parts) > 1 else raw_message.strip()

def text_length(text: str) -> int:
    return len(str(text).split())

# ═══════════════════════════════════════════════════════════════════════════════
# 1. LOAD DATASETS
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("LOADING DATASETS")
print("=" * 60)

# Enron — legitimate emails only, sample 10k to keep memory manageable
print("\n[1/4] Loading Enron...")
enron_raw = pd.read_csv(os.path.join(RAW_DIR, 'enron.csv'))
enron = pd.DataFrame()
enron['text']  = enron_raw['message'].apply(extract_enron_body)
enron['label'] = 0          # all legitimate
enron['source'] = 'enron'
enron = enron[enron['text'].str.len() > 20].sample(
    n=min(10000, len(enron)), random_state=42
).reset_index(drop=True)
print(f"   Enron loaded  → {len(enron):,} rows  | label=0 (legitimate)")

# Phishing
print("\n[2/4] Loading Phishing...")
phishing_raw = pd.read_csv(os.path.join(RAW_DIR, 'phishing.csv'))
phishing = pd.DataFrame()
phishing['text']   = phishing_raw['text_combined']
phishing['label']  = phishing_raw['label']
phishing['source'] = 'phishing'
phishing = phishing.dropna(subset=['text']).reset_index(drop=True)
print(f"   Phishing loaded → {len(phishing):,} rows  | labels: {phishing['label'].value_counts().to_dict()}")

# Spam Email
print("\n[3/4] Loading Spam Email...")
spam_raw = pd.read_csv(os.path.join(RAW_DIR, 'spam_email.csv'))
spam = pd.DataFrame()
spam['text']   = spam_raw['body']
spam['label']  = spam_raw['label']
spam['source'] = 'spam_email'
spam = spam.dropna(subset=['text']).reset_index(drop=True)
print(f"   Spam loaded    → {len(spam):,} rows  | labels: {spam['label'].value_counts().to_dict()}")

# Synthetic
print("\n[4/4] Loading Synthetic...")
synthetic_raw = pd.read_csv(os.path.join(RAW_DIR, 'synthetic.csv'))
synthetic = pd.DataFrame()
synthetic['text']   = synthetic_raw['text']
synthetic['label']  = synthetic_raw['label']
synthetic['source'] = 'synthetic'
synthetic = synthetic.dropna(subset=['text']).reset_index(drop=True)
print(f"   Synthetic loaded → {len(synthetic):,} rows | labels: {synthetic['label'].value_counts().to_dict()}")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. MERGE ALL DATASETS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("MERGING DATASETS")
print("=" * 60)

df = pd.concat([enron, phishing, spam, synthetic], ignore_index=True)
df = df[['text', 'label', 'source']]
df = df.dropna(subset=['text', 'label'])
df['text']  = df['text'].astype(str)
df['label'] = df['label'].astype(int)

print(f"\nTotal rows after merge  : {len(df):,}")
print(f"Total attack (1)        : {(df['label'] == 1).sum():,}")
print(f"Total legitimate (0)    : {(df['label'] == 0).sum():,}")
print(f"\nSource breakdown:")
print(df['source'].value_counts())

# ═══════════════════════════════════════════════════════════════════════════════
# 3. EDA — CLASS DISTRIBUTION
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("EDA — CLASS DISTRIBUTION")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Overall label distribution
label_counts = df['label'].value_counts()
axes[0].bar(['Legitimate (0)', 'Attack (1)'],
            label_counts.values,
            color=['#3b82f6', '#ef4444'], edgecolor='black')
axes[0].set_title('Overall Class Distribution', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Count')
for i, v in enumerate(label_counts.values):
    axes[0].text(i, v + 100, f'{v:,}', ha='center', fontweight='bold')

# Per source distribution
source_label = df.groupby(['source', 'label']).size().unstack(fill_value=0)
source_label.plot(kind='bar', ax=axes[1],
                  color=['#3b82f6', '#ef4444'], edgecolor='black')
axes[1].set_title('Class Distribution Per Source', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Source')
axes[1].set_ylabel('Count')
axes[1].legend(['Legitimate (0)', 'Attack (1)'])
axes[1].tick_params(axis='x', rotation=30)

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'class_distribution.png'), dpi=150)
plt.show()
print(f"   Saved → results/eda_plots/class_distribution.png")

# ═══════════════════════════════════════════════════════════════════════════════
# 4. EDA — TEXT LENGTH ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("EDA — TEXT LENGTH ANALYSIS")
print("=" * 60)

df['word_count'] = df['text'].apply(text_length)

print(f"\nWord count statistics:")
print(df.groupby('label')['word_count'].describe().round(2))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Word count distribution
for label, color, name in [(0, '#3b82f6', 'Legitimate'),
                             (1, '#ef4444', 'Attack')]:
    subset = df[df['label'] == label]['word_count']
    subset_clipped = subset.clip(upper=500)
    axes[0].hist(subset_clipped, bins=50, alpha=0.6,
                 color=color, label=name, edgecolor='black')
axes[0].set_title('Word Count Distribution (clipped at 500)',
                   fontsize=13, fontweight='bold')
axes[0].set_xlabel('Word Count')
axes[0].set_ylabel('Frequency')
axes[0].legend()

# Boxplot
df_plot = df[df['word_count'] < 500].copy()
df_plot['Class'] = df_plot['label'].map({0: 'Legitimate', 1: 'Attack'})
sns.boxplot(data=df_plot, x='Class', y='word_count',
            palette={'Legitimate': '#3b82f6', 'Attack': '#ef4444'},
            ax=axes[1])
axes[1].set_title('Word Count Boxplot', fontsize=13, fontweight='bold')
axes[1].set_ylabel('Word Count')

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'text_length.png'), dpi=150)
plt.show()
print(f"   Saved → results/eda_plots/text_length.png")

# ═══════════════════════════════════════════════════════════════════════════════
# 5. EDA — MISSING VALUES & DATA QUALITY
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("EDA — DATA QUALITY CHECK")
print("=" * 60)

print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nDuplicate rows    : {df.duplicated(subset=['text']).sum():,}")
print(f"Empty text rows   : {(df['text'].str.strip() == '').sum():,}")
print(f"Very short (<5w)  : {(df['word_count'] < 5).sum():,}")

# ═══════════════════════════════════════════════════════════════════════════════
# 6. SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"\nFinal merged dataset shape : {df.shape}")
print(f"Attack samples (1)         : {(df['label']==1).sum():,} "
      f"({(df['label']==1).mean()*100:.1f}%)")
print(f"Legitimate samples (0)     : {(df['label']==0).sum():,} "
      f"({(df['label']==0).mean()*100:.1f}%)")
print(f"Imbalance ratio            : "
      f"1:{(df['label']==0).sum() // max((df['label']==1).sum(),1)}")
print("\nEDA complete. Ready for preprocessing.")