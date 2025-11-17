import pandas as pd
from sklearn.model_selection import train_test_split

def load_news_data(fake_path: str, true_path: str):
    fake_df = pd.read_csv(fake_path)
    true_df = pd.read_csv(true_path)

    fake_df = fake_df[['title', 'text']].copy()
    true_df = true_df[['title', 'text']].copy()

    fake_df['label'] = 0
    true_df['label'] = 1

    df = pd.concat([fake_df, true_df], ignore_index=True)
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    # Combine title + text into one string
    df['content'] = (df['title'].fillna('') + ' ' + df['text'].fillna('')).str.strip()

    # Drop rows with missing content
    df = df[df['content'].str.len() > 0]

    train_df, val_df = train_test_split(
        df[['content', 'label']],
        test_size=0.2,
        stratify=df['label'],
        random_state=42
    )

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)