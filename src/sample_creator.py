import pandas as pd

df = pd.read_csv('~/Downloads/train_data_cleaned.csv')
df = df.sample(frac=1).reset_index(drop=True)

for rating in [1, 2, 3, 4, 5] :
    texts = list( df[df.y == rating].head(10).text )
    for i, text in enumerate(texts) :
        with open(f'../sample/rating0{rating}_sample0{i}.txt' ,'w') as file:
            file.write(text)
