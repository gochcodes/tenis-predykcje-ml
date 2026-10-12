import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

#wczytywanie
df = pd.read_csv('atp_tennis_z_elo.csv', low_memory=False)

#kto wygral
w1 = df['Winner'] == df['Player_1']
w2 = df['Winner'] == df['Player_2']

#elo ogolne
df['p1_elo'] = np.where(w1, df['W_ELO_ogolne'], df['L_ELO_ogolne'])
df['p2_elo'] = np.where(w2, df['W_ELO_ogolne'], df['L_ELO_ogolne'])
df['diff_elo'] = df['p1_elo'] - df['p2_elo']

#elo hard
df['p1_hard'] = np.where(w1, df['W_ELO_hard'], df['L_ELO_hard'])
df['p2_hard'] = np.where(w2, df['W_ELO_hard'], df['L_ELO_hard'])
df['diff_hard'] = df['p1_hard'] - df['p2_hard']

#elo clay
df['p1_clay'] = np.where(w1, df['W_ELO_clay'], df['L_ELO_clay'])
df['p2_clay'] = np.where(w2, df['W_ELO_clay'], df['L_ELO_clay'])
df['diff_clay'] = df['p1_clay'] - df['p2_clay']

#elo grass
df['p1_grass'] = np.where(w1, df['W_ELO_grass'], df['L_ELO_grass'])
df['p2_grass'] = np.where(w2, df['W_ELO_grass'], df['L_ELO_grass'])
df['diff_grass'] = df['p1_grass'] - df['p2_grass']

#elo carpet
df['p1_carpet'] = np.where(w1, df['W_ELO_carpet'], df['L_ELO_carpet'])
df['p2_carpet'] = np.where(w2, df['W_ELO_carpet'], df['L_ELO_carpet'])
df['diff_carpet'] = df['p1_carpet'] - df['p2_carpet']

#streaki
df['p1_streak'] = np.where(w1, df['W_Streak'], df['L_Streak'])
df['p2_streak'] = np.where(w2, df['W_Streak'], df['L_Streak'])
df['diff_streak'] = df['p1_streak'] - df['p2_streak']

df['p1_str_log'] = np.where(w1, df['W_Streak_Log'], df['L_Streak_Log'])
df['p2_str_log'] = np.where(w2, df['W_Streak_Log'], df['L_Streak_Log'])
df['diff_str_log'] = df['p1_str_log'] - df['p2_str_log']

#akt elo
df['akt_p1_elo'] = 1500.0
df['akt_p2_elo'] = 1500.0

df.loc[df['Surface'] == 'Hard', 'akt_p1_elo'] = df['p1_hard']
df.loc[df['Surface'] == 'Hard', 'akt_p2_elo'] = df['p2_hard']
df.loc[df['Surface'] == 'Clay', 'akt_p1_elo'] = df['p1_clay']
df.loc[df['Surface'] == 'Clay', 'akt_p2_elo'] = df['p2_clay']
df.loc[df['Surface'] == 'Grass', 'akt_p1_elo'] = df['p1_grass']
df.loc[df['Surface'] == 'Grass', 'akt_p2_elo'] = df['p2_grass']
df.loc[df['Surface'] == 'Carpet', 'akt_p1_elo'] = df['p1_carpet']
df.loc[df['Surface'] == 'Carpet', 'akt_p2_elo'] = df['p2_carpet']

df['akt_diff'] = df['akt_p1_elo'] - df['akt_p2_elo']

df['target'] = np.where(w1, 1, 0)

#wybor kolumn do modelu
kolumny = [
    'diff_elo', 'p1_elo', 'p2_elo',
    'diff_hard', 'p1_hard', 'p2_hard',
    'diff_clay', 'p1_clay', 'p2_clay',
    'diff_grass', 'p1_grass', 'p2_grass',
    'diff_carpet', 'p1_carpet', 'p2_carpet',
    'diff_streak', 'p1_streak', 'p2_streak',
    'diff_str_log', 'p1_str_log', 'p2_str_log',
    'akt_diff', 'akt_p1_elo', 'akt_p2_elo'
]

#czyszczenie
df = df.dropna(subset=kolumny + ['target'])
X = df[kolumny]
y = df['target']

#podzial i nauka
s = int(len(df) * 0.8)
xtrain, xtest = X.iloc[:s], X.iloc[s:]
ytrain, ytest = y.iloc[:s], y.iloc[s:]

model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=12)
model.fit(xtrain, ytrain)

#symulacja
elo_df = pd.read_csv('aktualne_elo_zawodnikow.csv').set_index('Player')

while True:
    g1 = input("wpisz gracza 1: ")
    if g1 == 'q': break
    g2 = input("wpisz gracza 2: ")
    if g2 == 'q': break
    
    print("1-hard, 2-clay, 3-grass, 4-carpet")
    naw = input("wybierz nawierzchnie: ")

    if g1 not in elo_df.index or g2 not in elo_df.index:
        print("blad - nie ma gracza")
        continue

    e1_og = elo_df.loc[g1, 'ELOogolne']
    e1_h = elo_df.loc[g1, 'ELOhard']
    e1_c = elo_df.loc[g1, 'ELOclay']
    e1_g = elo_df.loc[g1, 'ELOgrass']
    e1_ca = elo_df.loc[g1, 'ELOcarpet']

    e2_og = elo_df.loc[g2, 'ELOogolne']
    e2_h = elo_df.loc[g2, 'ELOhard']
    e2_c = elo_df.loc[g2, 'ELOclay']
    e2_g = elo_df.loc[g2, 'ELOgrass']
    e2_ca = elo_df.loc[g2, 'ELOcarpet']

    #ifowanie nawierzchni
    if naw == '1':
        e1_akt, e2_akt = e1_h, e2_h
    elif naw == '2':
        e1_akt, e2_akt = e1_c, e2_c
    elif naw == '3':
        e1_akt, e2_akt = e1_g, e2_g
    else:
        e1_akt, e2_akt = e1_ca, e2_ca

    dane = pd.DataFrame([{
        'diff_elo': e1_og - e2_og, 'p1_elo': e1_og, 'p2_elo': e2_og,
        'diff_hard': e1_h - e2_h, 'p1_hard': e1_h, 'p2_hard': e2_h,
        'diff_clay': e1_c - e2_c, 'p1_clay': e1_c, 'p2_clay': e2_c,
        'diff_grass': e1_g - e2_g, 'p1_grass': e1_g, 'p2_grass': e2_g,
        'diff_carpet': e1_ca - e2_ca, 'p1_carpet': e1_ca, 'p2_carpet': e2_ca,
        'diff_streak': 0, 'p1_streak': 0, 'p2_streak': 0,
        'diff_str_log': 0.0, 'p1_str_log': 0.0, 'p2_str_log': 0.0,
        'akt_diff': e1_akt - e2_akt, 'akt_p1_elo': e1_akt, 'akt_p2_elo': e2_akt
    }])
    
    wynik = model.predict_proba(dane)[0]
    
    szansa_g1 = round(wynik[1] * 100, 2)
    szansa_g2 = round(wynik[0] * 100, 2)
    
    print(f"szanse {g1}: {szansa_g1}%")
    print(f"szanse {g2}: {szansa_g2}%")
    print()