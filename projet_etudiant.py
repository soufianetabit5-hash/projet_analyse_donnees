# %% [markdown]
# # PROJET : Analyse de données · régression linéaire, ANOVA, khi-2
# Nom, prénom : TABIT SOUFIANE   Groupe :2
# Bibliothèques : pandas, numpy, scipy, matplotlib, seaborn (statsmodels autorisé en option).
# Complétez les zones "# TODO". Commentez vos choix et interprétez TOUS vos résultats par écrit.
# %%
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from scipy import stats
sns.set_theme(style="whitegrid")
ALPHA = 0.05

# %% PARTIE 0 : chargement et nettoyage (2 pts)
df = pd.read_csv("donnees_projet.csv")
print("Projet lancé avec succès !")
print(df.shape)
print(df.head())
# TODO 0.1 : dimensions, types, valeurs manquantes par variable

print("Dimensions du dataset :")
print(df.shape)

print("\nTypes des variables :")
print(df.dtypes)

print("\nValeurs manquantes par variable :")
print(df.isnull().sum())

# TODO 0.2 : repérer une valeur aberrante dans heures_etude (describe, boxplot). Que faites-vous ? Justifiez.


print(df["heures_etude"].describe())

sns.boxplot(x=df["heures_etude"])
plt.title("Boxplot des heures d'étude")
plt.xlabel("Heures d'étude par semaine")
plt.show()

# TODO 0.3 : construire le tableau d'analyse "d" sans valeurs manquantes sur les variables utilisées


variables = [
    "heures_etude", "moyenne_bac", "note_stat",
    "filiere", "sexe", "logement", "assiduite"
]

d = df.dropna(subset=variables).copy()

print("Dimensions avant nettoyage :", df.shape)
print("Dimensions après nettoyage :", d.shape)


# %% PARTIE 1 : analyse descriptive (2 pts)
# TODO 1.1 : moyenne, écart-type, min, max de moyenne_bac, heures_etude, note_stat


colonnes = ["moyenne_bac", "heures_etude", "note_stat"]

print(d[colonnes].agg(["mean", "std", "min", "max"]).round(2))

# TODO 1.2 : moyenne de note_stat par filière ; effectifs de chaque modalité qualitative

#moyenne :
print(d.groupby("filiere")["note_stat"].mean().round(2))

# Effectifs des variables qualitatives

variables_qualitatives = [
    "filiere",
    "sexe",
    "logement",
    "assiduite",
    "reussite"
]

for variable in variables_qualitatives:
    print("\nEffectifs de :", variable)
    print(d[variable].value_counts())

# TODO 1.3 : 2 graphiques commentés (histogramme et boxplot par exemple)


# Graphique 1 : Histogramme des notes
plt.figure(figsize=(8, 5))

sns.histplot(data=d, x="note_stat", bins=15)

plt.title("Distribution des notes en statistique")
plt.xlabel("Note en statistique (/20)")
plt.ylabel("Nombre d'étudiants")

# Sauvegarde pour le rapport
plt.savefig("histogramme_notes.png", dpi=300, bbox_inches="tight")

plt.show()


# Graphique 2 : Boxplot des notes par filière
plt.figure(figsize=(8, 5))

sns.boxplot(data=d, x="filiere", y="note_stat")

plt.title("Notes en statistique selon la filière")
plt.xlabel("Filière")
plt.ylabel("Note en statistique (/20)")

# Sauvegarde pour le rapport
plt.savefig("boxplot_filieres.png", dpi=300, bbox_inches="tight")

plt.show()

# %% PARTIE 2 : régression linéaire simple, note_stat ~ heures_etude (6 pts)
# TODO 2.1 : nuage de points. La relation semble-t-elle linéaire ?

# TODO 2.1 : Nuage de points

plt.figure(figsize=(8, 5))

sns.scatterplot(data=d, x="heures_etude", y="note_stat")

plt.title("Relation entre heures d'étude et note en statistique")
plt.xlabel("Heures d'étude par semaine")
plt.ylabel("Note en statistique (/20)")

# Sauvegarde pour le rapport
plt.savefig("nuage_points_regression.png", dpi=300, bbox_inches="tight")

plt.show()

# TODO 2.2 : coefficient de corrélation r et son test


x = d["heures_etude"]
y = d["note_stat"]

# Calcul avec la formule
numerateur = ((x - x.mean()) * (y - y.mean())).sum()

denominateur = np.sqrt(
    ((x - x.mean())**2).sum()
    * ((y - y.mean())**2).sum()
)

r = numerateur / denominateur

print("Coefficient r =", round(r, 4))

# Verification avec scipy
resultat = stats.linregress(x, y)
print("Coefficient r scipy =", round(resultat.rvalue, 4))

# Test de significativite de la correlation

n = len(d)

# Calcul de la statistique t
t = r * np.sqrt((n - 2) / (1 - r**2))

# Calcul de la p-value (test bilateral)
p_value = 2 * stats.t.sf(abs(t), df=n - 2)

print("Statistique t :", round(t, 4))
print("p-value :", p_value)

# Comparaison avec scipy
print("p-value scipy :", resultat.pvalue)

# Decision statistique
if p_value < ALPHA:
    print("On rejette H0 : correlation significative")
else:
    print("On ne rejette pas H0")


# TODO 2.3 : estimer la droite (stats.linregress) ; écrire l'équation ; interpréter pente et ordonnée


resultat = stats.linregress(x, y)

a = resultat.slope
b = resultat.intercept

print("Pente a =", round(a, 4))
print("Ordonnee a l'origine b =", round(b, 4))

print(f"Equation : note_stat = {a:.4f} * heures_etude + {b:.4f}")

# TODO 2.4 : R², p-value de la pente, intervalle de confiance de la pente

# 1. Coefficient de determination
R2 = r**2

print("R2 =", round(R2, 4))
print("R2 scipy =", round(resultat.rvalue**2, 4))

# 2. Test de la pente
t_pente = a / resultat.stderr
p_pente = 2 * stats.t.sf(abs(t_pente), df=n-2)

print("p-value de la pente =", p_pente)
print("p-value scipy =", resultat.pvalue)

if p_pente < ALPHA:
    print("On rejette H0 : pente significative")
else:
    print("On ne rejette pas H0")

# 3. Intervalle de confiance a 95%
t_critique = stats.t.ppf(1 - ALPHA/2, df=n-2)

IC_inf = a - t_critique * resultat.stderr
IC_sup = a + t_critique * resultat.stderr

print("IC 95% de la pente :",
      round(IC_inf, 4), round(IC_sup, 4))

# TODO 2.5 : retrouver pente et ordonnée "à la main" (formules du cours) et comparer

# avec les formules du cours

# Calcul des moyennes
x_moy = x.mean()
y_moy = y.mean()

# Calcul de la pente
a_formule = (
    ((x - x_moy) * (y - y_moy)).sum()
    / ((x - x_moy)**2).sum()
)

# Calcul de l'ordonnee a l'origine
b_formule = y_moy - a_formule * x_moy

# Comparaison avec scipy
print("Pente avec formule :", round(a_formule, 4))
print("Pente scipy :", round(resultat.slope, 4))

print("Ordonnee avec formule :", round(b_formule, 4))
print("Ordonnee scipy :", round(resultat.intercept, 4))

# TODO 2.6 : prédire la note pour 10 h d'étude par semaine ; peut-on prédire pour 40 h ? pourquoi ?


# Prediction pour 10 heures d'etude
note_10h = a * 10 + b

print("Note predite pour 10h :", round(note_10h, 2))

# Calcul theorique pour 40 heures
note_40h = a * 40 + b

print("Note calculee pour 40h :", round(note_40h, 2))

# Intervalle des heures observees
print("Minimum :", d["heures_etude"].min())
print("Maximum :", d["heures_etude"].max())

# TODO 2.7 : résidus : graphique résidus vs ajustées, QQ-plot, test de Shapiro ; conditions respectées ?


# Calcul des valeurs ajustees et des residus du modele lineaire
valeurs_ajustees = a * x + b
residus = y - valeurs_ajustees

# QQ-plot des residus

plt.figure(figsize=(8, 5))

stats.probplot(residus, dist="norm", plot=plt)

plt.title("QQ-plot des residus")

# Sauvegarde pour le rapport
plt.savefig("qqplot_residus.png",
            dpi=300, bbox_inches="tight")

plt.show()


# BONUS 2.8 : régression multiple (heures_etude + moyenne_bac) avec scikit-learn ou statsmodels ; comparer les R²


from sklearn.linear_model import LinearRegression

# Variables explicatives
X_multiple = d[["heures_etude", "moyenne_bac"]]

# Variable a expliquer
y_multiple = d["note_stat"]

# Creation et ajustement du modele
modele_multiple = LinearRegression()
modele_multiple.fit(X_multiple, y_multiple)

# Coefficients
b0 = modele_multiple.intercept_
b1, b2 = modele_multiple.coef_

print("Ordonnee :", round(b0, 4))
print("Coefficient heures_etude :", round(b1, 4))
print("Coefficient moyenne_bac :", round(b2, 4))

# Comparaison des R2
R2_simple = resultat.rvalue**2
R2_multiple = modele_multiple.score(X_multiple, y_multiple)

print("R2 simple :", round(R2_simple, 4))
print("R2 multiple :", round(R2_multiple, 4))


# %% PARTIE 3 : ANOVA à un facteur, note_stat ~ filiere (5 pts)
# TODO 3.1 : H0 et H1 ; boxplot de la note par filière


# H0 : les moyennes des 3 filieres sont egales
# H1 : au moins une moyenne est differente
# Seuil alpha = 0.05

plt.figure(figsize=(8, 5))

sns.boxplot(data=d, x="filiere", y="note_stat")

plt.title("ANOVA : Notes selon la filiere")
plt.xlabel("Filiere")
plt.ylabel("Note en statistique (/20)")

# Sauvegarde pour le rapport
plt.savefig("anova_boxplot.png",
            dpi=300, bbox_inches="tight")

plt.show()

# TODO 3.2 : vérifier les conditions : normalité par groupe (Shapiro), égalité des variances (Levene)


# 1. Normalite par groupe : Shapiro-Wilk

bio = d[d["filiere"] == "Biologie"]["note_stat"]
chimie = d[d["filiere"] == "Chimie"]["note_stat"]
info = d[d["filiere"] == "Informatique"]["note_stat"]

groupes = {
    "Biologie": bio,
    "Chimie": chimie,
    "Informatique": info
}

for nom, notes in groupes.items():
    stat, p = stats.shapiro(notes)

    print(nom, ": p-value =", round(p, 4))

    if p < ALPHA:
        print("On rejette H0")
    else:
        print("On ne rejette pas H0")

# 2. Egalite des variances : Levene

stat_levene, p_levene = stats.levene(
    bio, chimie, info, center="median"
)

print("Statistique Levene :", round(stat_levene, 4))
print("p-value Levene :", round(p_levene, 4))

if p_levene < ALPHA:
    print("On rejette H0 : variances differentes")
else:
    print("On ne rejette pas H0")

# TODO 3.3 : calculer SC_entre, SC_intra, CM, F, ddl à la main puis comparer avec stats.f_oneway

# Groupes des notes
groupes = [
    d[d["filiere"] == "Biologie"]["note_stat"],
    d[d["filiere"] == "Chimie"]["note_stat"],
    d[d["filiere"] == "Informatique"]["note_stat"]
]

# Nombre total et nombre de groupes
N = len(d)
k = len(groupes)

# Moyenne generale
moyenne_generale = d["note_stat"].mean()

# Somme des carres entre les groupes
SC_entre = sum(
    len(g) * (g.mean() - moyenne_generale)**2
    for g in groupes
)

# Somme des carres dans les groupes
SC_intra = sum(
    ((g - g.mean())**2).sum()
    for g in groupes
)

# Degres de liberte
ddl_entre = k - 1
ddl_intra = N - k

# Carres moyens
CM_entre = SC_entre / ddl_entre
CM_intra = SC_intra / ddl_intra

# Statistique F
F_manuel = CM_entre / CM_intra

# Comparaison avec scipy
F_scipy, p_anova = stats.f_oneway(*groupes)

print("SC_entre =", round(SC_entre, 4))
print("SC_intra =", round(SC_intra, 4))
print("ddl_entre =", ddl_entre)
print("ddl_intra =", ddl_intra)
print("CM_entre =", round(CM_entre, 4))
print("CM_intra =", round(CM_intra, 4))
print("F manuel =", round(F_manuel, 4))
print("F scipy =", round(F_scipy, 4))
print("p-value =", p_anova)

# TODO 3.4 : F critique (stats.f.ppf) ; décision par comparaison F / F critique ET par la p-value

# Hypotheses :
# H0 : les moyennes des 3 filieres sont egales
# H1 : au moins une moyenne est differente

# Calcul de F critique
F_critique = stats.f.ppf(
    1 - ALPHA, ddl_entre, ddl_intra
)

print("F calcule =", round(F_manuel, 4))
print("F critique =", round(F_critique, 4))

# Decision avec F critique
if F_manuel > F_critique:
    print("Avec F : on rejette H0")
else:
    print("Avec F : on ne rejette pas H0")

# Decision avec la p-value
print("p-value =", p_anova)

if p_anova < ALPHA:
    print("Avec p-value : on rejette H0")
else:
    print("Avec p-value : on ne rejette pas H0")

# TODO 3.5 : taille d'effet η² ; interprétation


# Somme des carres totale
SC_total = SC_entre + SC_intra

# Calcul de eta carre
eta2 = SC_entre / SC_total

print("Eta carre =", round(eta2, 4))
print("Pourcentage explique =", round(eta2 * 100, 2), "%")

# TODO 3.6 : comparaisons par paires (stats.tukey_hsd) : quelles filières diffèrent ?


# Test de Tukey
tukey = stats.tukey_hsd(bio, chimie, info)

print("Resultats du test de Tukey :")
print(tukey)

# Noms des groupes dans le meme ordre
noms = ["Biologie", "Chimie", "Informatique"]

# Comparaison de chaque paire
for i in range(3):
    for j in range(i + 1, 3):
        p = tukey.pvalue[i, j]

        print(noms[i], "vs", noms[j])
        print("p-value =", round(p, 6))

        if p < ALPHA:
            print("On rejette H0 : difference significative")
        else:
            print("On ne rejette pas H0")

# BONUS 3.7 : test non paramétrique (stats.kruskal) ; conclusion identique ?

# H0 : les distributions sont identiques
# H1 : au moins une distribution est differente

H, p_kruskal = stats.kruskal(bio, chimie, info)

print("Statistique H =", round(H, 4))
print("p-value =", p_kruskal)

if p_kruskal < ALPHA:
    print("On rejette H0")
else:
    print("On ne rejette pas H0")

# %% PARTIE 4 : khi-2 d'indépendance (5 pts)
# Deux couples à tester : (logement, assiduite) et (sexe, filiere)
# Pour CHAQUE couple :
# TODO 4.1 : H0, H1 ; tableau croisé observé (pd.crosstab)

# Couple 1 : logement et assiduite
# H0 : logement et assiduite sont independants
# H1 : logement et assiduite sont dependants

tableau1 = pd.crosstab(d["logement"], d["assiduite"])

print("\nTableau observe : logement / assiduite")
print(tableau1)

# Couple 2 : sexe et filiere
# H0 : sexe et filiere sont independants
# H1 : sexe et filiere sont dependants

tableau2 = pd.crosstab(d["sexe"], d["filiere"])

print("\nTableau observe : sexe / filiere")
print(tableau2)

# TODO 4.2 : tableau des effectifs théoriques calculés avec la formule du cours (ligne x colonne / total)

def calcul_effectifs_theoriques(tableau):

    total = tableau.to_numpy().sum()

    E = pd.DataFrame(
        index=tableau.index,
        columns=tableau.columns,
        dtype=float
    )

    for ligne in tableau.index:
        for colonne in tableau.columns:

            total_ligne = tableau.loc[ligne].sum()
            total_colonne = tableau[colonne].sum()

            E.loc[ligne, colonne] = (
                total_ligne * total_colonne / total
            )

    return E


# Couple 1 : logement et assiduite
E1 = calcul_effectifs_theoriques(tableau1)

print("Effectifs theoriques : logement / assiduite")
print(E1.round(2))


# Couple 2 : sexe et filiere
E2 = calcul_effectifs_theoriques(tableau2)

print("\nEffectifs theoriques : sexe / filiere")
print(E2.round(2))

# TODO 4.3 : contribution de chaque cellule (O-E)²/E, puis le χ² total


# Couple 1 : logement et assiduite

contributions1 = (tableau1 - E1)**2 / E1

chi2_1 = contributions1.to_numpy().sum()

print("Contributions : logement / assiduite")
print(contributions1.round(4))

print("Khi-2 total 1 =", round(chi2_1, 4))


# Couple 2 : sexe et filiere

contributions2 = (tableau2 - E2)**2 / E2

chi2_2 = contributions2.to_numpy().sum()

print("\nContributions : sexe / filiere")
print(contributions2.round(4))

print("Khi-2 total 2 =", round(chi2_2, 4))

# TODO 4.4 : ddl, χ² critique, p-value ; comparer avec stats.chi2_contingency

# Comparaison avec scipy

couples = [
    ("Logement / Assiduite", tableau1, chi2_1),
    ("Sexe / Filiere", tableau2, chi2_2)
]

for nom, tableau, chi2_manuel in couples:

    print("\n---", nom, "---")

    # Degres de liberte
    L, C = tableau.shape
    ddl = (L - 1) * (C - 1)

    # Khi-2 critique
    chi2_critique = stats.chi2.ppf(1 - ALPHA, ddl)

    # p-value
    p_value = stats.chi2.sf(chi2_manuel, ddl)

    # Comparaison avec scipy
    chi2_scipy, p_scipy, ddl_scipy, E_scipy = (
        stats.chi2_contingency(tableau, correction=False)
    )

    print("ddl =", ddl)
    print("Khi-2 manuel =", round(chi2_manuel, 4))
    print("Khi-2 critique =", round(chi2_critique, 4))
    print("p-value =", p_value)

    print("Khi-2 scipy =", round(chi2_scipy, 4))
    print("p-value scipy =", p_scipy)

# TODO 4.5 : vérifier la condition E >= 5 ; décision ; conclusion en une phrase


# Decision et conclusion

couples = [
    ("Logement / Assiduite", E1, chi2_1, tableau1),
    ("Sexe / Filiere", E2, chi2_2, tableau2)
]

for nom, E, chi2, tableau in couples:

    print("\n---", nom, "---")

    # Verification de la condition
    print("Effectif theorique minimum :", E.min().min())

    condition = (E >= 5).all().all()

    print("Condition E >= 5 :", condition)

    # Calcul de la p-value
    L, C = tableau.shape
    ddl = (L - 1) * (C - 1)

    p = stats.chi2.sf(chi2, ddl)

    # Decision statistique
    if condition:
        if p < ALPHA:
            print("On rejette H0 : association significative")
        else:
            print("On ne rejette pas H0")
    else:
        print("Condition non respectee : test a revoir")

# TODO 4.6 : V de Cramér ; graphique en barres empilées en pourcentages

# Couple 1 : logement / assiduite
N1 = tableau1.to_numpy().sum()
L1, C1 = tableau1.shape

V1 = np.sqrt(chi2_1 / (N1 * min(L1 - 1, C1 - 1)))

print("V de Cramer logement / assiduite :", round(V1, 4))


# Couple 2 : sexe / filiere
N2 = tableau2.to_numpy().sum()
L2, C2 = tableau2.shape

V2 = np.sqrt(chi2_2 / (N2 * min(L2 - 1, C2 - 1)))

print("V de Cramer sexe / filiere :", round(V2, 4))

# Graphique 1 : logement / assiduite

pourcentages1 = tableau1.div(
    tableau1.sum(axis=1), axis=0
) * 100

ax = pourcentages1.plot(
    kind="bar", stacked=True, figsize=(8, 5)
)

plt.title("Assiduite selon le logement")
plt.xlabel("Logement")
plt.ylabel("Pourcentage (%)")
plt.ylim(0, 100)
plt.legend(title="Assiduite")
plt.xticks(rotation=0)
plt.tight_layout()

# Sauvegarde pour le rapport
plt.savefig("khi2_logement_assiduite.png",
            dpi=300, bbox_inches="tight")

plt.show()


# Graphique 2 : sexe / filiere

pourcentages2 = tableau2.div(
    tableau2.sum(axis=1), axis=0
) * 100

ax = pourcentages2.plot(
    kind="bar", stacked=True, figsize=(8, 5)
)

plt.title("Repartition des filieres selon le sexe")
plt.xlabel("Sexe")
plt.ylabel("Pourcentage (%)")
plt.ylim(0, 100)
plt.legend(title="Filiere")
plt.xticks(rotation=0)
plt.tight_layout()

# Sauvegarde pour le rapport
plt.savefig("khi2_sexe_filiere.png",
            dpi=300, bbox_inches="tight")

plt.show()


# Question de réflexion : pourquoi un khi-2 significatif ne prouve-t-il pas une relation de cause à effet ?

# %% PARTIE 5 : synthèse (clarté, 0 pt ici, évalué dans le rapport)
# Rédigez dans le rapport une conclusion de 10 lignes reliant les trois méthodes.

