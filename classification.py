import pandas as pd
#pd.set_option('display.max_colwidth', None)

data = pd.read_csv('10yr_normalized_better.csv')
data["Date"] = pd.to_datetime(data["Date"])
data = data.rename(columns={"Date":"date"})
ranks = pd.concat([data["Z_Score"],data["Z_Score"].rank(pct=True)],axis=1).set_axis(["Z_Score", "Perc_Rank"], axis=1).sort_values("Perc_Rank")
bottom_cutoff, top_cutoff = (ranks.query(f"Perc_Rank < 0.25").tail(1).Z_Score.iloc[0],
                                 ranks.query(f"Perc_Rank > 0.75").head(1).Z_Score.iloc[0])
data["Class"] = ["H" if i>top_cutoff else "N" if i<bottom_cutoff else "D" for i in data["Z_Score"]]
print(data["Class"])
speeches = pd.read_csv("speeches.csv")
speeches["date"] = pd.to_datetime(speeches["date"])
final_data = {"date":[],"speech":[],"classif":[]}
no_sent = 0
for d in speeches["date"]:
    if not d in final_data["date"]:
        diff = abs(data["date"]-d)
        print(d,data.loc[diff==diff.min()]["date"])
        results = data.loc[diff==diff.min()]["Class"]
        final_data["date"].append(d)
        final_data["speech"].append(speeches.loc[speeches["date"] == d]["text"].iloc[0])
        final_data["classif"].append(data.loc[diff==diff.min()]["Class"].iloc[0])
final_data = pd.DataFrame(final_data)

final_data.to_csv("classifications.csv")