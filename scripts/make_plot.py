import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df = pd.DataFrame(
    {
        "Factor": [
            "sines in white",
            "sines in pink",
            "filtered pink",
            "state space model white",
            "state space model pink",
        ],
        "Pink KF": [1.44, 4.5, 15.26, 35.77, 56.07],
        "White KF": [2.0, 13.68, 22.43, 35.29, 95.5],
    }
)
tidy = df.melt(id_vars="Factor").rename(columns=str.title)
tt = tidy.rename(
    columns={"Value": "circstd", "Factor": "Simulation type", "Variable": "Algorithm"}
)
sns.barplot(data=tt, x="Simulation type", hue="Algorithm", y="circstd")
plt.grid()
plt.show()
