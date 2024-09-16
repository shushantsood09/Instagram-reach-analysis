import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import nltk
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler


data = pd.read_csv("Instagram_data.csv", encoding = 'latin1')
print(data.head())
data.isnull().sum()
data = data.dropna()
data.info()
numeric_features = data[['Likes', 'Comments', 'Saves', 'Shares', 'From Home', 'From Hashtags', 'From Explore', 'From Other', 'Profile Visits', 'Follows']]

correlation_with_impressions = numeric_features.corrwith(data['Impressions'])
strong_correlations = correlation_with_impressions[abs(correlation_with_impressions) > 0.5]


print("Correlation with Impressions (Numeric Features):\n", correlation_with_impressions)
print("\nFactors with Strong Impact on Post Reach:\n", strong_correlations)

plt.figure(figsize=(7, 5))
sns.heatmap(numeric_features.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap for Numeric Features')
plt.show()


engagement_stats = data[['Likes', 'Comments', 'Saves', 'Shares', 'From Home', 'From Hashtags', 'From Explore', 'From Other', 'Profile Visits', 'Follows']].describe()
print("Engagement Statistics:\n", engagement_stats)


text = " ".join(i for i in data.Caption)
stopwords = set(STOPWORDS)
wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)
plt.style.use('classic')
plt.figure( figsize=(12,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()



text = " ".join(i for i in data.Hashtags)
stopwords = set(STOPWORDS)
wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)
plt.figure( figsize=(12,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

#  Feature Engineering
data['Engagement_Rate'] = (data['Likes'] + data['Comments'] + data['Saves']) / data['Impressions']

#  Machine Learning Predictive Models
# Build Predictive Models
X = data[['Likes', 'Comments', 'Saves', 'Engagement_Rate' ]]
y = data['Impressions']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
model = RandomForestRegressor(random_state=42)
model.fit(X_train_scaled, y_train)
# Evaluate Model Performance
y_pred = model.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Mean Absolute Error:", mae)
print("Root Mean Squared Error:", rmse)